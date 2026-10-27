use ffmpeg_next as ffmpeg;
use ffmpeg::{Rescale, codec::packet::Mut};

use std::collections::VecDeque;
use std::io::{Seek, SeekFrom, Write};
use std::ops::Range;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::decoder::FrameDecoder;
use crate::media::{Clip, Content, CursorRead, Encoded, Export, PacketSource, Track};
use crate::{Audio, AudioFrame, Cursor, Image, Media, Pixel, Sample, Sound, Video, VideoFrame};

/// Supported source and export container/codec combinations.
/// Matching encoded packets are reused; bitrate applies when encoding is necessary.
/// MPEG-4 bitrate control uses the first video frame's duration as the nominal frame interval.
/// FFV1 records native slice error status when encoding corrupted video frames.
/// Known corrupted intervals also use media2 comment metadata; other software may discard it.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Format {
    /// Video bitrate in bits per second.
    Mp4_Mpeg4p2_Aac {bitrate: usize},
    /// Video bitrate in bits per second.
    Mp4_Mpeg4p2 {bitrate: usize},
    Mkv_Ffv1_Flac,
    Mkv_Ffv1_PcmF32le,
    Mkv_Ffv1,
    /// Audio bitrate in bits per second, from 6,000 through 255,000.
    Mkv_Opus {bitrate: usize},
    Mov_ProRes422_PcmS16le,
    Mov_ProRes422,
    M4a_Aac,
    M4a_Alac,
    Wav_PcmS16le,
    Wav_PcmF32le,
    Flac_Flac,
}
impl Format {
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Mp4_Mpeg4p2_Aac {..} | Self::Mp4_Mpeg4p2 {..} => "mp4",
            Self::Mkv_Ffv1_Flac | Self::Mkv_Ffv1_PcmF32le | Self::Mkv_Ffv1
                | Self::Mkv_Opus {..} => "mkv",
            Self::Mov_ProRes422_PcmS16le | Self::Mov_ProRes422 => "mov",
            Self::M4a_Aac | Self::M4a_Alac => "m4a",
            Self::Wav_PcmS16le | Self::Wav_PcmF32le => "wav",
            Self::Flac_Flac => "flac",
        }
    }
    pub fn video(&self) -> bool {self.video_codec().is_some()}
    pub fn audio(&self) -> bool {self.audio_codec().is_some()}
    fn container(&self) -> &'static str {
        match self {
            Self::Mp4_Mpeg4p2_Aac {..} | Self::Mp4_Mpeg4p2 {..} => "mp4",
            Self::Mkv_Ffv1_Flac | Self::Mkv_Ffv1_PcmF32le | Self::Mkv_Ffv1
                | Self::Mkv_Opus {..} => "matroska",
            Self::Mov_ProRes422_PcmS16le | Self::Mov_ProRes422 => "mov",
            Self::M4a_Aac | Self::M4a_Alac => "ipod",
            Self::Wav_PcmS16le | Self::Wav_PcmF32le => "wav",
            Self::Flac_Flac => "flac",
        }
    }
    fn video_codec(&self) -> Option<ffmpeg::codec::Id> {
        match self {
            Self::Mp4_Mpeg4p2_Aac {..} | Self::Mp4_Mpeg4p2 {..} => Some(ffmpeg::codec::Id::MPEG4),
            Self::Mkv_Ffv1_Flac | Self::Mkv_Ffv1_PcmF32le | Self::Mkv_Ffv1 => {
                Some(ffmpeg::codec::Id::FFV1)
            }
            Self::Mov_ProRes422_PcmS16le | Self::Mov_ProRes422 => Some(ffmpeg::codec::Id::PRORES),
            _ => None,
        }
    }
    fn audio_codec(&self) -> Option<ffmpeg::codec::Id> {
        match self {
            Self::Mp4_Mpeg4p2_Aac {..} | Self::M4a_Aac => Some(ffmpeg::codec::Id::AAC),
            Self::Mkv_Ffv1_Flac | Self::Flac_Flac => Some(ffmpeg::codec::Id::FLAC),
            Self::Mkv_Ffv1_PcmF32le | Self::Wav_PcmF32le => Some(ffmpeg::codec::Id::PCM_F32LE),
            Self::Mov_ProRes422_PcmS16le | Self::Wav_PcmS16le => Some(ffmpeg::codec::Id::PCM_S16LE),
            Self::M4a_Alac => Some(ffmpeg::codec::Id::ALAC),
            Self::Mkv_Opus {..} => Some(ffmpeg::codec::Id::OPUS),
            _ => None,
        }
    }
}

/// A persistent container encoder. Each iterator item continues the same byte stream.
/// Chunks are released automatically; their boundaries are not independently playable files.
/// Reused packets preserve their original damage. Re-encoded FFV1 uses native slice error status.
/// Formats without a native marker encode the decoder's silence or black replacements,
/// with interpolation and held images for sufficiently short gaps.
pub struct Encoder {
    output: ffmpeg::format::context::Output,
    bytes: Option<Arc<Mutex<Vec<u8>>>>,
    media: Media,
    cursor: Cursor,
    format: Format,
    video: Option<ffmpeg::encoder::Video>,
    audio: Option<ffmpeg::encoder::Audio>,
    video_copy: Option<Copy>,
    audio_copy: Option<Copy>,
    video_stream: Option<usize>,
    audio_stream: Option<usize>,
    resampler: Option<ffmpeg::software::resampling::Context>,
    samples: *mut ffmpeg::ffi::AVAudioFifo,
    video_position: i64,
    video_timing: VecDeque<(Duration, Duration, bool)>,
    audio_position: i64,
    audio_end: Duration,
    video_end: Duration,
    audio_frames: VecDeque<Arc<AudioFrame>>,
    video_frames: VecDeque<Arc<VideoFrame>>,
    audio_cursor: Duration,
    video_cursor: Duration,
    video_seen: Duration,
    pending_audio: Option<Arc<AudioFrame>>,
    pending_video: Option<Arc<VideoFrame>>,
    empty_audio: bool,
    finished: bool,
}
impl Encoder {
    pub(crate) fn new(audio: Audio, video: Video, format: Format) -> Result<Self, String> {
        let bytes = Arc::new(Mutex::new(Vec::new()));
        let io = ffmpeg::format::context::StreamIo::from_write(Output(bytes.clone()))
            .map_err(|error| error.to_string())?;
        let mut encoder = Self::open(Media {audio, video}, format, io, Some(bytes), true)
            .map_err(|error| error.to_string())?;
        encoder.flush().map_err(|error| error.to_string())?;
        Ok(encoder)
    }

    fn open(
        media: Media, format: Format, io: ffmpeg::format::context::StreamIo,
        bytes: Option<Arc<Mutex<Vec<u8>>>>, streaming: bool,
    ) -> Result<Self, ffmpeg::Error> {
        FrameDecoder::init()?;
        if let Format::Mp4_Mpeg4p2 {bitrate} | Format::Mp4_Mpeg4p2_Aac {bitrate} = format {
            assert!(bitrate > 0);
        }
        if let Format::Mkv_Opus {bitrate} = format {assert!((6_000..=255_000).contains(&bitrate));}
        let video_copy = Copy::new(&media.video.clips(), format)?;
        let audio_copy = Copy::new(&media.audio.clips(), format)?;
        let mut cursor = Cursor::from(&media);
        let mut audio_cursor = media.audio.clips().first().map_or(Duration::ZERO, |clip| clip.start);
        let mut video_cursor = media.video.clips().first().map_or(Duration::ZERO, |clip| clip.start);
        let mut audio_frames = VecDeque::new();
        let mut video_frames = VecDeque::new();
        while format.audio() && audio_copy.is_none() && audio_cursor < media.audio.duration() {
            let end = (audio_cursor + Duration::from_millis(100)).min(media.audio.duration());
            audio_frames.extend(cursor.audio_frames(audio_cursor..end, Duration::from_millis(100))
                .iter().cloned());
            audio_cursor = end;
            if !audio_frames.is_empty() {break}
        }
        while format.video() && video_copy.is_none() && video_cursor < media.video.duration() {
            let end = (video_cursor + Duration::from_millis(100)).min(media.video.duration());
            video_frames.extend(cursor.video_frames(video_cursor..end).iter().cloned());
            video_cursor = end;
            if !video_frames.is_empty() {break}
        }
        let first_audio = audio_frames.pop_front();
        let first_video = video_frames.pop_front();
        let video_seen = first_video.as_ref().map_or(Duration::ZERO, |frame| {
            frame.timestamp + frame.duration
        });
        let sound = first_audio.as_ref().map(|frame| (frame.sample_rate(), frame.sound.channels));
        let empty = format.audio() && first_audio.is_none() && audio_copy.is_none();
        let sound = if empty && first_video.is_none() && video_copy.is_none() {
            Some((48_000, 2))
        } else {sound};
        if sound.is_none() && first_video.is_none() && audio_copy.is_none() && video_copy.is_none() {
            return Err(ffmpeg::Error::StreamNotFound)
        }
        let audio = sound;
        let video = first_video.as_deref();
        let filename = format!("media.{}", format.extension());
        let mut output = ffmpeg::format::output_to_stream(
            io, Some(&filename), Some(format.container()),
        )?;
        let comment = Self::corrupted(&media, format);
        if let Some(comment) = comment {
            let mut metadata = ffmpeg::Dictionary::new();
            metadata.set("comment", &comment);
            output.set_metadata(metadata);
        }
        let global = output.format().flags().contains(ffmpeg::format::Flags::GLOBAL_HEADER);
        let video = if let Some(copy) = &video_copy {
            copy.add_stream(&mut output)?;
            None
        } else if let Some((frame, id)) = video.zip(format.video_codec()) {
            let (width, height) = frame.image.dimensions;
            let codec = if id == ffmpeg::codec::Id::PRORES {
                ffmpeg::encoder::find_by_name("prores_ks")
            } else {ffmpeg::encoder::find(id)}.ok_or(ffmpeg::Error::EncoderNotFound)?;
            let mut encoder = ffmpeg::codec::context::Context::new_with_codec(codec).encoder().video()?;
            encoder.set_threading(ffmpeg::threading::Config::count(1));
            encoder.set_width(width);
            encoder.set_height(height);
            let (pixels, ticks) = match id {
                ffmpeg::codec::Id::MPEG4 => (ffmpeg::format::Pixel::YUV420P, 60_000),
                ffmpeg::codec::Id::PRORES => (ffmpeg::format::Pixel::YUV422P10LE, 60_000),
                _ => (ffmpeg::format::Pixel::BGRA, 1_000_000),
            };
            encoder.set_format(pixels);
            encoder.set_time_base((1, ticks));
            encoder.set_max_b_frames(0);
            let mut flags = ffmpeg::codec::Flags::from_bits_retain(
                ffmpeg::ffi::AV_CODEC_FLAG_FRAME_DURATION,
            );
            if global {flags |= ffmpeg::codec::Flags::GLOBAL_HEADER;}
            if let Format::Mp4_Mpeg4p2_Aac {bitrate} | Format::Mp4_Mpeg4p2 {bitrate} = format {
                assert!(bitrate > 0);
                encoder.set_bit_rate(bitrate);
                // MPEG-4 rate control requires a frame clock; muxed timing is restored per packet.
                let clock = unsafe {ffmpeg::ffi::av_d2q(frame.duration.as_secs_f64(), 65_535)};
                assert!(clock.num > 0 && clock.den > 0);
                encoder.set_time_base(clock);
            }
            encoder.set_flags(flags);
            let mut options = ffmpeg::Dictionary::new();
            if id == ffmpeg::codec::Id::PRORES {options.set("profile", "2");}
            if id == ffmpeg::codec::Id::FFV1 {
                options.set("level", "3");
                options.set("slicecrc", "1");
            }
            let encoder = encoder.open_as_with(codec, options)?;
            let mut stream = output.add_stream(codec)?;
            stream.set_parameters(&encoder);
            stream.set_time_base((1, ticks));
            let rate = unsafe {ffmpeg::ffi::av_d2q(1.0 / frame.duration.as_secs_f64(), i32::MAX)};
            stream.set_avg_frame_rate(rate);
            stream.set_rate(rate);
            Some(encoder)
        } else {None};
        let video_stream = (video.is_some() || video_copy.is_some()).then_some(0);
        let audio = if let Some(copy) = &audio_copy {
            copy.add_stream(&mut output)?;
            None
        } else if let Some(((rate, channels), id)) = audio.zip(format.audio_codec()) {
            let codec = if id == ffmpeg::codec::Id::OPUS {
                ffmpeg::encoder::find_by_name("opus")
            } else {ffmpeg::encoder::find(id)}.ok_or(ffmpeg::Error::EncoderNotFound)?;
            let supported = codec.audio()?;
            let rate = supported.rates().and_then(|rates| {
                rates.min_by_key(|candidate| i64::from(*candidate).abs_diff(i64::from(rate)))
            }).unwrap_or(rate as i32);
            let layout = if id == ffmpeg::codec::Id::OPUS {
                ffmpeg::ChannelLayout::default(i32::from(channels.min(2)))
            } else {
                supported.channel_layouts().map_or_else(
                    || ffmpeg::ChannelLayout::default(i32::from(channels)),
                    |layouts| layouts.best(i32::from(channels)),
                )
            };
            let packed = ffmpeg::format::sample::Type::Packed;
            let planar = ffmpeg::format::sample::Type::Planar;
            let samples = match id {
                ffmpeg::codec::Id::AAC | ffmpeg::codec::Id::OPUS => ffmpeg::format::Sample::F32(planar),
                ffmpeg::codec::Id::ALAC => ffmpeg::format::Sample::I32(planar),
                ffmpeg::codec::Id::FLAC => ffmpeg::format::Sample::I32(packed),
                ffmpeg::codec::Id::PCM_S16LE => ffmpeg::format::Sample::I16(packed),
                _ => ffmpeg::format::Sample::F32(packed),
            };
            let mut encoder = ffmpeg::codec::context::Context::new_with_codec(codec).encoder().audio()?;
            encoder.set_threading(ffmpeg::threading::Config::count(1));
            encoder.set_rate(rate);
            encoder.set_channel_layout(layout);
            encoder.set_format(samples);
            encoder.set_time_base((1, rate));
            if id == ffmpeg::codec::Id::AAC {encoder.set_bit_rate(96_000 * layout.channels() as usize);}
            if matches!(id, ffmpeg::codec::Id::FLAC | ffmpeg::codec::Id::ALAC) {
                unsafe {(*encoder.as_mut_ptr()).bits_per_raw_sample = 24;}
            }
            if global {encoder.set_flags(ffmpeg::codec::Flags::GLOBAL_HEADER);}
            let mut options = ffmpeg::Dictionary::new();
            if let Format::Mkv_Opus {bitrate} = format {
                encoder.compliance(ffmpeg::codec::Compliance::Experimental);
                encoder.set_bit_rate(bitrate);
                options.set("opus_delay", "20");
            }
            let encoder = encoder.open_as_with(codec, options)?;
            let mut stream = output.add_stream(codec)?;
            stream.set_parameters(&encoder);
            stream.set_time_base((1, rate));
            Some(encoder)
        } else {None};
        let audio_stream = (audio.is_some() || audio_copy.is_some())
            .then_some(usize::from(video_stream.is_some()));
        let mut options = ffmpeg::Dictionary::new();
        if streaming {
            match format.extension() {
                "mp4" | "mov" | "m4a" => {
                    options.set("movflags", "delay_moov+default_base_moof+skip_trailer");
                    options.set("frag_duration", "100000");
                    options.set("frag_size", "65536");
                    options.set("use_editlist", "1");
                }
                "mkv" => {
                    options.set("cluster_time_limit", "100");
                    options.set("cluster_size_limit", "65536");
                    options.set("live", "1");
                }
                _ => {},
            }
            options.set("flush_packets", "1");
        } else if empty && matches!(format, Format::M4a_Aac | Format::M4a_Alac) {
            options.set("movflags", "empty_moov");
            options.set("use_editlist", "0");
        }
        output.write_header_with(options)?;
        if empty && matches!(format, Format::Wav_PcmS16le | Format::Wav_PcmF32le) {
            let mut packet = ffmpeg::Packet::new(0);
            packet.set_stream(0);
            packet.set_pts(Some(0));
            packet.set_dts(Some(0));
            // Initialize WAV timing without adding samples; the wrapper rejects empty packets.
            let result = unsafe {
                ffmpeg::ffi::av_interleaved_write_frame(output.as_mut_ptr(), packet.as_mut_ptr())
            };
            if result < 0 {return Err(ffmpeg::Error::from(result))}
        }
        let samples = audio.as_ref().map_or(std::ptr::null_mut(), |encoder| unsafe {
            let samples = ffmpeg::ffi::av_audio_fifo_alloc(
                encoder.format().into(), i32::from(encoder.channels()), 1,
            );
            assert!(!samples.is_null());
            samples
        });
        Ok(Self {
            output, bytes, media, cursor, format, video, audio, samples, resampler: None,
            video_copy, audio_copy, video_stream, audio_stream,
            video_position: 0, video_timing: VecDeque::new(),
            audio_position: 0, audio_end: Duration::ZERO, video_end: Duration::ZERO,
            audio_frames, video_frames, audio_cursor, video_cursor, video_seen,
            pending_audio: first_audio, pending_video: first_video,
            empty_audio: empty, finished: false,
        })
    }

    fn step(&mut self) -> Result<(), ffmpeg::Error> {
        if self.pending_audio.is_none() && self.format.audio() && self.audio_copy.is_none() {
            loop {
                if let Some(frame) = self.audio_frames.pop_front() {
                    self.pending_audio = Some(frame);
                    break
                }
                if self.audio_cursor >= self.media.audio.duration() {break}
                let end = (self.audio_cursor + Duration::from_millis(100))
                    .min(self.media.audio.duration());
                self.audio_frames.extend(self.cursor.audio_frames(
                    self.audio_cursor..end, Duration::from_millis(100),
                ).iter().cloned());
                self.audio_cursor = end;
            }
        }
        if self.pending_video.is_none() && self.format.video() && self.video_copy.is_none() {
            loop {
                if let Some(frame) = self.video_frames.pop_front() {
                    let end = frame.timestamp + frame.duration;
                    if end <= self.video_seen {continue}
                    self.video_seen = end;
                    self.pending_video = Some(frame);
                    break
                }
                self.video_cursor = self.video_cursor.max(self.video_seen);
                if self.video_cursor >= self.media.video.duration() {break}
                let end = (self.video_cursor + Duration::from_millis(100))
                    .min(self.media.video.duration());
                self.video_frames.extend(self.cursor.video_frames(self.video_cursor..end)
                    .iter().cloned());
                self.video_cursor = end;
            }
        }
        let duration = self.media.duration();
        let audio_time = self.pending_audio.as_ref().map(|frame| frame.timestamp);
        let audio_time = if self.audio.is_some() && !self.empty_audio && self.audio_end < duration {
            Some(audio_time.unwrap_or(duration).min(self.audio_end))
        } else {audio_time};
        let audio_time = self.audio_copy.as_ref().map_or(audio_time, Copy::time);
        let video_time = self.pending_video.as_ref().map(|frame| frame.timestamp);
        let video_time = if self.video.is_some() && self.video_end < duration {
            Some(video_time.unwrap_or(duration).min(self.video_end))
        } else {video_time};
        let video_time = self.video_copy.as_ref().map_or(video_time, Copy::time);
        match (audio_time, video_time) {
            (None, None) => return self.finish(),
            (Some(audio), video) if video.is_none_or(|video| audio <= video) => {
                if let Some(copy) = &mut self.audio_copy {
                    copy.write(&mut self.output, self.audio_stream.unwrap())?;
                    return self.flush()
                }
                let next = self.pending_audio.as_ref().map_or(duration, |frame| frame.timestamp);
                if self.audio_end < next {
                    let end = next.min(self.audio_end + Duration::from_millis(100));
                    self.silence(end)?;
                } else {
                    let frame = self.pending_audio.take().unwrap();
                    self.audio(&frame)?;
                }
            }
            _ => {
                if let Some(copy) = &mut self.video_copy {
                    copy.write(&mut self.output, self.video_stream.unwrap())?;
                    return self.flush()
                }
                let next = self.pending_video.as_ref().map_or(duration, |frame| frame.timestamp);
                if self.video_end < next {
                    let dimensions = self.media.video.dimensions().unwrap();
                    let count = dimensions.0 as usize * dimensions.1 as usize;
                    self.video(&VideoFrame {
                        image: Image::new(dimensions, vec![Pixel::default(); count]),
                        timestamp: self.video_end, duration: next - self.video_end, corrupted: false,
                    })?;
                    self.video_end = next;
                } else {
                    let frame = self.pending_video.take().unwrap();
                    self.video(&frame)?;
                    self.video_end = frame.timestamp + frame.duration;
                }
            }
        }
        self.flush()
    }
    fn corrupted(media: &Media, format: Format) -> Option<String> {
        let mut corrupted: [Vec<Range<Duration>>; 2] = Default::default();
        for (index, clips) in [media.audio.clips(), media.video.clips()].into_iter().enumerate() {
            if (index == 0 && !format.audio()) || (index == 1 && !format.video()) {continue}
            for clip in clips.iter() {
                let raw = match &clip.data {
                    Content::Audio(frame) => frame.corrupted,
                    Content::Video(frame) => frame.corrupted,
                    Content::Encoded {..} => false,
                };
                let ranges = if raw {vec![clip.start..clip.end]} else {
                    clip.regions(clip.start..clip.end).into_iter().filter_map(|(range, corrupted)| {
                        corrupted.then_some(range)
                    }).collect()
                };
                for range in ranges {
                    let ranges = &mut corrupted[index];
                    if let Some(last) = ranges.last_mut().filter(|last| last.end >= range.start) {
                        last.end = last.end.max(range.end);
                    } else {ranges.push(range);}
                }
            }
        }
        if corrupted.iter().all(Vec::is_empty) {return None}
        let mut comment = String::from("media2-corrupted-v1");
        for (name, ranges) in ["a", "v"].into_iter().zip(corrupted) {
            comment.push_str(&format!(";{name}="));
            for (index, range) in ranges.iter().enumerate() {
                if index > 0 {comment.push(',');}
                comment.push_str(&format!("{}-{}", range.start.as_nanos(), range.end.as_nanos()));
            }
        }
        Some(comment)
    }
    fn video(&mut self, frame: &VideoFrame) -> Result<(), ffmpeg::Error> {
        let encoder = self.video.as_mut().ok_or(ffmpeg::Error::InvalidData)?;
        let image = &frame.image;
        let (width, height) = image.dimensions;
        assert!(width > 0 && height > 0 && !frame.duration.is_zero());
        assert_eq!(image.len(), width as usize * height as usize);
        let resized;
        let pixels = if image.dimensions == (encoder.width(), encoder.height()) {image.as_ref()} else {
            let rgba = image::RgbaImage::from_raw(width, height, image.as_ref().to_vec()).unwrap();
            resized = image::imageops::resize(
                &rgba, encoder.width(), encoder.height(), image::imageops::FilterType::Triangle,
            );
            resized.as_raw().as_slice()
        };
        let (width, height) = (encoder.width(), encoder.height());
        let mut rgba = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::RGBA, width, height);
        let stride = rgba.stride(0);
        for (y, row) in pixels.chunks_exact(width as usize * 4).enumerate() {
            rgba.data_mut(0)[y * stride..y * stride + row.len()].copy_from_slice(row);
        }
        let mut raw = ffmpeg::frame::Video::empty();
        ffmpeg::software::scaling::Context::get(
            ffmpeg::format::Pixel::RGBA, width, height, encoder.format(), width, height,
            ffmpeg::software::scaling::flag::Flags::BILINEAR,
        )?.run(&rgba, &mut raw)?;
        if matches!(encoder.id(), ffmpeg::codec::Id::MPEG4 | ffmpeg::codec::Id::FFV1) {
            self.video_timing.push_back((frame.timestamp, frame.duration, frame.corrupted));
        }
        let (position, duration) = if encoder.id() == ffmpeg::codec::Id::MPEG4 {
            let position = self.video_position;
            self.video_position += 1;
            (position, 1)
        } else {
            let position = i64::try_from(frame.timestamp.as_nanos()).unwrap()
                .rescale((1, 1_000_000_000), encoder.time_base());
            let duration = i64::try_from(frame.duration.as_nanos()).unwrap()
                .rescale((1, 1_000_000_000), encoder.time_base());
            (position, duration)
        };
        raw.set_pts(Some(position));
        unsafe {(*raw.as_mut_ptr()).duration = duration;}
        encoder.send_frame(&raw)?;
        self.drain(0)
    }
    fn audio(&mut self, frame: &AudioFrame) -> Result<(), ffmpeg::Error> {
        let sample_rate = frame.sample_rate();
        assert!(sample_rate > 0 && frame.sound.samples.iter().all(|sample| sample.0.is_finite()));
        self.silence(frame.timestamp)?;
        let format = ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Packed);
        let layout = ffmpeg::ChannelLayout::default(i32::from(frame.sound.channels));
        if self.resampler.as_ref().is_some_and(|resampler| {
            resampler.input().rate != sample_rate || resampler.input().channel_layout != layout
        }) {self.flush_resampler()?;}
        let encoder = self.audio.as_ref().ok_or(ffmpeg::Error::InvalidData)?;
        if self.resampler.is_none() {
            self.resampler = Some(ffmpeg::software::resampling::Context::get(
                format, layout, sample_rate,
                encoder.format(), encoder.channel_layout(), encoder.rate(),
            )?);
        }
        let count = frame.sound.samples.len() / frame.sound.channels as usize;
        let mut input = ffmpeg::frame::Audio::new(format, count, layout);
        input.set_rate(sample_rate);
        input.plane_mut::<f32>(0).copy_from_slice(frame.sound.as_ref());
        let resampler = self.resampler.as_mut().unwrap();
        let delay = resampler.delay().map_or(0, |delay| delay.output.max(0) as usize);
        let capacity = (count as f64 * encoder.rate() as f64 / sample_rate as f64).ceil();
        let mut converted = ffmpeg::frame::Audio::new(
            encoder.format(), capacity as usize + delay + 32, encoder.channel_layout(),
        );
        resampler.run(&input, &mut converted)?;
        self.audio_end = frame.timestamp + frame.duration;
        self.queue_audio(&mut converted)
    }
    fn silence(&mut self, end: Duration) -> Result<(), ffmpeg::Error> {
        let encoder = self.audio.as_ref().ok_or(ffmpeg::Error::InvalidData)?;
        let (sample_rate, channels) = (encoder.rate(), encoder.channels());
        loop {
            let remaining = end.saturating_sub(self.audio_end).as_secs_f64();
            let count = (remaining * sample_rate as f64).round() as usize;
            if count == 0 {self.audio_end = end; break}
            let count = count.min(sample_rate as usize);
            self.audio(&AudioFrame {
                timestamp: self.audio_end,
                duration: Duration::from_secs_f64(count as f64 / sample_rate as f64),
                sound: Sound {channels, samples: vec![Sample(0.0); count * channels as usize]},
                corrupted: false,
            })?;
        }
        Ok(())
    }
    fn finish(&mut self) -> Result<(), ffmpeg::Error> {
        self.flush_resampler()?;
        self.encode_audio(true)?;
        if let Some(encoder) = &mut self.video {
            encoder.send_eof()?;
            self.drain(0)?;
        }
        if let Some(encoder) = &mut self.audio {
            encoder.send_eof()?;
            self.drain(self.audio_stream.unwrap())?;
        }
        assert!(self.video_timing.is_empty());
        if self.empty_audio && self.video_stream.is_none() && self.format.extension() == "mkv" {
            // An empty Cluster completes the container without adding audio samples.
            let cluster = [0x1f, 0x43, 0xb6, 0x75, 0x83, 0xe7, 0x81, 0x00];
            unsafe {
                ffmpeg::ffi::avio_write(
                    (*self.output.as_mut_ptr()).pb, cluster.as_ptr(), cluster.len() as i32,
                );
            }
        }
        self.output.write_trailer()?;
        self.flush()?;
        self.finished = true;
        Ok(())
    }
    fn flush(&mut self) -> Result<(), ffmpeg::Error> {
        let result = unsafe {
            let io = (*self.output.as_mut_ptr()).pb;
            ffmpeg::ffi::avio_flush(io);
            (*io).error
        };
        if result < 0 {return Err(ffmpeg::Error::from(result))}
        Ok(())
    }
    fn flush_resampler(&mut self) -> Result<(), ffmpeg::Error> {
        let Some(mut resampler) = self.resampler.take() else {return Ok(())};
        let encoder = self.audio.as_ref().unwrap();
        let (format, rate, layout) = (encoder.format(), encoder.rate(), encoder.channel_layout());
        let end = (self.audio_end.as_secs_f64() * rate as f64).round() as i64;
        loop {
            let delay = resampler.delay().map_or(0, |delay| delay.output.max(0) as usize);
            let mut converted = ffmpeg::frame::Audio::new(format, delay + 32, layout);
            resampler.flush(&mut converted)?;
            if converted.samples() == 0 {break}
            let queued = unsafe {ffmpeg::ffi::av_audio_fifo_size(self.samples)};
            let remaining = (end - self.audio_position - i64::from(queued)).max(0) as usize;
            converted.set_samples(converted.samples().min(remaining));
            self.queue_audio(&mut converted)?;
        }
        Ok(())
    }
    fn queue_audio(&mut self, frame: &mut ffmpeg::frame::Audio) -> Result<(), ffmpeg::Error> {
        if frame.samples() == 0 {return Ok(())}
        let written = unsafe {
            ffmpeg::ffi::av_audio_fifo_write(
                self.samples, (*frame.as_mut_ptr()).extended_data.cast(), frame.samples() as i32,
            )
        };
        if written < 0 {return Err(ffmpeg::Error::from(written))}
        self.encode_audio(false)
    }
    fn encode_audio(&mut self, finishing: bool) -> Result<(), ffmpeg::Error> {
        loop {
            let Some(encoder) = &mut self.audio else {return Ok(())};
            let available = unsafe {ffmpeg::ffi::av_audio_fifo_size(self.samples)} as usize;
            let size = encoder.frame_size() as usize;
            if available == 0 || (!finishing && size > 0 && available < size) {return Ok(())}
            let count = if size == 0 {available} else {available.min(size)};
            let mut frame = ffmpeg::frame::Audio::new(
                encoder.format(), count, encoder.channel_layout(),
            );
            frame.set_rate(encoder.rate());
            frame.set_pts(Some(self.audio_position));
            let read = unsafe {
                ffmpeg::ffi::av_audio_fifo_read(
                    self.samples, (*frame.as_mut_ptr()).extended_data.cast(), count as i32,
                )
            };
            if read < 0 {return Err(ffmpeg::Error::from(read))}
            assert_eq!(read as usize, count);
            self.audio_position += count as i64;
            encoder.send_frame(&frame)?;
            self.drain(self.audio_stream.unwrap())?;
        }
    }
    fn drain(&mut self, index: usize) -> Result<(), ffmpeg::Error> {
        let encoder: &mut ffmpeg::encoder::Encoder = match (index, self.video.as_mut()) {
            (0, Some(encoder)) => encoder,
            _ => self.audio.as_mut().unwrap(),
        };
        loop {
            let mut packet = ffmpeg::Packet::empty();
            match encoder.receive_packet(&mut packet) {
                Ok(()) => {},
                Err(ffmpeg::Error::Eof) => return Ok(()),
                Err(ffmpeg::Error::Other {errno}) if errno == ffmpeg::error::EAGAIN => return Ok(()),
                Err(error) => return Err(error),
            }
            packet.set_stream(index);
            let time_base = self.output.stream(index).unwrap().time_base();
            if matches!(encoder.id(), ffmpeg::codec::Id::MPEG4 | ffmpeg::codec::Id::FFV1) {
                let (timestamp, duration, corrupted) = self.video_timing.pop_front().unwrap();
                let timestamp = i64::try_from(timestamp.as_nanos()).unwrap()
                    .rescale((1, 1_000_000_000), time_base);
                let duration = i64::try_from(duration.as_nanos()).unwrap()
                    .rescale((1, 1_000_000_000), time_base);
                packet.set_pts(Some(timestamp));
                packet.set_dts(Some(timestamp));
                packet.set_duration(duration);
                if encoder.id() == ffmpeg::codec::Id::FFV1 && corrupted {
                    let result = unsafe {ffmpeg::ffi::av_packet_make_writable(packet.as_mut_ptr())};
                    if result < 0 {return Err(ffmpeg::Error::from(result))}
                    let bytes = packet.data_mut().unwrap();
                    let table = unsafe {
                        ffmpeg::ffi::av_crc_get_table(ffmpeg::ffi::AVCRCId::AV_CRC_32_IEEE)
                    };
                    let mut end = bytes.len();
                    while end > 0 {
                        let size = u32::from_be_bytes([
                            0, bytes[end - 8], bytes[end - 7], bytes[end - 6],
                        ]) as usize;
                        let start = end.checked_sub(size + 8).unwrap();
                        // RFC 9043 signals source damage without invalidating the slice CRC.
                        bytes[end - 5] = 2;
                        let crc = unsafe {
                            ffmpeg::ffi::av_crc(table, 0, bytes[start..].as_ptr(), end - start - 4)
                        };
                        bytes[end - 4..end].copy_from_slice(&crc.to_le_bytes());
                        end = start;
                    }
                }
            } else if encoder.id() == ffmpeg::codec::Id::OPUS {
                let pts = packet.pts().map(|stamp| {
                    stamp.rescale_with(encoder.time_base(), time_base, ffmpeg::Rounding::Down)
                });
                let dts = packet.dts().map(|stamp| {
                    stamp.rescale_with(encoder.time_base(), time_base, ffmpeg::Rounding::Down)
                });
                packet.rescale_ts(encoder.time_base(), time_base);
                packet.set_pts(pts);
                packet.set_dts(dts);
            } else {packet.rescale_ts(encoder.time_base(), time_base);}
            packet.set_position(-1);
            packet.write_interleaved(&mut self.output)?;
        }
    }
}
impl Export for Encoder {
    fn write<W: Write + Seek + Send + 'static>(audio: &Audio, video: &Video, writer: W, format: Format) -> Result<(), String> {
        let writer = Arc::new(Mutex::new(writer));
        let io = ffmpeg::format::context::StreamIo::from_write_seek(Output(writer.clone()))
            .map_err(|error| error.to_string())?;
        let media = Media {audio: audio.clone(), video: video.clone()};
        let mut encoder = Self::open(media, format, io, None, false)
            .map_err(|error| error.to_string())?;
        while !encoder.finished {encoder.step().map_err(|error| error.to_string())?;}
        drop(encoder);
        writer.lock().unwrap().flush().map_err(|error| error.to_string())
    }
}
impl Iterator for Encoder {
    type Item = Result<Vec<u8>, String>;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let bytes = self.bytes.as_ref().unwrap();
            let bytes = std::mem::take(&mut *bytes.lock().unwrap());
            if !bytes.is_empty() {return Some(Ok(bytes))}
            if self.finished {return None}
            if let Err(error) = self.step() {
                self.finished = true;
                self.bytes.as_ref().unwrap().lock().unwrap().clear();
                return Some(Err(error.to_string()))
            }
        }
    }
}
impl Drop for Encoder {
    fn drop(&mut self) {unsafe {ffmpeg::ffi::av_audio_fifo_free(self.samples);}}
}

enum PacketReader {
    File(ffmpeg::format::context::Input, usize),
    Stored(PacketSource, usize),
}
impl PacketReader {
    fn read(&mut self) -> Result<Option<ffmpeg::Packet>, ffmpeg::Error> {
        match self {
            Self::File(input, index) => loop {
                let mut packet = ffmpeg::Packet::empty();
                match packet.read(input) {
                    Ok(()) if packet.stream() == *index => return Ok(Some(packet)),
                    Ok(()) => {},
                    Err(ffmpeg::Error::Eof) => return Ok(None),
                    Err(error) => return Err(error),
                }
            },
            Self::Stored(source, index) => {
                let packet = source.packet(*index);
                *index += usize::from(packet.is_some());
                Ok(packet)
            }
        }
    }
}

struct Copy {
    reader: PacketReader,
    parameters: ffmpeg::codec::Parameters,
    time_base: ffmpeg::Rational,
    rate: ffmpeg::Rational,
    origin: f64,
    source_end: f64,
    range: Range<f64>,
    shift: f64,
    next: Option<ffmpeg::Packet>,
    packet: Option<ffmpeg::Packet>,
    following: VecDeque<Clip>,
}
impl Copy {
    fn new(clips: &[Clip], format: Format) -> Result<Option<Self>, ffmpeg::Error> {
        let Some(first) = clips.first() else {return Ok(None)};
        let codec = if first.dimensions.0 > 0 {format.video_codec()} else {format.audio_codec()};
        let Some(codec) = codec else {return Ok(None)};
        let Some(mut copy) = Self::clip(first, codec)? else {return Ok(None)};
        if format == Format::Flac_Flac {
            let tick = f64::from(copy.time_base);
            if clips.len() > 1 || copy.shift.abs() > tick / 2.0
                || (copy.range.start - copy.origin).abs() > tick / 2.0
                || (copy.range.end - copy.source_end).abs() > tick / 2.0 {
                return Ok(None)
            }
        }
        if clips.len() > 1 {
            let parameters = unsafe {&*copy.parameters.as_ptr()};
            if matches!(codec, ffmpeg::codec::Id::AAC | ffmpeg::codec::Id::OPUS)
                || parameters.video_delay > 0 {return Ok(None)}
            let mut end = first.end;
            for clip in &clips[1..] {
                if clip.start != end {return Ok(None)}
                let Some(next) = Self::clip(clip, codec)? else {return Ok(None)};
                if !copy.compatible(&next) {return Ok(None)}
                let Some(packet) = &next.packet else {return Ok(None)};
                let start = packet.dts().or(packet.pts()).unwrap() as f64
                    * f64::from(next.time_base);
                if start < clip.start.as_secs_f64() - f64::from(next.time_base) / 2.0 {
                    return Ok(None)
                }
                copy.following.push_back(clip.clone());
                end = clip.end;
            }
        }
        Ok(Some(copy))
    }
    fn clip(clip: &Clip, codec: ffmpeg::codec::Id) -> Result<Option<Self>, ffmpeg::Error> {
        if !matches!(clip.data, Content::Encoded {..}) {return Ok(None)}
        let mut copy = Self::open(clip, clip.offset)?;
        if copy.parameters.id() != codec {return Ok(None)}
        let parameters = unsafe {&*copy.parameters.as_ptr()};
        if codec == ffmpeg::codec::Id::PRORES && parameters.profile != 2 {return Ok(None)}
        if codec == ffmpeg::codec::Id::MPEG4 && parameters.extradata_size == 0 {return Ok(None)}
        let tick = f64::from(copy.time_base);
        let start_trimmed = copy.range.start > copy.origin + tick / 2.0;
        let end_trimmed = copy.range.end < copy.source_end - tick / 2.0;
        let video = copy.parameters.medium() == ffmpeg::media::Type::Video;
        if start_trimmed && matches!(codec, ffmpeg::codec::Id::AAC | ffmpeg::codec::Id::OPUS) {
            return Ok(None)
        }
        if video && (start_trimmed || end_trimmed) && parameters.video_delay > 0 {
            return Ok(None)
        }
        if !video && end_trimmed {
            let mut boundary = Self::open(clip, copy.range.end - tick)?;
            let mut aligned = false;
            while let Some(packet) = boundary.reader.read()? {
                let Some(stamp) = packet.pts().or(packet.dts()) else {return Ok(None)};
                let start = stamp as f64 * tick;
                let end = start + packet.duration() as f64 * tick;
                if (start - copy.range.end).abs() < tick / 2.0
                    || (end - copy.range.end).abs() < tick / 2.0 {
                    aligned = true;
                    break
                }
                if start > copy.range.end || end > copy.range.end {break}
            }
            if !aligned {return Ok(None)}
        }
        copy.advance()?;
        let Some(packet) = &copy.packet else {return Ok(None)};
        if start_trimmed {
            let Some(stamp) = packet.pts().or(packet.dts()) else {return Ok(None)};
            if (stamp as f64 * tick - clip.start.as_secs_f64()).abs() > tick / 2.0 {
                return Ok(None)
            }
            if video && !packet.is_key() {return Ok(None)}
        }
        Ok(Some(copy))
    }
    fn open(clip: &Clip, position: f64) -> Result<Self, ffmpeg::Error> {
        let Content::Encoded {source, end, ..} = &clip.data else {unreachable!()};
        let kind = if clip.dimensions.0 > 0 {ffmpeg::media::Type::Video} else {
            ffmpeg::media::Type::Audio
        };
        let (reader, parameters, time_base, origin, rate) = match source {
            Encoded::File(source) => {
                let mut input = FrameDecoder::input(source.clone())?;
                let stream = input.streams().best(kind).ok_or(ffmpeg::Error::StreamNotFound)?;
                let index = stream.index();
                let time_base = stream.time_base();
                let origin = if stream.start_time() == i64::MIN {0.0} else {
                    stream.start_time() as f64 * f64::from(time_base)
                };
                let parameters = stream.parameters().clone();
                let rate = stream.avg_frame_rate();
                if position > origin {
                    let stamp = (position * 1_000_000.0).round() as i64;
                    input.seek(stamp, ..stamp)?;
                }
                (PacketReader::File(input, index), parameters, time_base, origin, rate)
            }
            Encoded::Packets(source) => {
                let packets = source.source.lock().unwrap();
                let time_base = packets.time_base;
                let index = if position > packets.start {
                    packets.packets[..source.length].partition_point(|packet| {
                        packet.dts().or(packet.pts()).is_some_and(|stamp| {
                            stamp as f64 * f64::from(time_base) < position
                        })
                    }).saturating_sub(1)
                } else {0};
                let parameters = packets.parameters.clone();
                let origin = packets.start;
                let rate = if source.interval > 0.0 {
                    ffmpeg::Rational::from(1.0 / source.interval)
                } else {ffmpeg::Rational(0, 1)};
                (PacketReader::Stored(source.clone(), index), parameters, time_base, origin, rate)
            }
        };
        Ok(Self {
            reader, parameters, time_base, origin, rate, source_end: *end,
            range: clip.offset..clip.offset + (clip.end - clip.start).as_secs_f64(),
            shift: clip.start.as_secs_f64() - clip.offset, next: None, packet: None,
            following: VecDeque::new(),
        })
    }
    fn compatible(&self, other: &Self) -> bool {
        let a = unsafe {&*self.parameters.as_ptr()};
        let b = unsafe {&*other.parameters.as_ptr()};
        if a.codec_id != b.codec_id || a.format != b.format || a.profile != b.profile
            || a.level != b.level || a.width != b.width || a.height != b.height
            || a.sample_rate != b.sample_rate || a.bits_per_raw_sample != b.bits_per_raw_sample
            || a.frame_size != b.frame_size || a.initial_padding != b.initial_padding
            || a.seek_preroll != b.seek_preroll || a.extradata_size != b.extradata_size {
            return false
        }
        if a.sample_rate > 0
            && unsafe {ffmpeg::ffi::av_channel_layout_compare(&a.ch_layout, &b.ch_layout)} != 0 {
            return false
        }
        if a.extradata_size == 0 {return true}
        unsafe {
            std::slice::from_raw_parts(a.extradata, a.extradata_size as usize)
                == std::slice::from_raw_parts(b.extradata, b.extradata_size as usize)
        }
    }
    fn finish_clip(&mut self) -> Result<(), ffmpeg::Error> {
        let Some(clip) = self.following.pop_front() else {return Ok(())};
        let mut next = Self::clip(&clip, self.parameters.id())?.ok_or(ffmpeg::Error::InvalidData)?;
        next.following = std::mem::take(&mut self.following);
        *self = next;
        Ok(())
    }
    fn add_stream(&self, output: &mut ffmpeg::format::context::Output) -> Result<(), ffmpeg::Error> {
        let mut stream = output.add_stream(ffmpeg::encoder::find(self.parameters.id()))?;
        stream.set_parameters(self.parameters.clone());
        stream.set_time_base(self.time_base);
        stream.set_avg_frame_rate(self.rate);
        unsafe {(*stream.parameters().as_mut_ptr()).codec_tag = 0;}
        Ok(())
    }
    fn advance(&mut self) -> Result<(), ffmpeg::Error> {
        self.packet = None;
        let tick = f64::from(self.time_base);
        let start_trimmed = self.range.start > self.origin + tick / 2.0;
        let end_trimmed = self.range.end < self.source_end - tick / 2.0;
        loop {
            let packet = match self.next.take() {
                Some(packet) => Some(packet),
                None => self.reader.read()?,
            };
            let Some(mut packet) = packet else {return self.finish_clip()};
            self.next = self.reader.read()?;
            let stamp = packet.pts().or(packet.dts()).ok_or(ffmpeg::Error::InvalidData)?;
            let start = stamp as f64 * tick;
            if start_trimmed && start < self.range.start - tick / 2.0 {continue}
            if start >= self.range.end - tick / 2.0 {return self.finish_clip()}
            if packet.duration() == 0 {
                let end = self.next.as_ref().and_then(|next| next.pts().or(next.dts()))
                    .map_or(self.source_end, |stamp| stamp as f64 * tick);
                packet.set_duration(((end - start) / tick).round().max(0.0) as i64);
            }
            if end_trimmed {
                let remaining = ((self.range.end - start) / tick).round().max(0.0) as i64;
                packet.set_duration(packet.duration().min(remaining));
            }
            let offset = (self.shift / tick).round() as i64;
            packet.set_pts(packet.pts().map(|stamp| stamp + offset));
            packet.set_dts(packet.dts().map(|stamp| stamp + offset));
            self.packet = Some(packet);
            return Ok(())
        }
    }
    fn time(&self) -> Option<Duration> {
        self.packet.as_ref().map(|packet| {
            let stamp = packet.dts().or(packet.pts()).unwrap();
            Duration::from_secs_f64((stamp as f64 * f64::from(self.time_base)).max(0.0))
        })
    }
    fn write(&mut self, output: &mut ffmpeg::format::context::Output, index: usize) -> Result<(), ffmpeg::Error> {
        let mut packet = self.packet.take().unwrap();
        packet.set_stream(index);
        packet.set_position(-1);
        packet.rescale_ts(self.time_base, output.stream(index).unwrap().time_base());
        packet.write_interleaved(output)?;
        self.advance()
    }
}

struct Output<W>(Arc<Mutex<W>>);
impl<W: Write> Write for Output<W> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {self.0.lock().unwrap().write(bytes)}
    fn flush(&mut self) -> std::io::Result<()> {self.0.lock().unwrap().flush()}
}
impl<W: Seek> Seek for Output<W> {
    fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
        self.0.lock().unwrap().seek(position)
    }
}
