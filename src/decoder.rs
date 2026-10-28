use ffmpeg_next as ffmpeg;
use ffmpeg::{codec::{Id, packet::Mut}, decoder::Check, format::stream::Disposition, media::Type};
use ffmpeg::ffi::AVPacketSideDataType::AV_PKT_DATA_SKIP_SAMPLES;

use std::io::Cursor;
use std::ops::{Deref, Range};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use crate::media::{Clip, Content, Encoded, Input, Packets, PacketSource, Track};
use crate::{Format, Image, Media};

// FFmpeg initializes an unsynchronized process-wide error-string table.
static START: OnceLock<Result<(), ffmpeg::Error>> = OnceLock::new();

#[derive(Debug)]
pub enum DecodeError {Corrupt(f64), Failed(ffmpeg::Error)}
impl From<ffmpeg::Error> for DecodeError {
    fn from(error: ffmpeg::Error) -> Self {Self::Failed(error)}
}
impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Corrupt(position) => write!(f, "Corrupt frame at {position:.3}s"),
            Self::Failed(error) => error.fmt(f),
        }
    }
}

/// Incrementally receives a continuous encoded stream. Decoded frames remain lazy.
/// MP4, MOV and M4A input must be fragmented; network chunks may split any container boundary.
pub struct Decoder {
    media: Media,
    pending: Vec<u8>,
    header: Vec<u8>,
    container: Container,
    tracks: Vec<(usize, Arc<Mutex<Packets>>)>,
}
impl Decoder {
    pub fn new() -> Self {
        Self {
            media: Media::new(), pending: Vec::new(), header: Vec::new(),
            container: Container::Unknown, tracks: Vec::new(),
        }
    }
    pub fn decode(&mut self, bytes: &[u8]) -> Result<(), String> {
        if let Container::Failed(error) = &self.container {return Err(error.clone())}
        self.pending.extend_from_slice(bytes);
        self.receive(false)
    }
    pub fn finish(mut self) -> Result<Media, String> {
        self.receive(true)?;
        if self.tracks.is_empty() {return Err("Stream has no audio or video track".into())}
        for (_, source) in &self.tracks {source.lock().unwrap().finished = true;}
        self.publish();
        Ok(self.media)
    }
    fn receive(&mut self, finished: bool) -> Result<(), String> {
        let mut container = std::mem::replace(&mut self.container, Container::Unknown);
        let result = (|| {
            if matches!(container, Container::Unknown) {
                if self.pending.len() < 12 {
                    return if finished {Err("Incomplete media header".into())} else {Ok(())}
                }
                container = if &self.pending[..4] == b"fLaC" {
                    self.header.extend(self.pending.drain(..4));
                    Container::Flac {parser: std::ptr::null_mut(), context: None, position: 0}
                } else if &self.pending[..4] == b"RIFF" && &self.pending[8..12] == b"WAVE" {
                    self.header.extend(self.pending.drain(..12));
                    Container::Wave {remaining: None, alignment: 0, position: 0, padding: false}
                } else if self.pending[..4] == [0x1a, 0x45, 0xdf, 0xa3] {
                    Container::Matroska {segment: false, initialized: false}
                } else if &self.pending[4..8] == b"ftyp" || &self.pending[4..8] == b"styp" {
                    Container::Mp4 {fragment: Vec::new(), initialized: false, position: 0, origin: 0}
                } else {return Err("Unsupported streaming container".into())};
            }
            match &mut container {
                Container::Unknown => Ok(()),
                Container::Mp4 {fragment, initialized, position, origin} => {
                    self.mp4(fragment, initialized, position, origin, finished)
                }
                Container::Matroska {segment, initialized} => {
                    self.matroska(segment, initialized, finished)
                }
                Container::Wave {remaining, alignment, position, padding} => {
                    self.wave(remaining, alignment, position, padding, finished)
                }
                Container::Flac {parser, context, position} => {
                    self.flac(parser, context, position, finished)
                }
                Container::Failed(error) => Err(error.clone()),
            }
        })();
        self.container = match &result {
            Ok(()) => container,
            Err(error) => Container::Failed(error.clone()),
        };
        result
    }
    fn mp4(&mut self, fragment: &mut Vec<u8>, initialized: &mut bool, position: &mut u64, origin: &mut u64, finished: bool) -> Result<(), String> {
        loop {
            if self.pending.len() < 8 {break}
            let mut size = u32::from_be_bytes(self.pending[..4].try_into().unwrap()) as u64;
            let header = if size == 1 {
                if self.pending.len() < 16 {break}
                size = u64::from_be_bytes(self.pending[8..16].try_into().unwrap());
                16
            } else {8};
            if size == 0 {
                if !finished {break}
                size = self.pending.len() as u64;
            }
            if size < header {return Err("Invalid MP4 box length".into())}
            if size > self.pending.len() as u64 {break}
            let size = size as usize;
            let kind: [u8; 4] = self.pending[4..8].try_into().unwrap();
            match &kind {
                b"ftyp" | b"styp" => {
                    if *initialized {return Err("MP4 initialization changed within a stream".into())}
                    self.header.extend(self.pending.drain(..size));
                }
                b"moov" => {
                    if *initialized {return Err("MP4 initialization changed within a stream".into())}
                    self.header.extend(self.pending.drain(..size));
                    *initialized = true;
                }
                b"moof" => {
                    if !*initialized || !fragment.is_empty() {
                        return Err("MP4 fragment is missing initialization or sample data".into())
                    }
                    *origin = *position;
                    fragment.extend(self.pending.drain(..size));
                }
                b"mdat" => {
                    if fragment.is_empty() {return Err("Streaming MP4 input must be fragmented".into())}
                    fragment.extend(self.pending.drain(..size));
                    Self::relocate(fragment, *origin, self.header.len() as u64)?;
                    let mut bytes = self.header.clone();
                    bytes.append(fragment);
                    self.demux(bytes)?;
                }
                _ => {
                    if !fragment.is_empty() {fragment.extend(self.pending.drain(..size));}
                    else {self.pending.drain(..size);}
                }
            }
            *position += size as u64;
        }
        if finished {
            if !self.pending.is_empty() || !fragment.is_empty() || !*initialized {
                return Err("Incomplete MP4 fragment".into())
            }
            if self.tracks.is_empty() {self.demux(self.header.clone())?;}
        }
        Ok(())
    }
    fn relocate(bytes: &mut [u8], original: u64, position: u64) -> Result<(), String> {
        let mut cursor = 0;
        while cursor < bytes.len() {
            if bytes.len() - cursor < 8 {return Err("Incomplete MP4 box".into())}
            let mut size = u32::from_be_bytes(bytes[cursor..cursor + 4].try_into().unwrap()) as u64;
            let header = if size == 1 {
                if bytes.len() - cursor < 16 {return Err("Incomplete MP4 box".into())}
                size = u64::from_be_bytes(bytes[cursor + 8..cursor + 16].try_into().unwrap());
                16
            } else {8};
            if size == 0 {size = (bytes.len() - cursor) as u64;}
            if size < header as u64 || size > (bytes.len() - cursor) as u64 {
                return Err("Invalid MP4 box length".into())
            }
            let end = cursor + size as usize;
            let kind: [u8; 4] = bytes[cursor + 4..cursor + 8].try_into().unwrap();
            let content = &mut bytes[cursor + header..end];
            if &kind == b"moof" || &kind == b"traf" {Self::relocate(content, original, position)?;}
            else if &kind == b"tfhd" {
                if content.len() < 8 {return Err("Incomplete MP4 track fragment header".into())}
                if content[3] & 1 != 0 {
                    if content.len() < 16 {return Err("Incomplete MP4 track data offset".into())}
                    let base = u64::from_be_bytes(content[8..16].try_into().unwrap());
                    let base = u64::try_from(base as i128 + position as i128 - original as i128)
                        .map_err(|_| "MP4 fragment refers to data outside its samples")?;
                    content[8..16].copy_from_slice(&base.to_be_bytes());
                }
            }
            cursor = end;
        }
        Ok(())
    }
    fn matroska(&mut self, segment: &mut bool, initialized: &mut bool, finished: bool) -> Result<(), String> {
        while let Some((id, header, length)) = Self::element(&self.pending)? {
            if id == 0x18538067 && !*segment {
                let mut bytes: Vec<_> = self.pending.drain(..header).collect();
                let width = header - 4;
                bytes[4] = 0xff >> (width - 1);
                bytes[5..].fill(0xff);
                self.header.extend(bytes);
                *segment = true;
                continue
            }
            let size = if let Some(length) = length {
                let size = header.checked_add(length).ok_or("Invalid Matroska element length")?;
                if size > self.pending.len() {break}
                size
            } else if id == 0x1f43b675 {
                let mut cursor = header;
                let mut complete = false;
                while let Some((next, size, length)) = Self::element(&self.pending[cursor..])? {
                    if matches!(next,
                        0x1f43b675 | 0x1c53bb6b | 0x1254c367 | 0x114d9b74
                        | 0x1549a966 | 0x1654ae6b | 0x1941a469 | 0x1043a770
                    ) {
                        complete = true;
                        break
                    }
                    let length = length.ok_or("Unknown length inside a Matroska cluster")?;
                    let end = cursor.checked_add(size).and_then(|end| end.checked_add(length))
                        .ok_or("Invalid Matroska element length")?;
                    if end > self.pending.len() {break}
                    cursor = end;
                }
                if !(complete || finished && cursor == self.pending.len()) {break}
                cursor
            } else {return Err("Unsupported unknown Matroska element length".into())};
            if id == 0x1f43b675 {
                if !*segment {return Err("Matroska cluster precedes its segment".into())}
                let mut bytes = self.header.clone();
                bytes.extend(self.pending.drain(..size));
                self.demux(bytes)?;
                *initialized = true;
            } else if !*initialized {self.header.extend(self.pending.drain(..size));}
            else {self.pending.drain(..size);}
        }
        if finished {
            if !self.pending.is_empty() || !*segment {return Err("Incomplete Matroska stream".into())}
            if self.tracks.is_empty() {self.demux(self.header.clone())?;}
        }
        Ok(())
    }
    fn element(bytes: &[u8]) -> Result<Option<(u64, usize, Option<usize>)>, String> {
        let Some(&first) = bytes.first() else {return Ok(None)};
        let id_size = first.leading_zeros() as usize + 1;
        if id_size > 4 {return Err("Invalid Matroska element ID".into())}
        if bytes.len() <= id_size {return Ok(None)}
        let size_size = bytes[id_size].leading_zeros() as usize + 1;
        if size_size > 8 {return Err("Invalid Matroska element length".into())}
        let header = id_size + size_size;
        if bytes.len() < header {return Ok(None)}
        let id = bytes[..id_size].iter().fold(0_u64, |value, &byte| value << 8 | byte as u64);
        let first = bytes[id_size] & (0xff_u16 >> size_size) as u8;
        let length = bytes[id_size + 1..header].iter()
            .fold(first as u64, |value, &byte| value << 8 | byte as u64);
        let length = if length == (1_u64 << (size_size * 7)) - 1 {None} else {
            Some(usize::try_from(length).map_err(|_| "Invalid Matroska element length")?)
        };
        Ok(Some((id, header, length)))
    }
    fn wave(
        &mut self, remaining: &mut Option<u64>, alignment: &mut usize,
        position: &mut i64, padding: &mut bool, finished: bool,
    ) -> Result<(), String> {
        loop {
            if let Some(left) = remaining.as_mut() {
                let available = self.pending.len().min((*left).min(65_536) as usize);
                let size = available / *alignment * *alignment;
                if size == 0 {break}
                let mut packet = ffmpeg::Packet::copy(&self.pending[..size]);
                self.pending.drain(..size);
                let duration = (size / *alignment) as i64;
                packet.set_pts(Some(*position));
                packet.set_dts(Some(*position));
                packet.set_duration(duration);
                packet.set_flags(ffmpeg::codec::packet::Flags::KEY);
                *position += duration;
                self.append(0, packet);
                if *left != u64::MAX {*left -= size as u64;}
                if *left == 0 {*remaining = None;}
                continue
            }
            if *padding {
                if self.pending.is_empty() {break}
                self.pending.remove(0);
                *padding = false;
            }
            if self.pending.len() < 8 {break}
            let kind: [u8; 4] = self.pending[..4].try_into().unwrap();
            let length = u32::from_le_bytes(self.pending[4..8].try_into().unwrap()) as usize;
            if &kind == b"data" {
                if *alignment == 0 {return Err("WAV samples precede the audio format".into())}
                self.header.extend(self.pending.drain(..8));
                let end = self.header.len();
                self.header[end - 4..].fill(0);
                if self.tracks.is_empty() {self.demux(self.header.clone())?;}
                *remaining = (length != 0).then_some(if length == u32::MAX as usize {
                    u64::MAX
                } else {length as u64});
                *padding = length != u32::MAX as usize && !length.is_multiple_of(2);
                continue
            }
            let size = 8 + length + length % 2;
            if self.pending.len() < size {break}
            if &kind == b"fmt " {
                if length < 16 {return Err("Invalid WAV audio format".into())}
                *alignment = u16::from_le_bytes(self.pending[20..22].try_into().unwrap()) as usize;
                if *alignment == 0 {return Err("Invalid WAV sample alignment".into())}
            }
            self.header.extend(self.pending.drain(..size));
        }
        self.publish();
        if finished && (!self.pending.is_empty() || remaining.is_some_and(|left| left != u64::MAX)) {
            return Err("Incomplete WAV samples".into())
        }
        Ok(())
    }
    fn flac(
        &mut self, parser: &mut *mut ffmpeg::ffi::AVCodecParserContext,
        context: &mut Option<ffmpeg::codec::Context>, position: &mut i64, finished: bool,
    ) -> Result<(), String> {
        if parser.is_null() {
            let mut ready = false;
            while self.pending.len() >= 4 {
                let length = u32::from_be_bytes([
                    0, self.pending[1], self.pending[2], self.pending[3],
                ]) as usize;
                if self.pending.len() < length + 4 {break}
                ready = self.pending[0] & 0x80 != 0;
                self.header.extend(self.pending.drain(..length + 4));
                if ready {break}
            }
            if !ready {
                return if finished {Err("Incomplete FLAC metadata".into())} else {Ok(())}
            }
            self.demux(self.header.clone())?;
            let source = self.tracks.first().ok_or("FLAC stream has no audio track")?.1.lock().unwrap();
            *context = Some(ffmpeg::codec::Context::from_parameters(source.parameters.clone())
                .map_err(|error| error.to_string())?);
            *parser = unsafe {ffmpeg::ffi::av_parser_init(ffmpeg::ffi::AVCodecID::AV_CODEC_ID_FLAC)};
            assert!(!parser.is_null());
            unsafe {(**parser).flags |= ffmpeg::ffi::PARSER_FLAG_USE_CODEC_TS;}
        }
        let mut data = std::mem::take(&mut self.pending);
        let length = data.len();
        data.resize(length + ffmpeg::ffi::AV_INPUT_BUFFER_PADDING_SIZE as usize, 0);
        let mut cursor = 0;
        loop {
            let flush = cursor == length;
            if flush && !finished {break}
            let mut output = std::ptr::null_mut();
            let mut size = 0;
            let consumed = unsafe {
                ffmpeg::ffi::av_parser_parse2(
                    *parser, context.as_mut().unwrap().as_mut_ptr(), &mut output, &mut size,
                    if flush {std::ptr::null()} else {data[cursor..].as_ptr()},
                    (length - cursor).min(i32::MAX as usize) as i32,
                    i64::MIN, i64::MIN, -1,
                )
            };
            if consumed < 0 {return Err(ffmpeg::Error::from(consumed).to_string())}
            cursor += consumed as usize;
            if size > 0 {
                let bytes = unsafe {std::slice::from_raw_parts(output, size as usize)};
                let mut packet = ffmpeg::Packet::copy(bytes);
                let description = unsafe {&**parser};
                let stamp = if description.pts >= *position {description.pts} else {*position};
                packet.set_pts(Some(stamp));
                packet.set_dts(Some(stamp));
                packet.set_duration(description.duration as i64);
                let mut flags = ffmpeg::codec::packet::Flags::KEY;
                if description.duration == 0 {flags |= ffmpeg::codec::packet::Flags::CORRUPT;}
                packet.set_flags(flags);
                *position = stamp + description.duration as i64;
                self.append(0, packet);
            }
            if consumed == 0 && size == 0 {break}
        }
        self.pending.extend_from_slice(&data[cursor..length]);
        self.publish();
        if finished && !self.pending.is_empty() {return Err("Incomplete FLAC frame".into())}
        Ok(())
    }
    fn demux(&mut self, bytes: Vec<u8>) -> Result<(), String> {
        FrameDecoder::init().map_err(|error| error.to_string())?;
        let io = ffmpeg::format::context::StreamIo::from_read(Cursor::new(bytes))
            .map_err(|error| error.to_string())?;
        let mut input = FrameDecoder::open(io).map_err(|error| error.to_string())?;
        let resumed = !self.tracks.is_empty();
        if !resumed {
            FrameDecoder::describe(&mut input).map_err(|error| error.to_string())?;
            let format = FrameDecoder::detect(&input);
            for kind in [Type::Audio, Type::Video] {
                let Some(stream) = input.streams().best(kind) else {continue};
                if stream.disposition().contains(Disposition::ATTACHED_PIC) {continue}
                let base = stream.time_base();
                let start = FrameDecoder::start(&stream);
                let rate = f64::from(stream.avg_frame_rate());
                let interval = if rate.is_finite() && rate > 0.0 {1.0 / rate} else {0.0};
                let corrupted = FrameDecoder::corruption(input.metadata().get("comment"), kind);
                self.tracks.push((stream.index(), Arc::new(Mutex::new(Packets {
                    parameters: stream.parameters().clone(), format, time_base: base,
                    start, interval, end: start,
                    packets: Vec::new(), published: 0, corrupted, finished: false,
                }))));
            }
        }
        loop {
            let mut packet = ffmpeg::Packet::empty();
            match packet.read(&mut input) {
                Ok(()) => {
                    if resumed {
                        // Reopening an initialization header repeats its leading codec-delay skip.
                        // Only the first fragment owns that skip; trailing padding remains intact.
                        unsafe {
                            let raw = &mut *packet.as_mut_ptr();
                            for index in 0..raw.side_data_elems as usize {
                                let side = &mut *raw.side_data.add(index);
                                if side.type_ == AV_PKT_DATA_SKIP_SAMPLES && side.size >= 4 {
                                    std::ptr::write_bytes(side.data, 0, 4);
                                }
                            }
                        }
                    }
                    self.append(packet.stream(), packet);
                }
                Err(ffmpeg::Error::Eof) => break,
                Err(error) => return Err(error.to_string()),
            }
        }
        self.publish();
        Ok(())
    }
    fn append(&mut self, index: usize, packet: ffmpeg::Packet) {
        let Some((_, source)) = self.tracks.iter().find(|(stream, _)| *stream == index) else {
            return
        };
        let mut source = source.lock().unwrap();
        let base = f64::from(source.time_base);
        if packet.duration() > 0 && source.interval == 0.0 {
            source.interval = packet.duration() as f64 * base;
        }
        let inferred = source.packets.last_mut().filter(|previous| previous.duration() == 0)
            .and_then(|previous| {
                let (previous_stamp, stamp) = (previous.pts()?, packet.pts()?);
                if stamp <= previous_stamp {return None}
                previous.set_duration(stamp - previous_stamp);
                Some((stamp - previous_stamp) as f64 * base)
            });
        if source.interval == 0.0 {source.interval = inferred.unwrap_or(0.0);}
        if let Some(stamp) = packet.pts().or(packet.dts()) {
            let duration = if packet.duration() > 0 {
                packet.duration() as f64 * base
            } else {source.interval};
            source.end = source.end.max(stamp as f64 * base + duration);
        }
        source.packets.push(packet);
    }
    fn publish(&mut self) {
        for (_, source) in &self.tracks {
            let mut packets = source.lock().unwrap();
            if packets.packets.is_empty()
                || (packets.published == packets.packets.len() && !packets.finished) {continue}
            let kind = packets.parameters.medium();
            let first = packets.published.saturating_sub(1);
            let changed = if packets.finished {0.0} else {
                packets.packets[first..].iter().filter_map(|packet| packet.dts().or(packet.pts()))
                    .min().map_or(0.0, |stamp| stamp as f64 * f64::from(packets.time_base))
            };
            packets.published = packets.packets.len();
            let corrupted = Arc::new(packets.corrupted.clone());
            let snapshot = PacketSource {
                source: source.clone(), length: packets.packets.len(), end: packets.end,
                interval: packets.interval, duration: packets.packets.last().unwrap().duration(),
                finished: packets.finished, corrupted: corrupted.clone(),
            };
            let parameters = unsafe {&*packets.parameters.as_ptr()};
            let clip = Clip {
                data: Content::Encoded {
                    source: Encoded::Packets(snapshot), end: packets.end, format: packets.format,
                },
                start: Duration::ZERO, end: Duration::from_secs_f64(packets.end.max(0.0)), offset: 0.0,
                dimensions: (parameters.width as u32, parameters.height as u32),
                sound: (parameters.sample_rate as u32, parameters.ch_layout.nb_channels as u16),
                declared_end: packets.finished, corrupted,
            };
            drop(packets);
            let changed = Duration::from_secs_f64(changed.max(0.0));
            if kind == Type::Audio {self.media.audio.update(clip, changed);}
            else {self.media.video.update(clip, changed);}
        }
    }
}
impl Default for Decoder {
    fn default() -> Self {Self::new()}
}
impl Deref for Decoder {
    type Target = Media;
    fn deref(&self) -> &Media {&self.media}
}

pub struct FrameDecoder {
    pub base: f64,
    pub start: f64,
    pub interval: f64,
    pub end: f64,
    input: DecodeInput,
    decoder: ffmpeg::decoder::Opened,
    index: usize,
    dimensions: (u32, u32),
    sound: (u32, u16),
    rgba: ffmpeg::frame::Video,
    packet: f64,
    draining: bool,
    feed: bool,
    native_corrupted: bool,
}
impl FrameDecoder {
    pub fn init() -> Result<(), ffmpeg::Error> {*START.get_or_init(ffmpeg::init)}
    pub fn new(source: Encoded, kind: Type) -> Result<Self, ffmpeg::Error> {
        Self::init()?;
        let (input, parameters, index, time_base, start, interval) = match source {
            Encoded::File(source) => {
                let input = Self::input(source)?;
                let stream = input.streams().best(kind).ok_or(ffmpeg::Error::StreamNotFound)?;
                if kind == Type::Video && stream.disposition().contains(Disposition::ATTACHED_PIC) {
                    return Err(ffmpeg::Error::StreamNotFound)
                }
                let index = stream.index();
                let base = stream.time_base();
                let start = Self::start(&stream);
                let rate = f64::from(stream.avg_frame_rate());
                let interval = if rate.is_finite() && rate > 0.0 {1.0 / rate} else {0.0};
                let parameters = stream.parameters().clone();
                (DecodeInput::File(input), parameters, index, base, start, interval)
            }
            Encoded::Packets(source) => {
                let packets = source.source.lock().unwrap();
                if packets.parameters.medium() != kind {return Err(ffmpeg::Error::StreamNotFound)}
                let parameters = packets.parameters.clone();
                let (base, start, interval) = (packets.time_base, packets.start, source.interval);
                drop(packets);
                (DecodeInput::Packets(source, 0), parameters, 0, base, start, interval)
            }
        };
        let mut decoder = ffmpeg::codec::context::Context::from_parameters(parameters)?.decoder();
        decoder.set_threading(ffmpeg::threading::Config::count(1));
        decoder.set_packet_time_base(time_base);
        decoder.check(Check::CRC | Check::BITSTREAM | Check::BUFFER | Check::EXPLODE);
        let codec = decoder.codec().or_else(|| ffmpeg::decoder::find(decoder.id()))
            .ok_or(ffmpeg::Error::DecoderNotFound)?;
        let mut decoder = decoder.open_as(codec)?;
        let mut sound = (0, 0);
        let dimensions = if kind == Type::Video {
            let video = decoder.video()?;
            let dimensions = (video.width(), video.height());
            if dimensions.0 == 0 || dimensions.1 == 0 {return Err(ffmpeg::Error::InvalidData)}
            decoder = video.0;
            dimensions
        } else {
            let audio = decoder.audio()?;
            sound = (audio.rate(), audio.channels());
            decoder = audio.0;
            (0, 0)
        };
        let base = f64::from(time_base);
        Ok(Self {
            input, decoder, index, dimensions, sound, base, start, interval, end: 0.0, packet: 0.0,
            draining: false, feed: false, native_corrupted: false,
            rgba: ffmpeg::frame::Video::empty(),
        })
    }
    pub fn input(source: Input) -> Result<ffmpeg::format::context::Input, ffmpeg::Error> {
        Self::init()?;
        let io = ffmpeg::format::context::StreamIo::from_read_seek(source)?;
        let mut input = Self::open(io)?;
        Self::describe(&mut input)?;
        Ok(input)
    }
    fn describe(input: &mut ffmpeg::format::context::Input) -> Result<(), ffmpeg::Error> {
        let incomplete = input.streams().any(|stream| {
            let parameters = stream.parameters();
            let raw = unsafe {&*parameters.as_ptr()};
            match parameters.medium() {
                Type::Video => raw.width == 0 || raw.height == 0,
                Type::Audio => raw.sample_rate == 0 || raw.ch_layout.nb_channels == 0,
                _ => false,
            }
        });
        if incomplete {
            // Some containers keep dimensions/channel layout only in the codec header.
            // Probe bounded initial packets once; later fragments reuse the resulting parameters.
            unsafe {
                let context = &mut *input.as_mut_ptr();
                context.probesize = 262_144;
                context.max_analyze_duration = 1_000_000;
                let result = ffmpeg::ffi::avformat_find_stream_info(
                    input.as_mut_ptr(), std::ptr::null_mut(),
                );
                if result < 0 {return Err(ffmpeg::Error::from(result))}
            }
        }
        Ok(())
    }
    fn open(mut io: ffmpeg::format::context::StreamIo) -> Result<ffmpeg::format::context::Input, ffmpeg::Error> {
        unsafe {
            let mut context = ffmpeg::ffi::avformat_alloc_context();
            assert!(!context.is_null());
            (*context).pb = io.as_mut_ptr();
            (*context).flags |= ffmpeg::ffi::AVFMT_FLAG_CUSTOM_IO;
            let result = ffmpeg::ffi::avformat_open_input(
                &mut context, std::ptr::null(), std::ptr::null_mut(), std::ptr::null_mut(),
            );
            if result < 0 {return Err(ffmpeg::Error::from(result))}
            Ok(ffmpeg::format::context::Input::wrap_with_custom_io(context, io))
        }
    }
    fn start(stream: &ffmpeg::Stream<'_>) -> f64 {
        let start = if stream.start_time() == i64::MIN {0.0} else {
            stream.start_time() as f64 * f64::from(stream.time_base())
        };
        if start < 0.0 {start + Self::padding(stream)} else {start}
    }
    fn padding(stream: &ffmpeg::Stream<'_>) -> f64 {
        let parameters = stream.parameters();
        let raw = unsafe {&*parameters.as_ptr()};
        if raw.initial_padding <= 0 || raw.sample_rate <= 0 {return 0.0}
        let samples = ffmpeg::Rational(1, raw.sample_rate);
        let ticks = unsafe {
            ffmpeg::ffi::av_rescale_q(
                raw.initial_padding as i64, samples.into(), stream.time_base().into(),
            )
        };
        ticks as f64 * f64::from(stream.time_base())
    }
    pub fn dimensions(&self) -> (u32, u32) {self.dimensions}
    pub fn sound(&self) -> (u32, u16) {self.sound}
    pub fn format(&self) -> Option<Format> {
        match &self.input {
            DecodeInput::File(input) => Self::detect(input),
            DecodeInput::Packets(source, _) => source.source.lock().unwrap().format,
        }
    }
    fn detect(input: &ffmpeg::format::context::Input) -> Option<Format> {
        let (mut audio, mut video) = (None, None);
        for stream in input.streams() {
            if stream.disposition().contains(Disposition::ATTACHED_PIC) {continue}
            let parameters = stream.parameters();
            let track = match parameters.medium() {
                Type::Audio => &mut audio,
                Type::Video => &mut video,
                _ => continue,
            };
            if track.replace(parameters).is_some() {return None}
        }
        let codecs = (video.as_ref().map(|track| track.id()), audio.as_ref().map(|track| track.id()));
        let container = input.format();
        if container.name().split(',').any(|name| name == "mov") {
            let metadata = input.metadata();
            match metadata.get("major_brand")? {
                "qt  " => {
                    let parameters = video.as_ref()?;
                    let raw = unsafe {&*parameters.as_ptr()};
                    let standard = raw.profile == ffmpeg::ffi::AV_PROFILE_PRORES_STANDARD
                        || (raw.profile == ffmpeg::ffi::AV_PROFILE_UNKNOWN
                            && raw.codec_tag == u32::from_le_bytes(*b"apcn"));
                    if !standard {return None}
                    match codecs {
                        (Some(Id::PRORES), Some(Id::PCM_S16LE)) => Some(Format::Mov_ProRes422_PcmS16le),
                        (Some(Id::PRORES), None) => Some(Format::Mov_ProRes422),
                        _ => None,
                    }
                }
                "M4A " => match codecs {
                    (None, Some(Id::AAC)) => Some(Format::M4a_Aac),
                    (None, Some(Id::ALAC)) => Some(Format::M4a_Alac),
                    _ => None,
                },
                "isom" | "iso2" | "iso3" | "iso4" | "iso5" | "iso6" | "iso7" | "iso8"
                    | "iso9" | "mp41" | "mp42" | "M4V " => {
                    let parameters = video.as_ref()?;
                    let bitrate = usize::try_from(parameters.bit_rate()).ok()
                        .filter(|rate| *rate > 0)?;
                    match codecs {
                        (Some(Id::MPEG4), Some(Id::AAC)) => Some(Format::Mp4_Mpeg4p2_Aac {bitrate}),
                        (Some(Id::MPEG4), None) => Some(Format::Mp4_Mpeg4p2 {bitrate}),
                        _ => None,
                    }
                }
                _ => None,
            }
        } else if container.name().split(',').any(|name| name == "matroska") {
            // FFmpeg shares this demuxer with WebM. FFV1 identifies Matroska; Opus alone cannot.
            match codecs {
                (Some(Id::FFV1), Some(Id::FLAC)) => Some(Format::Mkv_Ffv1_Flac),
                (Some(Id::FFV1), Some(Id::PCM_F32LE)) => Some(Format::Mkv_Ffv1_PcmF32le),
                (Some(Id::FFV1), None) => Some(Format::Mkv_Ffv1),
                _ => None,
            }
        } else {
            match (container.name(), codecs) {
                ("wav", (None, Some(Id::PCM_S16LE))) => Some(Format::Wav_PcmS16le),
                ("wav", (None, Some(Id::PCM_F32LE))) => Some(Format::Wav_PcmF32le),
                ("flac", (None, Some(Id::FLAC))) => Some(Format::Flac_Flac),
                _ => None,
            }
        }
    }
    pub fn corrupted(&self) -> Vec<Range<Duration>> {
        match &self.input {
            DecodeInput::File(input) => {
                let kind = input.stream(self.index).unwrap().parameters().medium();
                Self::corruption(input.metadata().get("comment"), kind)
            }
            DecodeInput::Packets(source, _) => (*source.corrupted).clone(),
        }
    }
    fn corruption(comment: Option<&str>, kind: Type) -> Vec<Range<Duration>> {
        let Some(comment) = comment.and_then(|value| value.strip_prefix("media2-corrupted-v1;")) else {
            return Vec::new()
        };
        let prefix = if kind == Type::Audio {"a="} else {"v="};
        let Some(ranges) = comment.split(';').find_map(|part| part.strip_prefix(prefix)) else {
            return Vec::new()
        };
        let mut ranges: Vec<_> = ranges.split(',').filter_map(|range| {
            let (start, end) = range.split_once('-')?;
            let [Some(start), Some(end)] = [start, end].map(|value| {
                let nanos = value.parse::<u128>().ok()?;
                let seconds = u64::try_from(nanos / 1_000_000_000).ok()?;
                Some(Duration::new(seconds, (nanos % 1_000_000_000) as u32))
            }) else {return None};
            (start < end).then_some(start..end)
        }).collect();
        ranges.sort_by_key(|range| range.start);
        ranges
    }
    pub fn duration(&self, offset: f64) -> Option<(Duration, bool)> {
        let input = match &self.input {
            DecodeInput::File(input) => input,
            DecodeInput::Packets(source, _) => {
                return Duration::try_from_secs_f64((source.end - offset).max(0.0)).ok()
                    .map(|duration| (duration, source.finished))
            }
        };
        let stream = input.stream(self.index).unwrap();
        let ticks = stream.duration();
        let end = stream.metadata().get("DURATION").and_then(|value| {
            value.split(':').try_fold(0.0, |time, part| {
                part.parse::<f64>().ok().map(|part| time * 60.0 + part)
            })
        });
        let seconds = if ticks > 0 {
            Some(ticks as f64 * self.base + self.start - offset)
        } else if let Some(end) = end {Some(end - Self::padding(&stream) - offset)} else {
            let duration = input.duration();
            (duration > 0).then_some(duration as f64 / 1_000_000.0)
        };
        seconds.and_then(|seconds| Duration::try_from_secs_f64(seconds).ok())
            .map(|duration| (duration, ticks > 0 || end.is_some()))
    }
    pub fn measure_duration(&mut self) -> Result<Duration, ffmpeg::Error> {
        let input = match &mut self.input {
            DecodeInput::Packets(source, _) => {
                return Ok(Duration::from_secs_f64((source.end - self.start).max(0.0)))
            }
            DecodeInput::File(input) => input,
        };
        let mut end = self.start;
        loop {
            let mut packet = ffmpeg::Packet::empty();
            match packet.read(input) {
                Ok(()) if packet.stream() == self.index => {
                    if let Some(stamp) = packet.pts().or(packet.dts()) {
                        let duration = if packet.duration() > 0 {
                            packet.duration() as f64 * self.base
                        } else {self.interval};
                        end = end.max(stamp as f64 * self.base + duration);
                    }
                }
                Ok(()) => {},
                Err(ffmpeg::Error::Eof) => break,
                Err(error) => return Err(error),
            }
        }
        Ok(Duration::from_secs_f64((end - self.start).max(0.0)))
    }
    pub fn seek(&mut self, position: f64) -> Result<(), ffmpeg::Error> {
        let stamp = ((position + self.start) * 1_000_000.0) as i64;
        match &mut self.input {
            DecodeInput::File(input) => input.seek(stamp, ..stamp)?,
            DecodeInput::Packets(source, cursor) => {
                let packets = source.source.lock().unwrap();
                let stamp = (position + self.start) / self.base;
                *cursor = packets.packets[..source.length].iter().enumerate().filter(|(_, packet)| {
                    packet.is_key()
                        && packet.pts().or(packet.dts()).is_some_and(|pts| pts as f64 <= stamp)
                }).map(|(index, _)| index).next_back().unwrap_or(0);
            }
        }
        self.decoder.flush();
        self.draining = false;
        self.feed = false;
        self.native_corrupted = false;
        self.end = position;
        self.packet = position;
        Ok(())
    }
    pub fn next(&mut self, decoded: &mut ffmpeg::Frame) -> Result<Option<f64>, DecodeError> {
        loop {
            let result = if std::mem::take(&mut self.feed) {
                Err(ffmpeg::Error::Other {errno: ffmpeg::error::EAGAIN})
            } else {self.decoder.receive_frame(decoded)};
            match result {
                Ok(()) => {
                    let position = decoded.timestamp().map_or(self.end, |stamp| {
                        stamp as f64 * self.base - self.start
                    });
                    if decoded.packet().duration > 0 {
                        self.interval = decoded.packet().duration as f64 * self.base;
                    }
                    let frame = unsafe {&*decoded.as_ptr()};
                    if frame.nb_samples > 0 && frame.sample_rate > 0 {
                        self.interval = frame.nb_samples as f64 / frame.sample_rate as f64;
                    }
                    self.end = position + self.interval;
                    let marked = std::mem::take(&mut self.native_corrupted);
                    if marked || decoded.is_corrupt() || decoded.has_decode_errors() {
                        return Err(DecodeError::Corrupt(position))
                    }
                    return Ok(Some(position))
                }
                Err(ffmpeg::Error::Eof) => return Ok(None),
                Err(ffmpeg::Error::Other {errno}) if errno == ffmpeg::error::EAGAIN => {},
                Err(ffmpeg::Error::InvalidData) => {
                    self.feed = true;
                    self.native_corrupted = false;
                    self.end = self.end.max(self.packet + self.interval);
                    return Err(DecodeError::Corrupt(self.packet))
                }
                Err(error) => return Err(error.into()),
            }
            if self.draining {return Ok(None)}
            loop {
                let mut packet = ffmpeg::Packet::empty();
                let result = match &mut self.input {
                    DecodeInput::File(input) => packet.read(input),
                    DecodeInput::Packets(source, index) => {
                        if let Some(next) = source.packet(*index) {
                            packet = next;
                            packet.set_stream(self.index);
                            *index += 1;
                            Ok(())
                        } else if source.finished {Err(ffmpeg::Error::Eof)} else {return Ok(None)}
                    }
                };
                match result {
                    Ok(()) => {
                        if packet.stream() != self.index {continue}
                        self.packet = packet.pts().or(packet.dts()).map_or(self.end, |stamp| {
                            stamp as f64 * self.base - self.start
                        });
                        if packet.duration() > 0 {
                            self.interval = packet.duration() as f64 * self.base;
                        }
                        let mut marked = false;
                        if self.decoder.id() == ffmpeg::codec::Id::FFV1 {
                            let data = packet.data().unwrap_or_default();
                            let mut end = data.len();
                            while end >= 8 {
                                let footer = &data[end - 8..end];
                                let size = u32::from_be_bytes([
                                    0, footer[0], footer[1], footer[2],
                                ]) as usize;
                                if size == 0 || size + 8 > end || footer[3] > 2 {break}
                                marked |= footer[3] != 0;
                                end -= size + 8;
                            }
                            marked &= end == 0;
                            if marked {
                                // FFV1's native status is meaningful only in a complete protected
                                // slice chain. Validate both the configuration and every slice CRC.
                                let context = unsafe {&*self.decoder.as_ptr()};
                                let table = unsafe {
                                    ffmpeg::ffi::av_crc_get_table(
                                        ffmpeg::ffi::AVCRCId::AV_CRC_32_IEEE,
                                    )
                                };
                                marked = context.extradata_size >= 4 && unsafe {
                                    ffmpeg::ffi::av_crc(
                                        table, 0, context.extradata, context.extradata_size as usize,
                                    ) == 0
                                };
                                let mut end = data.len();
                                while marked && end >= 8 {
                                    let footer = &data[end - 8..end];
                                    let size = u32::from_be_bytes([
                                        0, footer[0], footer[1], footer[2],
                                    ]) as usize + 8;
                                    let slice = &data[end - size..end];
                                    marked = unsafe {
                                        ffmpeg::ffi::av_crc(table, 0, slice.as_ptr(), slice.len()) == 0
                                    };
                                    end -= size;
                                }
                            }
                        }
                        let result = if packet.is_corrupt() {
                            Err(ffmpeg::Error::InvalidData)
                        } else {self.decoder.send_packet(&packet)};
                        match result {
                            // Feed native-status packets before flagging their decoded frame so
                            // FFV1's adaptive coding state remains valid for following frames.
                            Ok(()) => self.native_corrupted = marked,
                            Err(ffmpeg::Error::InvalidData) => {
                                self.end = self.end.max(self.packet + self.interval);
                                return Err(DecodeError::Corrupt(self.packet))
                            }
                            Err(error) => return Err(error.into()),
                        }
                    }
                    Err(ffmpeg::Error::Eof) => {
                        self.draining = true;
                        self.decoder.send_eof()?;
                    }
                    Err(error) => return Err(error.into()),
                }
                break
            }
        }
    }
    pub fn image(&mut self, decoded: &ffmpeg::frame::Video) -> Result<Image, ffmpeg::Error> {
        let (width, height) = (decoded.width(), decoded.height());
        let mut scaler = ffmpeg::software::scaling::context::Context::get(
            decoded.format(), width, height, ffmpeg::format::Pixel::RGBA, width, height,
            ffmpeg::software::scaling::flag::Flags::BILINEAR,
        )?;
        if self.rgba.width() != width || self.rgba.height() != height {
            self.rgba = ffmpeg::frame::Video::empty();
        }
        scaler.run(decoded, &mut self.rgba)?;
        let mut pixels = Vec::with_capacity(width as usize * height as usize);
        for row in self.rgba.data(0).chunks(self.rgba.stride(0)).take(height as usize) {
            pixels.extend_from_slice(bytemuck::cast_slice(&row[..width as usize * 4]));
        }
        Ok(Image::new((width, height), pixels))
    }
}

enum DecodeInput {
    File(ffmpeg::format::context::Input),
    Packets(PacketSource, usize),
}

enum Container {
    Unknown,
    Mp4 {fragment: Vec<u8>, initialized: bool, position: u64, origin: u64},
    Matroska {segment: bool, initialized: bool},
    Wave {remaining: Option<u64>, alignment: usize, position: i64, padding: bool},
    Flac {
        parser: *mut ffmpeg::ffi::AVCodecParserContext,
        context: Option<ffmpeg::codec::Context>,
        position: i64,
    },
    Failed(String),
}
impl Drop for Container {
    fn drop(&mut self) {
        if let Self::Flac {parser, ..} = self && !parser.is_null() {
            unsafe {ffmpeg::ffi::av_parser_close(*parser);}
        }
    }
}
