use bytemuck::{Pod, Zeroable};
use ffmpeg_next as ffmpeg;
use image::{ColorType, ImageEncoder};
use image::codecs::{bmp, gif, ico, jpeg, png, qoi, tiff, webp};

use std::io::{Read, Seek, Write};
use std::ops::{Add, Deref, DerefMut, Range};
use std::sync::Arc;
use std::time::{Duration, Instant};

mod media;
use media::{Clip, CursorRead, Encoded, Export, Input, Track};
mod audio;
pub use audio::{Audio, AudioFrame, Sample, Sound};
mod video;
pub use video::{Video, VideoFrame};
mod cursor;
pub use cursor::Cursor;
mod decoder;
pub use decoder::Decoder;
use decoder::FrameDecoder;
mod encoder;
pub use encoder::{Encoder, Format};
mod player;
pub use player::{Frame, PlaybackMode, Player};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Tracks {Audio, Video, Both}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImageFormat {Png, Jpeg, Webp, Gif, Bmp, Ico, Tiff, Qoi}
impl ImageFormat {
    pub fn extension(self) -> &'static str {
        match self {
            Self::Png => "png", Self::Jpeg => "jpg", Self::Webp => "webp", Self::Gif => "gif",
            Self::Bmp => "bmp", Self::Ico => "ico", Self::Tiff => "tiff", Self::Qoi => "qoi",
        }
    }
}

/// Straight-alpha RGBA8 in sRGB color space.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Pod, Zeroable)]
pub struct Pixel {
    pub red: u8,
    pub green: u8,
    pub blue: u8,
    pub alpha: u8,
}

/// Each track retains its own timestamp relative to the same source start.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MediaFrame {
    pub video: Option<VideoFrame>,
    pub audio: Option<AudioFrame>,
}

/// Pixels are ordered left to right, then top to bottom.
/// Clones share their pixels until mutable access copies a shared buffer.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Image {
    pub dimensions: (u32, u32),
    pixels: Arc<Vec<Pixel>>,
}
impl Image {
    pub fn new(dimensions: (u32, u32), pixels: Vec<Pixel>) -> Self {
        Self {dimensions, pixels: Arc::new(pixels)}
    }
    pub fn read<R: Read>(mut reader: R) -> Result<Self, String> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).map_err(|error| error.to_string())?;
        let image = image::load_from_memory(&bytes).map_err(|error| error.to_string())?.into_rgba8();
        let dimensions = image.dimensions();
        let pixels = bytemuck::cast_slice(image.as_raw()).to_vec();
        Ok(Self::new(dimensions, pixels))
    }
    pub fn write<W: Write>(&self, mut writer: W, format: ImageFormat) -> Result<(), String> {
        let (width, height) = self.dimensions;
        assert!(width > 0 && height > 0);
        assert_eq!(self.pixels.len(), width as usize * height as usize);
        let pixels = self.as_ref();
        let color = ColorType::Rgba8;
        let result = match format {
            ImageFormat::Png => png::PngEncoder::new(writer).write_image(pixels, width, height, color),
            ImageFormat::Jpeg => jpeg::JpegEncoder::new(writer)
                .write_image(pixels, width, height, color),
            ImageFormat::Webp => webp::WebPEncoder::new_lossless(writer)
                .write_image(pixels, width, height, color),
            ImageFormat::Gif => {
                // Encode before writing so a failed trailer write is reported to the caller.
                let mut bytes = Vec::new();
                gif::GifEncoder::new(&mut bytes).encode(pixels, width, height, color)
                    .map_err(|error| error.to_string())?;
                return writer.write_all(&bytes).map_err(|error| error.to_string())
            }
            ImageFormat::Bmp => bmp::BmpEncoder::new(&mut writer)
                .write_image(pixels, width, height, color),
            ImageFormat::Ico => ico::IcoEncoder::new(writer).write_image(pixels, width, height, color),
            ImageFormat::Qoi => qoi::QoiEncoder::new(writer).write_image(pixels, width, height, color),
            ImageFormat::Tiff => {
                let mut bytes = std::io::Cursor::new(Vec::new());
                tiff::TiffEncoder::new(&mut bytes).write_image(pixels, width, height, color)
                    .map_err(|error| error.to_string())?;
                return writer.write_all(bytes.get_ref()).map_err(|error| error.to_string())
            }
        };
        result.map_err(|error| error.to_string())
    }
}
impl Deref for Image {
    type Target = Vec<Pixel>;
    fn deref(&self) -> &Vec<Pixel> {&self.pixels}
}
impl DerefMut for Image {
    fn deref_mut(&mut self) -> &mut Vec<Pixel> {Arc::make_mut(&mut self.pixels)}
}
impl AsRef<[u8]> for Image {
    fn as_ref(&self) -> &[u8] {bytemuck::cast_slice(&self.pixels)}
}

#[derive(Clone, Debug, Default)]
pub struct Media {
    pub audio: Audio,
    pub video: Video,
}
impl Media {
    pub fn new() -> Self {Self::default()}
    /// Retains the reader for lazy access. Clones share the source with independent read cursors.
    /// Inputs without duration metadata require a packet scan, without decoding their frames.
    pub fn read<R: Read + Seek + Send + 'static>(reader: R) -> Result<Self, String> {
        let source = Encoded::File(Input::new(reader).map_err(|error| error.to_string())?);
        let mut clips = Vec::new();
        let mut found = false;
        for kind in [ffmpeg::media::Type::Audio, ffmpeg::media::Type::Video] {
            match FrameDecoder::new(source.clone(), kind) {
                Ok(decoder) => {
                    found = true;
                    if let Some(clip) = Clip::new(source.clone(), kind, decoder)? {
                        clips.push((kind, clip));
                    }
                }
                Err(ffmpeg::Error::StreamNotFound) => {},
                Err(error) => return Err(error.to_string()),
            }
        }
        if !found {return Err("Media has no audio or video track".into())}
        let origin = clips.iter().map(|(_, clip)| clip.offset).reduce(f64::min).unwrap_or(0.0);
        let mut media = Self::default();
        for (kind, mut clip) in clips {
            clip.end += Duration::from_secs_f64((clip.offset - origin).max(0.0));
            clip.offset = origin;
            if kind == ffmpeg::media::Type::Audio {media.audio = Audio::from(clip)}
            else {media.video = Video::from(clip)}
        }
        Ok(media)
    }
    pub fn duration(&self) -> Duration {self.audio.duration().max(self.video.duration())}
    pub fn cursor(&self) -> Cursor {Cursor::from(self)}
    pub fn player(&self) -> Player {Player::new(self.cursor())}
    /// Remove the same time interval from both tracks, closing the gap.
    pub fn remove(&mut self, range: Range<Duration>) {
        assert!(range.start <= range.end);
        let audio = self.audio.timeline();
        let video = self.video.timeline();
        let mut audio = audio.lock().unwrap();
        let mut video = video.lock().unwrap();
        let end = range.end.min(audio.duration().max(video.duration()));
        if range.start >= end {return}
        let range = range.start..end;
        let instant = Instant::now();
        audio.remove(range.clone(), instant, Audio::span);
        video.remove(range, instant, Video::span);
    }
    pub fn frames(&self, range: Range<Duration>, duration: Duration) -> (Arc<Vec<Arc<AudioFrame>>>, Arc<Vec<Arc<VideoFrame>>>) {
        let mut cursor = self.cursor();
        (cursor.audio_frames(range.clone(), duration), cursor.video_frames(range))
    }
    /// Offset source timestamps by placement, then insert time before existing content.
    /// When both tracks are present, they share an insertion span. An empty frame does nothing.
    pub fn insert(&mut self, frame: MediaFrame, placement: Duration) {
        match (frame.audio, frame.video) {
            (Some(mut frame), None) => {
                frame.timestamp += placement;
                self.audio.insert(frame);
            }
            (None, Some(mut frame)) => {
                frame.timestamp += placement;
                self.video.insert(frame);
            }
            (Some(mut audio), Some(mut video)) => {
                audio.timestamp += placement;
                video.timestamp += placement;
                let audio = Clip::from_audio(audio);
                let video = Clip::from_video(video);
                let space = audio.start.min(video.start)..audio.end.max(video.end);
                let audio_track = self.audio.timeline();
                let video_track = self.video.timeline();
                let mut audio_track = audio_track.lock().unwrap();
                let mut video_track = video_track.lock().unwrap();
                let instant = Instant::now();
                // The longer track determines whether this paired edit rebases a shared cursor.
                if audio_track.duration() >= video_track.duration() {
                    audio_track.insert(audio, space.clone(), instant, Audio::span);
                    video_track.insert(video, space, instant, Video::span);
                } else {
                    video_track.insert(video, space.clone(), instant, Video::span);
                    audio_track.insert(audio, space, instant, Audio::span);
                }
            }
            (None, None) => {},
        }
    }
    /// Exports a fixed timeline snapshot, reusing compatible encoded packets where possible.
    pub fn encode(&self, format: Format) -> Result<Encoder, String> {
        Encoder::new(self.audio.clone(), self.video.clone(), format)
    }
    /// Writes and finalizes directly into the owned destination.
    pub fn write<W: Write + Seek + Send + 'static>(&self, writer: W, format: Format) -> Result<(), String> {
        Encoder::write(&self.audio, &self.video, writer, format)
    }
}
impl AsRef<Media> for Media {
    fn as_ref(&self) -> &Media {self}
}
impl From<Audio> for Media {
    fn from(audio: Audio) -> Self {Self {audio, video: Video::default()}}
}
impl From<Video> for Media {
    fn from(video: Video) -> Self {Self {audio: Audio::default(), video}}
}
impl Add for Media {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        let audio_clips = other.audio.clips();
        let video_clips = other.video.clips();
        let added = other.duration();
        let audio = self.audio.timeline();
        let video = self.video.timeline();
        let mut audio = audio.lock().unwrap();
        let mut video = video.lock().unwrap();
        let duration = audio.duration().max(video.duration());
        let range = duration..duration + added;
        audio.join(audio_clips, range.clone(), Audio::span);
        video.join(video_clips, range, Video::span);
        drop(video);
        drop(audio);
        self
    }
}
