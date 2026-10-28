use ffmpeg_next as ffmpeg;

use std::fmt;
use std::io::{Read, Seek, Write};
use std::ops::{Add, Range};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::media::{Clip, CursorRead, Export, Join, Timeline, Track};
use crate::{Audio, Cursor, Encoder, Format, Frames, Image, Media, Player};

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct VideoFrame {
    pub image: Image,
    /// Elapsed time from the start of the source that produced this frame.
    pub timestamp: Duration,
    pub duration: Duration,
    /// True when source content was damaged, including held or replacement images.
    pub corrupted: bool,
}

/// Cloning shares the clip list, source data, and decoding cache without copying them.
/// Each clone has an independent timeline; attached cursors follow edits to their original timeline.
pub struct Video {
    timeline: Arc<Mutex<Timeline<VideoFrame>>>,
}
impl Video {
    pub fn new() -> Self {Self::default()}
    pub fn read<R: Read + Seek + Send + 'static>(reader: R) -> Result<Self, String> {
        Media::read(reader).map(|media| media.video)
    }
    pub fn duration(&self) -> Duration {self.timeline.lock().unwrap().duration()}
    /// Original container and codecs shared by the remaining encoded sources, including both tracks.
    /// Raw edits are ignored. Missing, conflicting or unrepresentable source metadata returns None.
    /// Bitrates come from source metadata and may differ from the original encoder's target.
    pub fn format(&self) -> Option<Format> {Clip::format(self.timeline.lock().unwrap().clips())}
    pub fn dimensions(&self) -> Option<(u32, u32)> {
        self.timeline.lock().unwrap().clips().first().map(|clip| clip.dimensions)
    }
    pub fn cursor(&self) -> Cursor {Cursor::from(self)}
    pub fn player(&self) -> Player {Player::new(self.cursor())}
    /// Insert at the frame's timestamp, shifting later content and attached cursors forward.
    pub fn insert(&mut self, frame: VideoFrame) {
        let clip = Clip::from_video(frame);
        let space = clip.start..clip.end;
        Track::insert(self, clip, space);
    }
    /// Remove the interval and shift later content earlier. Time outside the track is ignored.
    pub fn remove(&mut self, range: Range<Duration>) {
        assert!(range.start <= range.end);
        let end = range.end.min(self.duration());
        if range.start >= end {return}
        Track::remove(self, range.start..end);
    }
    /// Whole frames overlapping the half-open range. Later edits require a new range request.
    /// Damage up to 100 ms holds the last healthy image; otherwise it shows an opaque black image.
    /// Replacements remain flagged, including when the source file already records the damage.
    pub fn frames(&self, range: Range<Duration>) -> Frames<VideoFrame> {
        self.cursor().video_frames(range)
    }
    pub fn encode(&self, format: Format) -> Result<Encoder, String> {
        Encoder::new(Audio::default(), self.clone(), format)
    }
    pub fn write<W: Write + Seek + Send + 'static>(&self, writer: W, format: Format) -> Result<(), String> {
        Encoder::write(&Audio::default(), self, writer, format)
    }
}
impl Clone for Video {
    fn clone(&self) -> Self {
        Self {timeline: Arc::new(Mutex::new(self.timeline.lock().unwrap().clone()))}
    }
}
impl Default for Video {
    fn default() -> Self {
        Self {timeline: Arc::new(Mutex::new(Timeline::new(ffmpeg::media::Type::Video, None)))}
    }
}
impl fmt::Debug for Video {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let timeline = self.timeline.lock().unwrap();
        formatter.debug_struct("Video").field("clips", timeline.clips())
            .field("duration", &timeline.duration()).finish_non_exhaustive()
    }
}
impl From<Clip> for Video {
    fn from(clip: Clip) -> Self {
        Self {timeline: Arc::new(Mutex::new(Timeline::new(ffmpeg::media::Type::Video, Some(clip))))}
    }
}
impl From<VideoFrame> for Video {
    fn from(frame: VideoFrame) -> Self {Self::from(Clip::from_video(frame))}
}
impl Add for Video {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        let range = self.duration()..self.duration() + other.duration();
        self.join(other, range)
    }
}
impl Join for Video {
    fn join(self, other: Self, range: Range<Duration>) -> Self {
        let clips = other.clips();
        self.timeline.lock().unwrap().join(clips, range, |frame| {
            frame.timestamp..frame.timestamp + frame.duration
        });
        self
    }
}

impl Track<VideoFrame> for Video {
    fn timeline(&self) -> Arc<Mutex<Timeline<VideoFrame>>> {self.timeline.clone()}
    fn span(frame: &VideoFrame) -> Range<Duration> {frame.timestamp..frame.timestamp + frame.duration}
}
