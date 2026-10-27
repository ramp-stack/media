use bytemuck::{Pod, Zeroable};
use ffmpeg_next as ffmpeg;

use std::fmt;
use std::io::{Read, Seek, Write};
use std::ops::{Add, Range};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::media::{Clip, CursorRead, Export, Join, Timeline, Track};
use crate::{Cursor, Encoder, Format, Media, Player, Video};

/// One channel's PCM amplitude, conventionally between -1.0 and 1.0.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Sample(pub f32);

/// Samples are interleaved: all channels at one instant precede the next instant.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Sound {
    pub channels: u16,
    pub samples: Vec<Sample>,
}
impl AsRef<[f32]> for Sound {
    fn as_ref(&self) -> &[f32] {bytemuck::cast_slice(&self.samples)}
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct AudioFrame {
    pub sound: Sound,
    /// Elapsed time from the start of the source that produced this frame.
    pub timestamp: Duration,
    pub duration: Duration,
    /// True when source content was damaged, including interpolated or replacement samples.
    pub corrupted: bool,
}
impl AudioFrame {
    pub fn sample_rate(&self) -> u32 {
        assert!(self.sound.channels > 0 && !self.duration.is_zero());
        assert_eq!(self.sound.samples.len() % self.sound.channels as usize, 0);
        let samples = self.sound.samples.len() / self.sound.channels as usize;
        let rate = (samples as f64 / self.duration.as_secs_f64()).round();
        assert!(rate >= 1.0 && rate <= i32::MAX as f64);
        rate as u32
    }
}

/// Cloning shares the clip list, source data, and decoding cache without copying them.
/// Each clone has an independent timeline; attached cursors follow edits to their original timeline.
pub struct Audio {
    timeline: Arc<Mutex<Timeline<AudioFrame>>>,
}
impl Audio {
    pub fn new() -> Self {Self::default()}
    pub fn read<R: Read + Seek + Send + 'static>(reader: R) -> Result<Self, String> {
        Media::read(reader).map(|media| media.audio)
    }
    pub fn duration(&self) -> Duration {self.timeline.lock().unwrap().duration()}
    /// Original container and codecs shared by the remaining encoded sources, including both tracks.
    /// Raw edits are ignored. Missing, conflicting or unrepresentable source metadata returns None.
    /// Bitrates come from source metadata and may differ from the original encoder's target.
    pub fn format(&self) -> Option<Format> {Clip::format(self.timeline.lock().unwrap().clips())}
    pub fn cursor(&self) -> Cursor {Cursor::from(self)}
    pub fn player(&self) -> Player {Player::new(self.cursor())}
    /// Insert at the frame's timestamp, shifting later content and attached cursors forward.
    pub fn insert(&mut self, frame: AudioFrame) {
        let clip = Clip::from_audio(frame);
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
    /// Immutable sample-aligned chunks, shortened at gaps or changes in corruption state.
    /// Damage up to 2 ms is interpolated between healthy neighbors; other damage produces silence.
    /// Repaired portions remain flagged. Edits require a new range request.
    pub fn frames(&self, range: Range<Duration>, duration: Duration) -> Arc<Vec<Arc<AudioFrame>>> {
        self.cursor().audio_frames(range, duration)
    }
    pub fn encode(&self, format: Format) -> Result<Encoder, String> {
        Encoder::new(self.clone(), Video::default(), format)
    }
    pub fn write<W: Write + Seek + Send + 'static>(&self, writer: W, format: Format) -> Result<(), String> {
        Encoder::write(self, &Video::default(), writer, format)
    }
}
impl Clone for Audio {
    fn clone(&self) -> Self {
        Self {timeline: Arc::new(Mutex::new(self.timeline.lock().unwrap().clone()))}
    }
}
impl Default for Audio {
    fn default() -> Self {
        Self {timeline: Arc::new(Mutex::new(Timeline::new(ffmpeg::media::Type::Audio, None)))}
    }
}
impl fmt::Debug for Audio {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let timeline = self.timeline.lock().unwrap();
        formatter.debug_struct("Audio").field("clips", timeline.clips())
            .field("duration", &timeline.duration()).finish_non_exhaustive()
    }
}
impl From<Clip> for Audio {
    fn from(clip: Clip) -> Self {
        Self {timeline: Arc::new(Mutex::new(Timeline::new(ffmpeg::media::Type::Audio, Some(clip))))}
    }
}
impl From<AudioFrame> for Audio {
    fn from(frame: AudioFrame) -> Self {Self::from(Clip::from_audio(frame))}
}
impl Add for Audio {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        let range = self.duration()..self.duration() + other.duration();
        self.join(other, range)
    }
}
impl Join for Audio {
    fn join(self, other: Self, range: Range<Duration>) -> Self {
        let clips = other.clips();
        self.timeline.lock().unwrap().join(clips, range, |frame| {
            frame.timestamp..frame.timestamp + frame.duration
        });
        self
    }
}

impl Track<AudioFrame> for Audio {
    fn timeline(&self) -> Arc<Mutex<Timeline<AudioFrame>>> {self.timeline.clone()}
    fn span(frame: &AudioFrame) -> Range<Duration> {frame.timestamp..frame.timestamp + frame.duration}
}
