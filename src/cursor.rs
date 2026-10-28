use ffmpeg_next as ffmpeg;

use std::collections::VecDeque;
use std::ops::Range;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use crate::decoder::{DecodeError, FrameDecoder};
use crate::media::{Cache, Clip, Content, CursorRead, Playhead, Timeline, Track};
use crate::{Audio, AudioFrame, Frames, Image, Media, Pixel, Sample, Sound, Video, VideoFrame};

type FrameRead<'a> = (Range<f64>, &'a Clip, Option<Range<f64>>);

/// Independent decoder state over live editable tracks. Cloning a track creates a separate edit branch;
/// creating another cursor on the same track shares edits and the bounded decoded-frame cache.
/// Cloning a cursor creates an independent cursor at the same position over those same tracks.
pub struct Cursor {
    audio: Option<Arc<Mutex<Timeline<AudioFrame>>>>,
    video: Option<Arc<Mutex<Timeline<VideoFrame>>>>,
    playhead: Arc<Mutex<Playhead>>,
    audio_reader: Reader,
    video_reader: Reader,
    audio_position: Option<Duration>,
    video_position: Option<Duration>,
    audio_pending: Option<Arc<AudioFrame>>,
    video_pending: Option<Arc<VideoFrame>>,
    revisions: (u64, u64),
}
impl Cursor {
    pub fn position(&self) -> Duration {self.playhead().time(Instant::now())}
    pub fn duration(&self) -> Duration {self.playhead().end}
    pub fn seek(&mut self, position: Duration) {
        self.refresh();
        let mut playhead = self.playhead();
        playhead.position = if playhead.live {position} else {position.min(playhead.end)};
        if playhead.rendered.is_some() {playhead.rendered = Some(playhead.position);}
        if playhead.started.is_some() {playhead.started = Some(Instant::now());}
    }
    /// Read samples starting at the current position without advancing it.
    pub fn audio(&mut self, duration: Duration) -> Frames<AudioFrame> {
        loop {
            let head = self.playhead();
            let (position, rebased) = (head.time(Instant::now()), head.rebased);
            drop(head);
            let frames = self.audio_frames(position..position + duration, duration);
            if self.playhead().rebased == rebased {return frames}
        }
    }
    /// Read the image covering the current position, or the next image after a gap.
    pub fn video(&mut self) -> Option<Arc<VideoFrame>> {
        loop {
            self.refresh();
            let head = self.playhead();
            let (position, rebased) = (head.time(Instant::now()), head.rebased);
            drop(head);
            let frames = self.video_frames(position..position + Duration::from_nanos(1));
            if self.playhead().rebased != rebased {continue}
            if let Some(frame) = frames.first() {return Some(frame.clone())}
            let timeline = self.video.clone()?;
            let (clips, revision) = {
                let track = timeline.lock().unwrap();
                (track.clips().clone(), track.revision())
            };
            if revision != self.revisions.1 {continue}
            self.video_reader.seek(&clips, position);
            self.video_position = None;
            self.video_pending = self.video_reader.video(&clips);
            let next = self.video_pending.as_ref().map(|frame| frame.timestamp);
            let end = next.map_or(Duration::MAX, |time| time + Duration::from_nanos(1));
            if !timeline.lock().unwrap().unchanged(revision, position..end) {continue}
            if self.playhead().rebased != rebased {continue}
            let next = next?;
            let frames = self.video_frames(next..next + Duration::from_nanos(1));
            let head = self.validate((0, revision), Duration::ZERO..Duration::ZERO, position..end);
            if head.is_some_and(|head| head.rebased == rebased) {return frames.first().cloned()}
        }
    }

    fn new(audio: Option<Arc<Mutex<Timeline<AudioFrame>>>>, video: Option<Arc<Mutex<Timeline<VideoFrame>>>>) -> Self {
        let mut audio_track = audio.as_ref().map(|track| track.lock().unwrap());
        let mut video_track = video.as_ref().map(|track| track.lock().unwrap());
        let ends = (
            audio_track.as_ref().map_or(Duration::ZERO, |track| track.duration()),
            video_track.as_ref().map_or(Duration::ZERO, |track| track.duration()),
        );
        let revisions = (
            audio_track.as_ref().map_or(0, |track| track.revision()),
            video_track.as_ref().map_or(0, |track| track.revision()),
        );
        let playhead = Arc::new(Mutex::new(Playhead {
            position: Duration::ZERO, started: None, speed: 1.0, live: false,
            end: ends.0.max(ends.1), ends, rendered: None,
            revisions, rebased: 0, last_edit: None,
        }));
        if let Some(track) = &mut audio_track {track.attach(&playhead);}
        if let Some(track) = &mut video_track {track.attach(&playhead);}
        drop(video_track);
        drop(audio_track);
        Self {
            audio, video, playhead, revisions,
            audio_reader: Reader::new(ffmpeg::media::Type::Audio),
            video_reader: Reader::new(ffmpeg::media::Type::Video),
            audio_position: None, video_position: None, audio_pending: None, video_pending: None,
        }
    }
    fn audio_window(&mut self, clips: &[Clip], cache: &mut Cache<AudioFrame>, range: Range<Duration>) -> Frames<AudioFrame> {
        if let Some(frames) = cache.block(&range) {return frames}
        if let Some(frames) = cache.cover(&range, |frame| {
            frame.timestamp..frame.timestamp + frame.duration
        }) {return frames}
        if self.audio_position != Some(range.start) {
            self.audio_reader.seek(clips, range.start);
            self.audio_pending = None;
        }
        self.audio_position = None;
        let mut frames = Vec::new();
        loop {
            let frame = match self.audio_pending.take() {
                Some(frame) => Some(frame),
                None => self.audio_reader.audio(clips),
            };
            let Some(frame) = frame else {break};
            let frame = cache.share(frame, |frame| frame.timestamp..frame.timestamp + frame.duration);
            if frame.timestamp >= range.end {self.audio_pending = Some(frame); break}
            if frame.timestamp + frame.duration > range.start {frames.push(frame.clone())}
            if frame.timestamp + frame.duration >= range.end {
                self.audio_pending = Some(frame);
                break
            }
        }
        let bytes = frames.iter().map(|frame| {
            frame.sound.samples.len() * std::mem::size_of::<Sample>()
        }).sum();
        let frames = Arc::new(frames);
        self.audio_position = Some(range.end);
        cache.cache_block(range, frames.clone(), bytes);
        frames
    }
    fn video_window(&mut self, clips: &[Clip], cache: &mut Cache<VideoFrame>, range: Range<Duration>) -> Frames<VideoFrame> {
        if let Some(frames) = cache.block(&range) {return frames}
        if let Some(frames) = cache.cover(&range, |frame| {
            frame.timestamp..frame.timestamp + frame.duration
        }) {return frames}
        if self.video_position != Some(range.start) {
            self.video_reader.seek(clips, range.start);
            self.video_pending = None;
        }
        self.video_position = None;
        let mut frames = Vec::new();
        loop {
            let frame = match self.video_pending.take() {
                Some(frame) => Some(frame),
                None => self.video_reader.video(clips),
            };
            let Some(frame) = frame else {break};
            let frame = cache.share(frame, |frame| frame.timestamp..frame.timestamp + frame.duration);
            if frame.timestamp >= range.end {self.video_pending = Some(frame); break}
            if frame.timestamp + frame.duration > range.start {frames.push(frame.clone())}
            if frame.timestamp + frame.duration >= range.end {
                self.video_pending = Some(frame);
                break
            }
        }
        let bytes = frames.iter().map(|frame| {
            frame.image.len() * std::mem::size_of::<crate::Pixel>()
        }).sum();
        let frames = Arc::new(frames);
        self.video_position = Some(range.end);
        cache.cache_block(range, frames.clone(), bytes);
        frames
    }
    fn chunk_audio(frames: &[Arc<AudioFrame>], range: Range<Duration>, duration: Duration, cache: &Cache<AudioFrame>) -> Vec<Arc<AudioFrame>> {
        let mut output: Vec<Arc<AudioFrame>> = Vec::new();
        let mut current: Option<AudioFrame> = None;
        for source in frames {
            let channels = source.sound.channels as usize;
            let total = source.sound.samples.len() / channels;
            let rate = source.sample_rate();
            let wanted = (duration.as_secs_f64() * rate as f64).round().max(1.0) as usize;
            // Nanosecond-rounded frame times must choose the same side of a half-sample boundary.
            let rounding = (total as f64 / source.duration.as_secs_f64() * 1e-9).min(0.25);
            let sample_at = |position: Duration| {
                (position.saturating_sub(source.timestamp).as_secs_f64()
                    / source.duration.as_secs_f64() * total as f64 + rounding).round() as usize
            };
            let end = range.end.min(source.timestamp + source.duration);
            let count = sample_at(end);
            let mut position = range.start.max(source.timestamp);
            if let Some(previous) = output.last() {
                position = position.max(previous.timestamp + previous.duration);
            }
            if let Some(frame) = &current {
                position = position.max(frame.timestamp + frame.duration);
            }
            let mut offset = sample_at(position);
            while offset < count {
                let timestamp = source.timestamp
                    + source.duration.mul_f64(offset as f64 / total as f64);
                if current.as_ref().is_some_and(|frame| {
                    frame.sound.channels != source.sound.channels || frame.sample_rate() != rate
                        || frame.corrupted != source.corrupted
                        || (frame.timestamp + frame.duration).abs_diff(timestamp)
                            > Duration::from_nanos(1)
                }) {output.push(Arc::new(current.take().unwrap()))}
                if current.is_none() {
                    let cached = cache.snapshots().flat_map(|(_, frames)| frames.iter())
                        .find(|frame| {
                            frame.timestamp == timestamp && frame.corrupted == source.corrupted
                                && frame.sound.channels as usize == channels
                                && frame.sound.samples.len() / channels == wanted
                                && frame.sample_rate() == rate
                                && frame.timestamp + frame.duration <= range.end
                        });
                    if let Some(frame) = cached {
                        position = frame.timestamp + frame.duration;
                        output.push(frame.clone());
                        offset = sample_at(position);
                        continue
                    }
                    if offset == 0 && count == total && total == wanted {
                        output.push(source.clone());
                        break
                    }
                    current = Some(AudioFrame {
                        sound: Sound {channels: source.sound.channels, samples: Vec::new()},
                        timestamp, duration: Duration::ZERO, corrupted: source.corrupted,
                    });
                }
                let frame = current.as_mut().unwrap();
                let copied = (wanted - frame.sound.samples.len() / channels).min(count - offset);
                frame.sound.samples.extend_from_slice(
                    &source.sound.samples[offset * channels..(offset + copied) * channels],
                );
                offset += copied;
                let end = source.timestamp + source.duration.mul_f64(offset as f64 / total as f64);
                frame.duration = end - frame.timestamp;
                if frame.sound.samples.len() / channels == wanted {
                    output.push(Arc::new(current.take().unwrap()));
                }
            }
        }
        if let Some(frame) = current {output.push(Arc::new(frame))}
        output
    }
}
impl Clone for Cursor {
    fn clone(&self) -> Self {
        let cursor = Self::new(self.audio.clone(), self.video.clone());
        let _audio = self.audio.as_ref().map(|track| track.lock().unwrap());
        let _video = self.video.as_ref().map(|track| track.lock().unwrap());
        cursor.playhead.lock().unwrap().position = self.playhead.lock().unwrap().time(Instant::now());
        cursor
    }
}
impl From<&Media> for Cursor {
    fn from(media: &Media) -> Self {
        Self::new(Some(media.audio.timeline()), Some(media.video.timeline()))
    }
}
impl From<&Audio> for Cursor {
    fn from(audio: &Audio) -> Self {Self::new(Some(audio.timeline()), None)}
}
impl From<&Video> for Cursor {
    fn from(video: &Video) -> Self {Self::new(None, Some(video.timeline()))}
}
impl CursorRead for Cursor {
    fn has_audio(&self) -> bool {
        self.audio.as_ref().is_some_and(|audio| !audio.lock().unwrap().clips().is_empty())
    }
    fn has_video(&self) -> bool {
        self.video.as_ref().is_some_and(|video| !video.lock().unwrap().clips().is_empty())
    }
    fn playhead(&self) -> MutexGuard<'_, Playhead> {
        // Lock tracks in this order before the playhead; never acquire a track while holding it.
        let _audio = self.audio.as_ref().map(|track| track.lock().unwrap());
        let _video = self.video.as_ref().map(|track| track.lock().unwrap());
        self.playhead.lock().unwrap()
    }
    fn validate(&self, revisions: (u64, u64), audio: Range<Duration>, video: Range<Duration>) -> Option<MutexGuard<'_, Playhead>> {
        let audio_track = self.audio.as_ref().map(|track| track.lock().unwrap());
        let video_track = self.video.as_ref().map(|track| track.lock().unwrap());
        if audio_track.as_ref().is_some_and(|track| !track.unchanged(revisions.0, audio))
            || video_track.as_ref().is_some_and(|track| !track.unchanged(revisions.1, video)) {
            return None
        }
        Some(self.playhead.lock().unwrap())
    }
    fn refresh(&mut self) {
        let revisions = {
            let audio = self.audio.as_ref().map(|track| track.lock().unwrap());
            let video = self.video.as_ref().map(|track| track.lock().unwrap());
            (audio.as_ref().map_or(0, |track| track.revision()),
                video.as_ref().map_or(0, |track| track.revision()))
        };
        if revisions.0 != self.revisions.0 {
            self.audio_reader = Reader::new(ffmpeg::media::Type::Audio);
            self.audio_position = None;
            self.audio_pending = None;
        }
        if revisions.1 != self.revisions.1 {
            self.video_reader = Reader::new(ffmpeg::media::Type::Video);
            self.video_position = None;
            self.video_pending = None;
        }
        self.revisions = revisions;
    }
    fn audio_frames(&mut self, range: Range<Duration>, duration: Duration) -> Frames<AudioFrame> {
        assert!(range.start <= range.end);
        assert!(!duration.is_zero());
        let Some(timeline) = self.audio.clone() else {return Arc::default()};
        loop {
            self.refresh();
            let (clips, extent, revision, mut cache) = {
                let track = timeline.lock().unwrap();
                let mut cache = track.cache().lock().unwrap();
                if let Some(frames) = cache.find(&range, duration) {return frames}
                (track.clips().clone(), track.duration(), track.revision(), cache.clone())
            };
            if revision != self.revisions.0 {continue}
            let cached = cache.snapshots().rev().find(|(query, _)| {
                query.start <= range.start && query.end >= range.end
            }).map(|(_, frames)| frames.clone());
            let mut frames = Vec::new();
            if let Some(cached) = cached {
                frames.extend(cached.iter().filter(|frame| {
                    !range.is_empty() && frame.timestamp < range.end
                        && frame.timestamp + frame.duration > range.start
                }).cloned());
            } else {
                let remainder = (range.start.as_nanos() % 100_000_000) as u64;
                let mut start = range.start - Duration::from_nanos(remainder);
                while !range.is_empty() && start < range.end.min(extent) {
                    let end = start + Duration::from_millis(100);
                    let decoded = self.audio_window(&clips, &mut cache, start..end);
                    for frame in decoded.iter().filter(|frame| {
                        !range.is_empty() && frame.timestamp < range.end
                            && frame.timestamp + frame.duration > range.start
                    }) {
                        if frames.last().is_none_or(|last: &Arc<AudioFrame>| {
                            last.timestamp != frame.timestamp
                        }) {frames.push(frame.clone())}
                    }
                    start = end;
                }
            }
            let frames = Self::chunk_audio(&frames, range.clone(), duration, &cache);
            let bytes = frames.iter().map(|frame| {
                frame.sound.samples.len() * std::mem::size_of::<Sample>()
            }).sum();
            let frames = Arc::new(frames);
            cache.cache_snapshot(range.clone(), duration, frames.clone(), bytes);
            let track = timeline.lock().unwrap();
            if !track.unchanged(revision, range.clone()) || frames.iter().any(|frame| {
                !track.unchanged(revision, frame.timestamp..frame.timestamp + frame.duration)
            }) {continue}
            track.merge(revision, cache, |frame| frame.timestamp..frame.timestamp + frame.duration);
            return track.cache().lock().unwrap().find(&range, duration).unwrap_or(frames)
        }
    }
    fn video_frames(&mut self, range: Range<Duration>) -> Frames<VideoFrame> {
        assert!(range.start <= range.end);
        let Some(timeline) = self.video.clone() else {return Arc::default()};
        loop {
            self.refresh();
            let (clips, extent, revision, mut cache) = {
                let track = timeline.lock().unwrap();
                let mut cache = track.cache().lock().unwrap();
                if let Some(frames) = cache.find(&range, Duration::ZERO) {return frames}
                (track.clips().clone(), track.duration(), track.revision(), cache.clone())
            };
            if revision != self.revisions.1 {continue}
            let cached = cache.snapshots().rev().find(|(query, _)| {
                query.start <= range.start && query.end >= range.end
            }).map(|(_, frames)| frames.clone());
            let mut frames = Vec::new();
            if let Some(cached) = cached {
                frames.extend(cached.iter().filter(|frame| {
                    !range.is_empty() && frame.timestamp < range.end
                        && frame.timestamp + frame.duration > range.start
                }).cloned());
            } else {
                let remainder = (range.start.as_nanos() % 100_000_000) as u64;
                let mut start = range.start - Duration::from_nanos(remainder);
                while !range.is_empty() && start < range.end.min(extent) {
                    let end = start + Duration::from_millis(100);
                    let decoded = self.video_window(&clips, &mut cache, start..end);
                    for frame in decoded.iter().filter(|frame| {
                        !range.is_empty() && frame.timestamp < range.end
                            && frame.timestamp + frame.duration > range.start
                    }) {
                        if frames.last().is_none_or(|last: &Arc<VideoFrame>| {
                            last.timestamp != frame.timestamp
                        }) {frames.push(frame.clone())}
                    }
                    start = end;
                }
            }
            let bytes = frames.iter().map(|frame| {
                frame.image.len() * std::mem::size_of::<crate::Pixel>()
            }).sum();
            let frames = Arc::new(frames);
            cache.cache_snapshot(range.clone(), Duration::ZERO, frames.clone(), bytes);
            let track = timeline.lock().unwrap();
            if !track.unchanged(revision, range.clone()) || frames.iter().any(|frame| {
                !track.unchanged(revision, frame.timestamp..frame.timestamp + frame.duration)
            }) {continue}
            track.merge(revision, cache, |frame| frame.timestamp..frame.timestamp + frame.duration);
            return track.cache().lock().unwrap().find(&range, Duration::ZERO).unwrap_or(frames)
        }
    }
}

struct Reader {
    kind: ffmpeg::media::Type,
    index: usize,
    decoder: Option<FrameDecoder>,
    target: f64,
    audio_end: Option<f64>,
    audio_previous: Option<(f64, u32, Vec<Sample>)>,
    audio_frames: VecDeque<Arc<AudioFrame>>,
    video_previous: Option<(f64, Arc<VideoFrame>)>,
    pending: Option<(Range<f64>, usize, ffmpeg::Frame)>,
    damaged: Option<(Range<Duration>, usize, Range<f64>)>,
    frames: VecDeque<Arc<VideoFrame>>,
}
impl Reader {
    fn new(kind: ffmpeg::media::Type) -> Self {
        Self {
            kind, index: 0, decoder: None, target: 0.0,
            audio_end: None, pending: None, damaged: None, frames: VecDeque::new(),
            audio_previous: None, audio_frames: VecDeque::new(), video_previous: None,
        }
    }
    fn seek(&mut self, clips: &[Clip], position: Duration) {
        self.index = clips.partition_point(|clip| clip.end <= position);
        self.decoder = None;
        self.target = position.as_secs_f64();
        self.audio_end = None;
        self.audio_previous = None;
        self.audio_frames.clear();
        self.video_previous = None;
        self.pending = None;
        self.damaged = None;
        self.frames.clear();
    }
    fn next<'a>(&mut self, clips: &'a [Clip], frame: &mut ffmpeg::Frame) -> Option<FrameRead<'a>> {
        let mut corrupt = None;
        loop {
            if let Some((range, index, source)) = &mut self.damaged {
                let start = range.start;
                let end = if self.kind == ffmpeg::media::Type::Audio {
                    (start + Duration::from_millis(100)).min(range.end)
                } else {range.end};
                range.start = end;
                let clip = &clips[*index];
                let source = source.clone();
                if range.is_empty() {self.damaged = None;}
                return Some((start.as_secs_f64()..end.as_secs_f64(), clip, Some(source)))
            }
            if let Some((range, index, mut pending)) = self.pending.take() {
                if range.start >= clips[index].end.as_secs_f64() {
                    self.index = index + 1;
                    self.decoder = None;
                    continue
                }
                std::mem::swap(frame, &mut pending);
                return Some((range, &clips[index], None))
            }
            let clip = clips.get(self.index)?;
            let Content::Encoded {source, end: source_end, ..} = &clip.data else {
                self.index += 1;
                self.decoder = None;
                return Some((clip.start.as_secs_f64()..clip.end.as_secs_f64(), clip, None))
            };
            if self.decoder.is_none() {
                self.audio_previous = None;
                self.video_previous = None;
                let mut decoder = match FrameDecoder::new(source.clone(), self.kind) {
                    Ok(decoder) => decoder,
                    Err(_) => {
                        self.damage(self.index, clip, clip.start.as_secs_f64()..clip.end.as_secs_f64());
                        self.index += 1;
                        continue
                    }
                };
                let trimmed = clip.offset > decoder.start;
                let source_start = decoder.start;
                let lookback = (decoder.interval * 2.0).max(0.1);
                let position = (self.target - clip.start.as_secs_f64() - lookback)
                    .max(source_start - clip.offset);
                decoder.start = clip.offset;
                if (position > 0.0 || trimmed) && decoder.seek(position).is_err() {
                    // A broken seek index need not prevent sequential decoding of the same bytes.
                    decoder = FrameDecoder::new(source.clone(), self.kind).unwrap();
                    decoder.start = clip.offset;
                }
                self.decoder = Some(decoder);
                self.audio_end = None;
            }
            let decoder = self.decoder.as_mut().unwrap();
            match decoder.next(frame) {
                Ok(Some(position)) => {
                    let position = position + clip.start.as_secs_f64();
                    let end = decoder.end + clip.start.as_secs_f64();
                    if let Some(start) = corrupt.take() {
                        self.damage(self.index, clip, start..position);
                        let mut pending = unsafe {ffmpeg::Frame::empty()};
                        std::mem::swap(frame, &mut pending);
                        self.pending = Some((position..end, self.index, pending));
                        continue
                    }
                    if position >= clip.end.as_secs_f64() {
                        self.index += 1;
                        self.decoder = None;
                        continue
                    }
                    return Some((position..end, clip, None))
                }
                Ok(None) => {
                    let end = decoder.end + clip.start.as_secs_f64();
                    let tolerance = decoder.base.abs().max(0.000_001);
                    let held = self.frames.back().is_some_and(|frame| {
                        frame.timestamp >= clip.start && frame.timestamp < clip.end
                    });
                    let known_end = decoder.interval > 0.0 || !held;
                    if let Some(start) = corrupt.take().or_else(|| {
                        (clip.declared_end && known_end && end + tolerance < clip.end.as_secs_f64())
                            .then_some(end)
                    }) {
                        let end = source_end - clip.offset + clip.start.as_secs_f64();
                        self.damage(self.index, clip, start..end);
                    }
                    self.index += 1;
                    self.decoder = None;
                }
                Err(DecodeError::Corrupt(position)) => {
                    let position = position + clip.start.as_secs_f64();
                    corrupt = Some(corrupt.map_or(position, |start: f64| start.min(position)));
                }
                Err(DecodeError::Failed(_)) => {
                    let end = decoder.end + clip.start.as_secs_f64();
                    let end = if decoder.interval == 0.0 {
                        self.frames.back().map_or(end, |frame| {
                            (frame.timestamp + frame.duration).as_secs_f64()
                        })
                    } else {end};
                    let start = corrupt.take().unwrap_or(end);
                    let end = source_end - clip.offset + clip.start.as_secs_f64();
                    self.damage(self.index, clip, start..end);
                    self.index += 1;
                    self.decoder = None;
                }
            }
        }
    }
    fn damage(&mut self, index: usize, clip: &Clip, range: Range<f64>) {
        let start = Duration::from_secs_f64(range.start.max(clip.start.as_secs_f64()));
        let end = Duration::from_secs_f64(range.end.min(clip.end.as_secs_f64()).max(0.0));
        if start >= end || end.as_secs_f64() <= self.target {return}
        let mut start = start;
        if self.kind == ffmpeg::media::Type::Audio {
            let target = Duration::from_secs_f64(self.target.max(0.0));
            let skipped = target.saturating_sub(start).as_nanos() / 100_000_000;
            start += Duration::from_millis(u64::try_from(skipped).unwrap() * 100);
        }
        self.damaged = Some((start..end, index, range));
    }
    fn audio(&mut self, clips: &[Clip]) -> Option<Arc<AudioFrame>> {
        assert_eq!(self.kind, ffmpeg::media::Type::Audio);
        let mut raw = ffmpeg::frame::Audio::empty();
        loop {
            if let Some(frame) = self.audio_frames.pop_front() {
                self.target = self.target.max((frame.timestamp + frame.duration).as_secs_f64());
                return Some(frame)
            }
            let (range, clip, corrupt) = self.next(clips, &mut raw)?;
            let mut position = range.start;
            if let Some(damaged) = corrupt {
                let rate = if clip.sound.0 == 0 {48_000} else {clip.sound.0};
                let channels = clip.sound.1.max(1);
                let short = Duration::from_secs_f64((damaged.end - damaged.start).max(0.0))
                    <= Duration::from_millis(2);
                let repaired = if short {
                    self.audio_previous.as_ref().zip(self.pending.as_ref())
                        .and_then(|((left_end, left_rate, left), (right_range, _, frame))| {
                            let tolerance = self.decoder.as_ref().unwrap().base.abs()
                                .max(1.0 / rate as f64);
                            if *left_rate != rate || left.len() != channels as usize
                                || (left_end - damaged.start).abs() > tolerance
                                || (right_range.start - damaged.end).abs() > tolerance
                                || clip.is_corrupted(
                                    right_range.start..right_range.start + 1.0 / rate as f64,
                                ) {return None}
                            let pointer = unsafe {ffmpeg::ffi::av_frame_clone(frame.as_ptr())};
                            assert!(!pointer.is_null());
                            let mut right = unsafe {ffmpeg::frame::Audio::wrap(pointer)};
                            if right.rate() != rate || right.channels() != channels {return None}
                            let converted = Self::convert(&mut right).ok()?;
                            let right: &[Sample] = bytemuck::cast_slice(converted.plane::<f32>(0));
                            if right.len() < channels as usize {return None}
                            let total = ((right_range.start - left_end) * rate as f64).round();
                            if total <= 0.0 {return None}
                            let first = ((range.start - left_end) * rate as f64).round().max(0.0);
                            let last = ((range.end - left_end) * rate as f64).round().min(total);
                            if first >= last {return None}
                            let timestamp = Duration::from_secs_f64(
                                (left_end + first / rate as f64).max(0.0),
                            );
                            let samples = (first as usize..last as usize).flat_map(|sample| {
                                let weight = (sample + 1) as f32 / (total + 1.0) as f32;
                                left.iter().zip(right).map(move |(left, right)| {
                                    Sample(left.0 + (right.0 - left.0) * weight)
                                })
                            }).collect();
                            Some(AudioFrame {
                                timestamp,
                                duration: Duration::from_secs_f64((last - first) / rate as f64),
                                sound: Sound {channels, samples}, corrupted: true,
                            })
                        })
                } else {None};
                if let Some(frame) = repaired {
                    let end = (frame.timestamp + frame.duration).as_secs_f64();
                    self.target = self.target.max(end);
                    self.audio_end = Some(end);
                    self.audio_previous = None;
                    return Some(Arc::new(frame))
                }
                let first = (range.start * rate as f64).round() as u64;
                let last = (range.end * rate as f64).round() as u64;
                let timestamp = Duration::from_secs_f64(first as f64 / rate as f64);
                let end = Duration::from_secs_f64(last as f64 / rate as f64);
                let samples = vec![Sample(0.0); (last - first) as usize * channels as usize];
                self.target = self.target.max(end.as_secs_f64());
                self.audio_end = Some(end.as_secs_f64());
                self.audio_previous = None;
                if first == last {continue}
                return Some(Arc::new(AudioFrame {
                    timestamp, duration: end - timestamp, sound: Sound {channels, samples},
                    corrupted: true,
                }))
            }
            if let Content::Audio(frame) = &clip.data {
                position -= clip.offset;
                let channels = frame.sound.channels as usize;
                let total = frame.sound.samples.len() / channels;
                let rate = total as f64 / frame.duration.as_secs_f64();
                if clip.end.as_secs_f64() <= self.target {continue}
                let target = clip.start.as_secs_f64();
                let skip = ((target - position).max(0.0) * rate).round() as usize;
                let count = ((clip.end.as_secs_f64() - position).max(0.0) * rate).round() as usize;
                let count = count.min(total);
                if skip >= count {continue}
                if skip == 0 && count == total && clip.start == frame.timestamp
                    && clip.end == frame.timestamp + frame.duration {
                    self.target = clip.end.as_secs_f64();
                    return Some(frame.clone())
                }
                let samples = frame.sound.samples[skip * channels..count * channels].to_vec();
                let position = Duration::from_secs_f64((position + skip as f64 / rate).max(0.0));
                let frame = AudioFrame {
                    sound: Sound {channels: frame.sound.channels, samples}, timestamp: position,
                    duration: frame.duration.mul_f64((count - skip) as f64 / total as f64),
                    corrupted: frame.corrupted,
                };
                self.target = self.target.max((position + frame.duration).as_secs_f64());
                return Some(Arc::new(frame))
            }
            assert!(matches!(clip.data, Content::Encoded {..}));
            let (rate, channels) = (raw.rate(), raw.channels());
            if rate == 0 || channels == 0 {
                self.damage(self.index, clip, range);
                continue
            }
            let tolerance = self.decoder.as_ref().unwrap().base.abs().max(1.0 / rate as f64);
            if let Some(end) = self.audio_end.filter(|end| (position - end).abs() <= tolerance) {
                position = end;
            }
            let end = position + raw.samples() as f64 / rate as f64;
            self.audio_end = Some(end);
            let converted = match Self::convert(&mut raw) {
                Ok(converted) => converted,
                Err(_) => {self.damage(self.index, clip, position..end); continue}
            };
            let samples: &[Sample] = bytemuck::cast_slice(converted.plane::<f32>(0));
            self.audio_previous = if samples.len() >= channels as usize
                && !clip.is_corrupted(end - 1.0 / rate as f64..end) {
                Some((end, rate, samples[samples.len() - channels as usize..].to_vec()))
            } else {None};
            let target = self.target.max(clip.start.as_secs_f64());
            if end <= target {continue}
            let skip = ((clip.start.as_secs_f64() - position).max(0.0) * rate as f64)
                .round() as usize;
            let count = (samples.len() / channels as usize).min(
                ((clip.end.as_secs_f64() - position).max(0.0) * rate as f64).round() as usize,
            );
            if skip >= count {continue}
            let start = Duration::from_secs_f64((position + skip as f64 / rate as f64).max(0.0));
            let end = Duration::from_secs_f64((position + count as f64 / rate as f64).max(0.0));
            for (range, corrupted) in clip.regions(start..end) {
                let first = ((range.start.as_secs_f64() - position) * rate as f64).round() as usize;
                let last = ((range.end.as_secs_f64() - position) * rate as f64).round() as usize;
                let first = first.clamp(skip, count);
                let last = last.clamp(first, count);
                if first == last {continue}
                let timestamp = Duration::from_secs_f64(
                    (position + first as f64 / rate as f64).max(0.0),
                );
                let samples = samples[first * channels as usize..last * channels as usize].to_vec();
                self.audio_frames.push_back(Arc::new(AudioFrame {
                    timestamp, duration: Duration::from_secs_f64((last - first) as f64 / rate as f64),
                    sound: Sound {channels, samples}, corrupted,
                }));
            }
        }
    }
    fn convert(raw: &mut ffmpeg::frame::Audio) -> Result<ffmpeg::frame::Audio, ffmpeg::Error> {
        let (rate, channels) = (raw.rate(), raw.channels());
        let layout = if raw.channel_layout().is_empty() {
            ffmpeg::ChannelLayout::default(i32::from(channels))
        } else {raw.channel_layout()};
        raw.set_channel_layout(layout);
        let format = ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Packed);
        let mut converted = ffmpeg::frame::Audio::empty();
        ffmpeg::software::resampling::Context::get(raw.format(), layout, rate, format, layout, rate)?
            .run(raw, &mut converted)?;
        Ok(converted)
    }
    fn video(&mut self, clips: &[Clip]) -> Option<Arc<VideoFrame>> {
        assert_eq!(self.kind, ffmpeg::media::Type::Video);
        while self.frames.len() < 2 {
            let mut raw = ffmpeg::frame::Video::empty();
            let Some((range, clip, corrupt)) = self.next(clips, &mut raw) else {break};
            let position = Duration::from_secs_f64(range.start.max(clip.start.as_secs_f64()));
            if position >= clip.end {continue}
            let frame = if let Some(damaged) = corrupt {
                let end = Duration::from_secs_f64(range.end).min(clip.end);
                let short = Duration::from_secs_f64((damaged.end - damaged.start).max(0.0))
                    <= Duration::from_millis(100);
                let previous = self.video_previous.as_ref().filter(|(timestamp, _)| {
                    short && *timestamp < damaged.start
                });
                let image = previous.map_or_else(
                    || Self::corrupted_image(clip.dimensions), |(_, frame)| frame.image.clone(),
                );
                Arc::new(VideoFrame {
                    image, timestamp: position,
                    duration: end - position,
                    corrupted: true,
                })
            } else {
                match &clip.data {
                    Content::Video(frame) => {
                        if frame.timestamp == position && frame.duration == clip.end - position {
                            frame.clone()
                        } else {
                            Arc::new(VideoFrame {
                                image: frame.image.clone(), timestamp: position,
                                duration: clip.end - position,
                                corrupted: frame.corrupted,
                            })
                        }
                    }
                    Content::Audio(_) => unreachable!(),
                    Content::Encoded {..} => {
                        let decoder = self.decoder.as_mut().unwrap();
                        let mut duration = clip.end - position;
                        if raw.packet().duration > 0 {
                            let end = range.start + raw.packet().duration as f64 * decoder.base;
                            let end = Duration::from_secs_f64(end.max(0.0));
                            duration = duration.min(end.saturating_sub(position));
                        }
                        let image = decoder.image(&raw);
                        let corrupted = image.is_err();
                        let image = image
                            .unwrap_or_else(|_| Self::corrupted_image(clip.dimensions));
                        Arc::new(VideoFrame {image, timestamp: position, duration, corrupted})
                    }
                }
            };
            if !frame.corrupted && !clip.is_corrupted(range.start..range.start + 0.000_000_001) {
                self.video_previous = Some((range.start, frame.clone()));
            }
            if frame.duration.is_zero() {continue}
            self.queue_video(frame);
            while self.frames.front().is_some_and(|frame| {
                (frame.timestamp + frame.duration).as_secs_f64() <= self.target
            }) {self.frames.pop_front();}
        }
        let mut frame = self.frames.pop_front()?;
        let index = clips.partition_point(|clip| clip.start <= frame.timestamp).checked_sub(1).unwrap();
        if !frame.corrupted && !clips[index].corrupted.is_empty() {
            let regions = clips[index].regions(frame.timestamp..frame.timestamp + frame.duration);
            for (range, corrupted) in regions.iter().skip(1).rev() {
                self.frames.push_front(Arc::new(VideoFrame {
                    image: frame.image.clone(), timestamp: range.start,
                    duration: range.end - range.start, corrupted: *corrupted,
                }));
            }
            let (range, corrupted) = &regions[0];
            let first = Arc::make_mut(&mut frame);
            first.timestamp = range.start;
            first.duration = range.end - range.start;
            first.corrupted = *corrupted;
        }
        self.target = self.target.max((frame.timestamp + frame.duration).as_secs_f64());
        Some(frame)
    }

    fn corrupted_image(dimensions: (u32, u32)) -> Image {
        let (width, height) = dimensions;
        let pixels = vec![
            Pixel {red: 0, green: 0, blue: 0, alpha: 255}; width as usize * height as usize
        ];
        Image::new(dimensions, pixels)
    }
    fn queue_video(&mut self, frame: Arc<VideoFrame>) {
        if self.frames.back().is_some_and(|previous| previous.timestamp >= frame.timestamp) {
            self.frames.pop_back();
        }
        if let Some(previous) = self.frames.back_mut() {
            let duration = previous.duration.min(frame.timestamp - previous.timestamp);
            if duration != previous.duration {Arc::make_mut(previous).duration = duration}
        }
        self.frames.push_back(frame);
    }
}
