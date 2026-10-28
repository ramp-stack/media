use ffmpeg_next as ffmpeg;

use std::collections::VecDeque;
use std::io::{Read, Seek, SeekFrom, Write};
use std::ops::Range;
use std::sync::{Arc, Mutex, MutexGuard, Weak};
use std::time::{Duration, Instant};

use crate::decoder::FrameDecoder;
use crate::{Audio, AudioFrame, Format, Frames, Video, VideoFrame};

#[derive(Clone)]
pub enum Encoded {
    File(Input),
    Packets(PacketSource),
}
impl std::fmt::Debug for Encoded {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::File(_) => formatter.write_str("File"),
            Self::Packets(packets) => formatter.debug_struct("Packets")
                .field("count", &packets.length).finish(),
        }
    }
}

pub struct Packets {
    pub parameters: ffmpeg::codec::Parameters,
    pub format: Option<Format>,
    pub time_base: ffmpeg::Rational,
    pub start: f64,
    pub end: f64,
    pub interval: f64,
    pub packets: Vec<ffmpeg::Packet>,
    pub published: usize,
    pub corrupted: Vec<Range<Duration>>,
    pub finished: bool,
}

#[derive(Clone)]
pub struct PacketSource {
    pub source: Arc<Mutex<Packets>>,
    pub length: usize,
    pub end: f64,
    pub interval: f64,
    pub duration: i64,
    pub finished: bool,
    pub corrupted: Arc<Vec<Range<Duration>>>,
}
impl PacketSource {
    pub fn packet(&self, index: usize) -> Option<ffmpeg::Packet> {
        if index >= self.length {return None}
        let mut packet = self.source.lock().unwrap().packets[index].clone();
        if index + 1 == self.length {packet.set_duration(self.duration);}
        Some(packet)
    }
}

#[derive(Clone, Debug)]
pub enum Content {
    Encoded {source: Encoded, end: f64, format: Option<Format>},
    Audio(Arc<AudioFrame>),
    Video(Arc<VideoFrame>),
}

pub trait Join: Sized {
    fn join(self, other: Self, range: Range<Duration>) -> Self;
}

pub trait Export {
    fn write<W: Write + Seek + Send + 'static>(audio: &Audio, video: &Video, writer: W, format: Format) -> Result<(), String>;
}

pub trait CursorRead {
    fn has_audio(&self) -> bool;
    fn has_video(&self) -> bool;
    fn playhead(&self) -> MutexGuard<'_, Playhead>;
    fn validate(&self, revisions: (u64, u64), audio: Range<Duration>, video: Range<Duration>) -> Option<MutexGuard<'_, Playhead>>;
    fn refresh(&mut self);
    fn audio_frames(&mut self, range: Range<Duration>, duration: Duration) -> Frames<AudioFrame>;
    fn video_frames(&mut self, range: Range<Duration>) -> Frames<VideoFrame>;
}

pub trait Track<F> {
    fn timeline(&self) -> Arc<Mutex<Timeline<F>>>;
    fn span(frame: &F) -> Range<Duration>;
    fn clips(&self) -> Arc<Vec<Clip>> {self.timeline().lock().unwrap().clips.clone()}
    fn insert(&mut self, clip: Clip, space: Range<Duration>) {
        self.timeline().lock().unwrap().insert(clip, space, Instant::now(), Self::span);
    }
    fn remove(&mut self, range: Range<Duration>) {
        self.timeline().lock().unwrap().remove(range, Instant::now(), Self::span);
    }
    fn update(&mut self, clip: Clip, changed: Duration) {
        self.timeline().lock().unwrap().update(clip, changed, Self::span);
    }
}

pub struct Playhead {
    pub position: Duration,
    pub rendered: Option<Duration>,
    pub started: Option<Instant>,
    pub speed: f64,
    pub live: bool,
    pub end: Duration,
    pub ends: (Duration, Duration),
    pub revisions: (u64, u64),
    pub rebased: u64,
    pub last_edit: Option<Instant>,
}
impl Playhead {
    pub fn time(&self, now: Instant) -> Duration {
        let elapsed = self.started.map_or(Duration::ZERO, |start| now.duration_since(start));
        let position = self.position + elapsed.mul_f64(self.speed);
        if self.live {position} else {position.min(self.end)}
    }
    pub fn reanchor(&mut self, now: Instant) {
        self.position = self.time(now);
        if self.started.is_some() {
            self.started = (self.live || self.position < self.end).then_some(now);
        }
    }
}

#[derive(Clone, Debug)]
pub struct Clip {
    pub data: Content,
    pub start: Duration,
    pub end: Duration,
    pub offset: f64,
    pub dimensions: (u32, u32),
    pub sound: (u32, u16),
    pub declared_end: bool,
    pub corrupted: Arc<Vec<Range<Duration>>>,
}
impl Clip {
    pub fn from_audio(frame: AudioFrame) -> Self {
        let sound = (frame.sample_rate(), frame.sound.channels);
        assert!(frame.sound.samples.iter().all(|sample| sample.0.is_finite()));
        Self {
            start: frame.timestamp, end: frame.timestamp + frame.duration,
            data: Content::Audio(Arc::new(frame)), offset: 0.0,
            dimensions: (0, 0), sound, declared_end: false, corrupted: Arc::default(),
        }
    }
    pub fn from_video(frame: VideoFrame) -> Self {
        let (width, height) = frame.image.dimensions;
        let duration = frame.duration;
        assert!(width > 0 && height > 0 && !duration.is_zero());
        assert_eq!(frame.image.len(), width as usize * height as usize);
        Self {
            start: frame.timestamp, end: frame.timestamp + duration, dimensions: frame.image.dimensions,
            data: Content::Video(Arc::new(frame)), offset: 0.0,
            sound: (0, 0), declared_end: false, corrupted: Arc::default(),
        }
    }
    pub fn new(source: Encoded, kind: ffmpeg::media::Type, mut decoder: FrameDecoder) -> Result<Option<Self>, String> {
        let offset = decoder.start;
        let duration = decoder.duration(offset).filter(|(duration, _)| !duration.is_zero());
        let (duration, declared_end) = match duration {
            Some(duration) => duration,
            None => {
                (decoder.measure_duration().map_err(|error| error.to_string())?, false)
            }
        };
        if duration.is_zero() {
            if kind == ffmpeg::media::Type::Audio {return Ok(None)}
            return Err("Recording has no usable duration".into())
        }
        Ok(Some(Self {
            data: Content::Encoded {
                source, end: offset + duration.as_secs_f64(), format: decoder.format(),
            },
            start: Duration::ZERO, end: duration, offset,
            dimensions: decoder.dimensions(), sound: decoder.sound(), declared_end,
            corrupted: Arc::new(decoder.corrupted()),
        }))
    }
    pub fn format(clips: &[Self]) -> Option<Format> {
        let mut formats = clips.iter().filter_map(|clip| match &clip.data {
            Content::Encoded {format, ..} => Some(*format),
            _ => None,
        });
        let format = formats.next()??;
        formats.all(|next| next == Some(format)).then_some(format)
    }
    pub fn is_corrupted(&self, range: Range<f64>) -> bool {
        let start = range.start - self.start.as_secs_f64() + self.offset;
        let end = range.end - self.start.as_secs_f64() + self.offset;
        self.corrupted.iter().any(|range| {
            range.start.as_secs_f64() < end && range.end.as_secs_f64() > start
        })
    }
    pub fn regions(&self, range: Range<Duration>) -> Vec<(Range<Duration>, bool)> {
        let mut regions = Vec::new();
        let mut cursor = range.start;
        for damaged in self.corrupted.iter() {
            let start = self.start.as_secs_f64() + damaged.start.as_secs_f64() - self.offset;
            let end = self.start.as_secs_f64() + damaged.end.as_secs_f64() - self.offset;
            if start >= range.end.as_secs_f64() || end <= cursor.as_secs_f64() {continue}
            let start = Duration::from_secs_f64(start.max(0.0)).max(cursor);
            let end = Duration::from_secs_f64(end.min(range.end.as_secs_f64()).max(0.0));
            if start >= end {continue}
            if cursor < start {regions.push((cursor..start, false));}
            regions.push((start..end, true));
            cursor = end;
        }
        if cursor < range.end {regions.push((cursor..range.end, false));}
        regions
    }

    fn shift(mut self, offset: Duration) -> Self {self.start += offset; self.end += offset; self}
    fn trim(&self, range: Range<Duration>, position: Duration) -> Option<Self> {
        let start = self.start.max(range.start);
        let end = self.end.min(range.end);
        if start >= end {return None}
        let offset = match &self.data {
            Content::Video(_) => 0.0,
            _ => self.offset + (start - self.start).as_secs_f64(),
        };
        Some(Self {
            data: self.data.clone(), start: position + (start - range.start),
            end: position + (end - range.start), offset, dimensions: self.dimensions,
            sound: self.sound, declared_end: self.declared_end,
            corrupted: self.corrupted.clone(),
        })
    }
    fn insert(clips: &mut Vec<Self>, clip: Self, space: Range<Duration>) {
        let index = clips.partition_point(|clip| clip.end <= space.start);
        let before = clips.get(index).and_then(|clip| {
            clip.trim(Duration::ZERO..space.start, Duration::ZERO)
        });
        let after = clips.get(index).filter(|clip| clip.start < space.start)
            .and_then(|clip| clip.trim(space.start..clip.end, space.end));
        let split = after.is_some();
        for clip in clips.iter_mut().skip(index + usize::from(split)) {
            clip.start += space.end - space.start;
            clip.end += space.end - space.start;
        }
        clips.splice(
            index..index + usize::from(split), [before, Some(clip), after].into_iter().flatten(),
        );
    }
    fn remove(clips: &mut Vec<Self>, range: Range<Duration>) {
        let start = clips.partition_point(|clip| clip.end <= range.start);
        let end = clips.partition_point(|clip| clip.start < range.end).max(start);
        let before = clips.get(start).and_then(|clip| {
            clip.trim(Duration::ZERO..range.start, Duration::ZERO)
        });
        let after = clips.get(end.saturating_sub(1)).filter(|_| end > start)
            .and_then(|clip| clip.trim(range.end..clip.end, range.end));
        clips.splice(start..end, [before, after].into_iter().flatten());
        for clip in clips.iter_mut().skip(start).filter(|clip| clip.start >= range.end) {
            clip.start -= range.end - range.start;
            clip.end -= range.end - range.start;
        }
    }
}

#[derive(Clone)]
pub struct Input {
    reader: Arc<Mutex<Box<dyn ReadSeek + Send>>>,
    start: u64,
    position: u64,
}
impl Input {
    pub fn new<R: Read + Seek + Send + 'static>(mut reader: R) -> std::io::Result<Self> {
        let start = reader.stream_position()?;
        Ok(Self {reader: Arc::new(Mutex::new(Box::new(reader))), start, position: 0})
    }
}
impl Read for Input {
    fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
        let mut reader = self.reader.lock().unwrap();
        reader.seek(SeekFrom::Start(self.start + self.position))?;
        let count = reader.read(bytes)?;
        self.position += count as u64;
        Ok(count)
    }
}
impl Seek for Input {
    fn seek(&mut self, from: SeekFrom) -> std::io::Result<u64> {
        let position = match from {
            SeekFrom::Start(position) => i128::from(position),
            SeekFrom::Current(offset) => i128::from(self.position) + i128::from(offset),
            SeekFrom::End(offset) => {
                let end = self.reader.lock().unwrap().seek(SeekFrom::End(0))?;
                i128::from(end) - i128::from(self.start) + i128::from(offset)
            }
        };
        self.position = u64::try_from(position).map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "Invalid media seek")
        })?;
        Ok(self.position)
    }
}

pub struct Timeline<F> {
    kind: ffmpeg::media::Type,
    clips: Arc<Vec<Clip>>,
    duration: Duration,
    cache: Arc<Mutex<Cache<F>>>,
    revision: u64,
    edits: Arc<VecDeque<(u64, Range<Duration>)>>,
    playheads: Vec<Weak<Mutex<Playhead>>>,
}
impl<F> Timeline<F> {
    pub fn new(kind: ffmpeg::media::Type, clip: Option<Clip>) -> Self {
        Self {
            kind, duration: clip.as_ref().map_or(Duration::ZERO, |clip| clip.end),
            clips: Arc::new(clip.into_iter().collect()), cache: Arc::new(Mutex::new(Cache::new())),
            revision: 0, edits: Arc::default(), playheads: Vec::new(),
        }
    }
    pub fn clips(&self) -> &Arc<Vec<Clip>> {&self.clips}
    pub fn duration(&self) -> Duration {self.duration}
    pub fn cache(&self) -> &Mutex<Cache<F>> {&self.cache}
    pub fn revision(&self) -> u64 {self.revision}
    pub fn unchanged(&self, revision: u64, range: Range<Duration>) -> bool {
        if range.is_empty() || revision == self.revision {return true}
        let Some((first, _)) = self.edits.front() else {return false};
        revision < self.revision && revision + 1 >= *first && self.edits.iter().all(|(edit, changed)| {
            *edit <= revision || changed.end <= range.start || changed.start >= range.end
        })
    }
    pub fn merge(&self, revision: u64, mut cache: Cache<F>, span: impl Fn(&F) -> Range<Duration>) {
        cache.decoded.retain(|(range, frames, _)| {
            self.unchanged(revision, range.clone())
                && frames.iter().all(|frame| self.unchanged(revision, span(frame)))
        });
        cache.snapshots.retain(|(range, _, frames, _)| {
            self.unchanged(revision, range.clone())
                && frames.iter().all(|frame| self.unchanged(revision, span(frame)))
        });
        self.cache.lock().unwrap().merge(cache, span);
    }
    pub fn attach(&mut self, playhead: &Arc<Mutex<Playhead>>) {
        self.playheads.retain(|head| head.strong_count() > 0);
        self.playheads.push(Arc::downgrade(playhead));
    }
    pub fn join(&mut self, clips: Arc<Vec<Clip>>, range: Range<Duration>, span: impl Fn(&F) -> Range<Duration>) {
        Arc::make_mut(&mut self.clips).extend(
            Arc::unwrap_or_clone(clips).into_iter().map(|clip| clip.shift(range.start)),
        );
        self.duration = range.end;
        self.invalidate(range, span);
        self.notify(Instant::now(), |position| position);
    }
    pub fn insert(&mut self, clip: Clip, space: Range<Duration>, instant: Instant, span: impl Fn(&F) -> Range<Duration>) {
        assert!(space.start < space.end && clip.start >= space.start && clip.end <= space.end);
        let length = space.end - space.start;
        let shifts = space.start < self.duration;
        self.duration = if shifts {self.duration + length} else {self.duration.max(clip.end)};
        Clip::insert(Arc::make_mut(&mut self.clips), clip, space.clone());
        self.invalidate(space.start..Duration::MAX, span);
        self.notify(instant, |position| {
            if shifts && position >= space.start {position + length} else {position}
        });
    }
    pub fn remove(&mut self, range: Range<Duration>, instant: Instant, span: impl Fn(&F) -> Range<Duration>) {
        assert!(range.start <= range.end);
        if range.is_empty() {return}
        let end = range.end.min(self.duration);
        if range.start >= end {return}
        Clip::remove(Arc::make_mut(&mut self.clips), range.start..end);
        self.duration -= end - range.start;
        self.invalidate(range.start..Duration::MAX, span);
        self.notify(instant, |position| {
            if position >= range.end {position - (range.end - range.start)}
            else if position > range.start {range.start} else {position}
        });
    }
    fn update(&mut self, clip: Clip, changed: Duration, span: impl Fn(&F) -> Range<Duration>) {
        let start = changed.min(self.duration).min(clip.end);
        self.duration = self.duration.max(clip.end);
        let clips = Arc::make_mut(&mut self.clips);
        if let Some(last) = clips.last_mut() {*last = clip;} else {clips.push(clip);}
        self.invalidate(start..self.duration, span);
        self.notify(Instant::now(), |position| position);
    }
    fn invalidate(&mut self, range: Range<Duration>, span: impl Fn(&F) -> Range<Duration>) {
        let edits = Arc::make_mut(&mut self.edits);
        edits.push_back((self.revision + 1, range.clone()));
        while edits.len() > 256 {edits.pop_front();}
        if Arc::strong_count(&self.cache) > 1 {
            let cache = self.cache.lock().unwrap().clone();
            self.cache = Arc::new(Mutex::new(cache));
        }
        self.cache.lock().unwrap().invalidate(range, span);
    }
    fn notify(&mut self, instant: Instant, map: impl Fn(Duration) -> Duration) {
        self.revision += 1;
        self.playheads.retain(|weak| {
            let Some(head) = weak.upgrade() else {return false};
            let mut head = head.lock().unwrap();
            // A Media edit visits both tracks with one instant and moves each cursor only once.
            if head.last_edit != Some(instant) {
                head.reanchor(Instant::now());
                let position = map(head.position);
                let rendered = head.rendered.map(&map);
                if position != head.position || rendered != head.rendered {head.rebased += 1;}
                head.position = position;
                head.rendered = rendered;
                head.last_edit = Some(instant);
            }
            if self.kind == ffmpeg::media::Type::Audio {
                head.ends.0 = self.duration;
                head.revisions.0 = self.revision;
            } else {
                head.ends.1 = self.duration;
                head.revisions.1 = self.revision;
            }
            head.end = head.ends.0.max(head.ends.1);
            true
        });
    }
}
impl<F> Clone for Timeline<F> {
    fn clone(&self) -> Self {
        Self {
            kind: self.kind, clips: self.clips.clone(), duration: self.duration,
            cache: self.cache.clone(), revision: self.revision,
            edits: self.edits.clone(), playheads: Vec::new(),
        }
    }
}

pub struct Cache<F> {
    decoded: VecDeque<(Range<Duration>, Frames<F>, usize)>,
    snapshots: VecDeque<(Range<Duration>, Duration, Frames<F>, usize)>,
}
impl<F> Cache<F> {
    pub fn snapshots(&self) -> impl DoubleEndedIterator<Item = (&Range<Duration>, &Frames<F>)> {
        self.snapshots.iter().map(|(range, _, frames, _)| (range, frames))
    }
    pub fn find(&mut self, range: &Range<Duration>, chunk: Duration) -> Option<Frames<F>> {
        let index = self.snapshots.iter().position(|(query, duration, _, _)| {
            query == range && *duration == chunk
        })?;
        let entry = self.snapshots.remove(index).unwrap();
        let frames = entry.2.clone();
        self.snapshots.push_back(entry);
        Some(frames)
    }
    pub fn block(&mut self, range: &Range<Duration>) -> Option<Frames<F>> {
        let index = self.decoded.iter().position(|(query, _, _)| query == range)?;
        let entry = self.decoded.remove(index).unwrap();
        let frames = entry.1.clone();
        self.decoded.push_back(entry);
        Some(frames)
    }
    pub fn cover(&self, range: &Range<Duration>, span: impl Fn(&F) -> Range<Duration>) -> Option<Frames<F>> {
        let mut frames: Vec<_> = self.decoded.iter().map(|entry| &entry.1)
            .chain(self.snapshots.iter().map(|entry| &entry.2)).flat_map(|frames| frames.iter())
            .filter_map(|frame| {
                let extent = span(frame);
                (extent.start < range.end && extent.end > range.start)
                    .then(|| (extent, frame.clone()))
            }).collect();
        frames.sort_by(|(left, _), (right, _)| {
            left.start.cmp(&right.start).then(right.end.cmp(&left.end))
        });
        let mut covered = range.start;
        let mut output = Vec::new();
        for (extent, frame) in frames {
            if extent.start > covered {return None}
            if extent.end <= covered {continue}
            covered = extent.end;
            output.push(frame);
            if covered >= range.end {break}
        }
        (covered >= range.end).then(|| Arc::new(output))
    }
    pub fn share(&self, frame: Arc<F>, span: impl Fn(&F) -> Range<Duration>) -> Arc<F> {
        let extent = span(&frame);
        self.decoded.iter().map(|entry| &entry.1)
            .chain(self.snapshots.iter().map(|entry| &entry.2)).flat_map(|frames| frames.iter())
            .find(|frame| span(frame) == extent).cloned().unwrap_or(frame)
    }
    pub fn cache_block(&mut self, range: Range<Duration>, frames: Frames<F>, bytes: usize) {
        self.decoded.retain(|(query, _, _)| *query != range);
        if bytes > 128 * 1024 * 1024 {return}
        self.decoded.push_back((range, frames, bytes));
        self.trim();
    }
    pub fn cache_snapshot(&mut self, range: Range<Duration>, chunk: Duration, frames: Frames<F>, bytes: usize) {
        self.snapshots.retain(|(query, duration, _, _)| *query != range || *duration != chunk);
        if bytes > 128 * 1024 * 1024 {return}
        self.snapshots.push_back((range, chunk, frames, bytes));
        self.trim();
    }
    pub fn merge(&mut self, other: Self, span: impl Fn(&F) -> Range<Duration>) {
        for (range, frames, bytes) in other.decoded {
            if self.decoded.iter().any(|(query, _, _)| *query == range) {continue}
            let frames = self.share_frames(frames, &span);
            self.cache_block(range, frames, bytes);
        }
        for (range, chunk, frames, bytes) in other.snapshots {
            if self.snapshots.iter().any(|(query, duration, _, _)| {
                *query == range && *duration == chunk
            }) {continue}
            let frames = self.share_frames(frames, &span);
            self.cache_snapshot(range, chunk, frames, bytes);
        }
    }

    fn new() -> Self {Self {decoded: VecDeque::new(), snapshots: VecDeque::new()}}
    fn share_frames(&self, frames: Frames<F>, span: impl Fn(&F) -> Range<Duration>) -> Frames<F> {
        let shared: Vec<_> = frames.iter().map(|frame| self.share(frame.clone(), &span)).collect();
        if shared.iter().zip(frames.iter()).all(|(left, right)| Arc::ptr_eq(left, right)) {frames}
        else {Arc::new(shared)}
    }
    fn invalidate(&mut self, range: Range<Duration>, span: impl Fn(&F) -> Range<Duration>) {
        assert!(range.start <= range.end);
        if range.is_empty() {return}
        let overlaps = |query: &Range<Duration>| {
            !query.is_empty() && query.start < range.end && range.start < query.end
        };
        self.decoded.retain(|(query, frames, _)| {
            !overlaps(query) && !frames.iter().any(|frame| overlaps(&span(frame)))
        });
        self.snapshots.retain(|(query, _, frames, _)| {
            !overlaps(query) && !frames.iter().any(|frame| overlaps(&span(frame)))
        });
    }

    fn trim(&mut self) {
        while self.decoded.len() > 64 {self.decoded.pop_front();}
        while self.snapshots.len() > 64 {self.snapshots.pop_front();}
        let mut bytes = self.decoded.iter().map(|entry| entry.2).sum::<usize>()
            + self.snapshots.iter().map(|entry| entry.3).sum::<usize>();
        while bytes > 128 * 1024 * 1024 {
            if let Some((_, _, _, removed)) = self.snapshots.pop_front() {bytes -= removed;}
            else {bytes -= self.decoded.pop_front().unwrap().2;}
        }
    }
}
impl<F> Clone for Cache<F> {
    fn clone(&self) -> Self {
        Self {
            decoded: self.decoded.clone(), snapshots: self.snapshots.clone(),
        }
    }
}

trait ReadSeek: Read + Seek {}
impl<T: Read + Seek> ReadSeek for T {}
