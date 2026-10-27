use ffmpeg_next as ffmpeg;

use std::collections::VecDeque;
use std::ops::Range;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::media::CursorRead;
use crate::{AudioFrame, Cursor, FrameDecoder, Sample, Sound, VideoFrame};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PlaybackMode {#[default] Default, Live(Duration), Loop}

#[derive(Clone, Debug, PartialEq)]
pub struct Frame {
    pub video: Option<Arc<VideoFrame>>,
    pub position: Duration,
    pub duration: Option<Duration>,
    pub speed: f64,
    pub volume: f64,
    pub playing: bool,
}

/// An owned player. The clock advances independently of how often frame() is called.
pub struct Player {
    cursor: Cursor,
    audio_frames: Arc<Vec<Arc<AudioFrame>>>,
    video_frames: Arc<Vec<Arc<VideoFrame>>>,
    audio_range: Range<Duration>,
    video_range: Range<Duration>,
    revisions: (u64, u64),
    rebased: u64,
    frame: Frame,
    mode: PlaybackMode,
    audio_position: Duration,
    output_position: Duration,
    audio_end: Duration,
    audio_history: Duration,
    audio_format: Option<(u32, u16)>,
    audio_revision: u64,
    audio_buffer: VecDeque<Sample>,
    resampler: Option<ffmpeg::software::resampling::Context>,
}
impl Player {
    pub fn new(mut cursor: Cursor) -> Self {
        FrameDecoder::init().unwrap();
        cursor.refresh();
        let mut playhead = cursor.playhead();
        let position = playhead.time(Instant::now());
        playhead.rendered = Some(position);
        let (revisions, rebased) = (playhead.revisions, playhead.rebased);
        let mode = if playhead.live {PlaybackMode::Live(Duration::ZERO)} else {PlaybackMode::Default};
        let frame = Frame {
            duration: (!playhead.live).then_some(playhead.end), position, video: None,
            speed: playhead.speed, volume: 1.0, playing: playhead.started.is_some(),
        };
        drop(playhead);
        Self {
            cursor, audio_frames: Arc::default(), video_frames: Arc::default(),
            audio_range: Duration::ZERO..Duration::ZERO,
            video_range: Duration::ZERO..Duration::ZERO, revisions, rebased, frame, mode,
            audio_position: position, output_position: Duration::ZERO,
            audio_end: position, audio_history: position,
            audio_format: None, audio_revision: 0, audio_buffer: VecDeque::new(), resampler: None,
        }
    }
    pub fn has_audio(&self) -> bool {self.cursor.has_audio()}
    pub fn has_video(&self) -> bool {self.cursor.has_video()}
    pub fn mode(&self) -> PlaybackMode {self.mode}
    /// Live starts behind the available end; Loop restarts after natural completion.
    pub fn set_mode(&mut self, mode: PlaybackMode) {
        if self.mode == mode {return}
        self.refresh();
        let mut playhead = self.cursor.playhead();
        let now = Instant::now();
        playhead.reanchor(now);
        self.mode = mode;
        playhead.live = matches!(mode, PlaybackMode::Live(_));
        if let PlaybackMode::Live(delay) = mode {
            if playhead.started.is_some() {playhead.position = playhead.end.saturating_sub(delay);}
        } else {
            playhead.position = playhead.position.min(playhead.end);
            if playhead.position >= playhead.end {playhead.started = None;}
        }
        let position = playhead.position;
        playhead.rendered = Some(position);
        self.frame.duration = (!playhead.live).then_some(playhead.end);
        self.frame.playing = playhead.started.is_some();
        drop(playhead);
        self.frame.position = position;
        self.frame.video = None;
        self.video_range = Duration::ZERO..Duration::ZERO;
        self.reset_audio(position);
    }
    /// Changes when processed edits or playback commands invalidate rendered audio.
    /// Read frame() to process pending edits before checking a queued output's revision.
    pub fn audio_revision(&self) -> u64 {self.audio_revision}
    /// Discard prefetched audio after a host drops its output queue, preserving playback mode.
    pub fn synchronize_audio(&mut self) {
        self.refresh();
        let mut playhead = self.cursor.playhead();
        let position = playhead.time(Instant::now());
        playhead.rendered = Some(position);
        drop(playhead);
        self.reset_audio(position);
    }
    pub fn start(&mut self) {
        self.refresh();
        let mut playhead = self.cursor.playhead();
        let now = Instant::now();
        playhead.reanchor(now);
        if playhead.started.is_some() || (!playhead.live && playhead.end.is_zero()) {return}
        if let PlaybackMode::Live(delay) = self.mode {
            playhead.position = playhead.end.saturating_sub(delay);
        } else if playhead.position >= playhead.end {playhead.position = Duration::ZERO;}
        playhead.started = Some(now);
        let position = playhead.position;
        playhead.rendered = Some(position);
        drop(playhead);
        self.reset_audio(position);
        self.frame.position = position;
        self.frame.playing = true;
    }
    pub fn stop(&mut self) {
        self.refresh();
        let mut playhead = self.cursor.playhead();
        playhead.reanchor(Instant::now());
        playhead.started = None;
        let position = playhead.position;
        playhead.rendered = Some(position);
        drop(playhead);
        self.frame.playing = false;
        self.frame.position = position;
        self.reset_audio(position);
    }
    /// Seeking chooses a fixed timeline position and leaves live-following mode.
    pub fn seek(&mut self, position: Duration) {
        self.refresh();
        let mut playhead = self.cursor.playhead();
        if matches!(self.mode, PlaybackMode::Live(_)) {
            self.mode = PlaybackMode::Default;
            playhead.live = false;
        }
        let position = position.min(playhead.end);
        playhead.position = position;
        playhead.rendered = Some(position);
        if playhead.started.is_some() {
            playhead.started = (position < playhead.end).then(Instant::now);
        }
        self.frame.duration = Some(playhead.end);
        self.frame.playing = playhead.started.is_some();
        drop(playhead);
        self.frame.video = None;
        self.frame.position = position;
        self.video_range = Duration::ZERO..Duration::ZERO;
        self.reset_audio(position);
    }
    pub fn set_speed(&mut self, speed: f64) {
        assert!(speed.is_finite() && speed > 0.0);
        self.refresh();
        let mut playhead = self.cursor.playhead();
        playhead.reanchor(Instant::now());
        playhead.speed = speed;
        let position = playhead.position;
        playhead.rendered = Some(position);
        drop(playhead);
        self.frame.speed = speed;
        self.reset_audio(position);
    }
    pub fn set_volume(&mut self, volume: f64) {
        assert!(volume.is_finite() && volume >= 0.0);
        if self.frame.volume == volume {return}
        self.frame.volume = volume;
        self.synchronize_audio();
    }
    pub fn frame(&mut self) -> &Frame {
        loop {
            self.refresh();
            let mut playhead = self.cursor.playhead();
            if playhead.revisions != self.revisions || playhead.rebased != self.rebased {continue}
            let now = Instant::now();
            let mut position = playhead.time(now);
            let looping = self.mode == PlaybackMode::Loop && !playhead.end.is_zero()
                && position >= playhead.end
                && (playhead.started.is_some() || self.frame.playing);
            if looping {
                position = Duration::ZERO;
                playhead.position = position;
                playhead.rendered = Some(position);
                playhead.started = Some(now);
            } else if !playhead.live && position >= playhead.end {
                playhead.position = position;
                playhead.started = None;
            }
            self.frame.position = position;
            self.frame.playing = playhead.started.is_some();
            let available = playhead.ends.1;
            drop(playhead);
            if looping {self.reset_audio(position);}
            let query = if self.frame.duration == Some(position) {
                position.saturating_sub(Duration::from_nanos(1))
            } else {position};
            if !self.video_range.contains(&query) {
                let remainder = (query.as_nanos() % 100_000_000) as u64;
                let start = query - Duration::from_nanos(remainder);
                self.video_range = start..start + Duration::from_millis(100);
                self.video_frames = self.cursor.video_frames(
                    start.min(available)..self.video_range.end.min(available),
                );
            }
            let Some(playhead) = self.cursor.validate(
                self.revisions, Duration::ZERO..Duration::ZERO, self.video_span(),
            ) else {continue};
            if playhead.rebased != self.rebased {continue}
            let duration = (!playhead.live).then_some(playhead.end);
            let current_query = if duration == Some(position) {
                position.saturating_sub(Duration::from_nanos(1))
            } else {position};
            if current_query != query {continue}
            self.frame.duration = duration;
            self.frame.playing = playhead.started.is_some();
            let index = self.video_frames.partition_point(|frame| {
                frame.timestamp <= position
                    && self.frame.duration.is_none_or(|end| frame.timestamp < end)
            });
            self.frame.video = index.checked_sub(1).and_then(|index| {
                let frame = &self.video_frames[index];
                let end = frame.timestamp + frame.duration;
                let finished = self.frame.duration == Some(position) && frame.timestamp < position;
                (position < end || (finished && position == end)).then(|| frame.clone())
            });
            return &self.frame
        }
    }
    /// A host may await this deadline, then read frame(); scheduling does not drive the clock.
    pub fn deadline(&mut self) -> Option<Instant> {
        loop {
            self.frame();
            let mut consulted = self.video_span();
            if self.frame.video.is_none() {consulted.end = consulted.end.max(self.video_range.end);}
            let Some(playhead) = self.cursor.validate(
                self.revisions, Duration::ZERO..Duration::ZERO, consulted,
            ) else {continue};
            if playhead.rebased != self.rebased {continue}
            let started = playhead.started?;
            let index = self.video_frames.partition_point(|frame| {
                frame.timestamp <= self.frame.position
            });
            let next = self.video_frames.get(index).map(|frame| frame.timestamp);
            let end = self.frame.video.as_ref().map(|frame| frame.timestamp + frame.duration);
            let next = [next, end, self.frame.duration, Some(self.video_range.end)]
                .into_iter().flatten().filter(|position| *position > self.frame.position).min()?;
            return Some(started + next.saturating_sub(playhead.position).div_f64(playhead.speed))
        }
    }
    /// Render consecutive output samples. Speed changes pitch; gaps produce silence.
    /// Timestamps count rendered output time, including silence, independently of source seeking.
    pub fn audio(&mut self, duration: Duration, sample_rate: u32, channels: u16) -> Result<AudioFrame, String> {
        assert!(sample_rate > 0 && channels > 0 && !duration.is_zero());
        let count = (duration.as_secs_f64() * sample_rate as f64).round() as usize;
        assert!(count > 0);
        let duration = Duration::from_secs_f64(count as f64 / sample_rate as f64);
        let timestamp = self.output_position;
        let count = count * channels as usize;
        self.frame();
        loop {
            self.refresh();
            let (stopped, available) = {
                let playhead = self.cursor.playhead();
                if playhead.revisions != self.revisions || playhead.rebased != self.rebased {continue}
                (playhead.started.is_none(), playhead.ends.0)
            };
            if stopped {
                let samples = vec![Sample(0.0); count];
                let Some(playhead) = self.cursor.validate(
                    self.revisions, Duration::ZERO..Duration::ZERO, Duration::ZERO..Duration::ZERO,
                ) else {continue};
                if playhead.rebased != self.rebased {continue}
                self.output_position += duration;
                return Ok(AudioFrame {
                    timestamp, duration, sound: Sound {channels, samples}, corrupted: false,
                })
            }
            if self.audio_format != Some((sample_rate, channels)) {
                self.reset_audio(self.audio_position);
                self.audio_format = Some((sample_rate, channels));
            }
            let end = self.audio_position + duration.mul_f64(self.frame.speed);
            let mut consulted = self.audio_position.min(self.audio_history)..self.audio_end.max(end);
            for frame in self.audio_frames.iter().filter(|frame| {
                frame.timestamp < self.audio_end
                    && frame.timestamp + frame.duration > self.audio_history
            }) {
                consulted.start = consulted.start.min(frame.timestamp);
                consulted.end = consulted.end.max(frame.timestamp + frame.duration);
            }
            let covered = self.frame.duration.map_or(end, |duration| end.min(duration));
            let mut corrupted = self.audio_frames.iter().any(|frame| {
                self.audio_position < covered && frame.corrupted && frame.timestamp < covered
                    && frame.timestamp + frame.duration > self.audio_position
            });
            while self.audio_buffer.len() < count {
                if self.frame.duration.is_some_and(|end| self.audio_end >= end) {
                    self.flush_audio()?;
                    self.audio_buffer.resize(count.max(self.audio_buffer.len()), Sample(0.0));
                    break
                }
                if !self.audio_range.contains(&self.audio_end) {
                    let remainder = (self.audio_end.as_nanos() % 100_000_000) as u64;
                    let start = self.audio_end - Duration::from_nanos(remainder);
                    let limit = available.max(end)
                        .max(self.audio_end + Duration::from_millis(1));
                    self.load_audio(start..(start + Duration::from_millis(100)).min(limit));
                }
                let index = self.audio_frames.partition_point(|frame| {
                    frame.timestamp + frame.duration <= self.audio_end
                });
                let frame = self.audio_frames.get(index).filter(|frame| {
                    self.frame.duration.is_none_or(|end| frame.timestamp < end && self.audio_end < end)
                }).cloned();
                let next = frame.as_ref().map_or(self.audio_range.end, |frame| frame.timestamp);
                let next = self.frame.duration.map_or(next, |end| next.min(end));
                if next > self.audio_end {
                    let gap = (next - self.audio_end).as_secs_f64() / self.frame.speed;
                    let gap_count = (gap * sample_rate as f64).round() as usize;
                    if gap_count > 0 {self.flush_audio()?;}
                    let available = count.saturating_sub(self.audio_buffer.len()) / channels as usize;
                    let added = gap_count.min(available);
                    self.audio_buffer.resize(
                        self.audio_buffer.len() + added * channels as usize, Sample(0.0),
                    );
                    if added < gap_count {
                        self.audio_end += Duration::from_secs_f64(added as f64 / sample_rate as f64)
                            .mul_f64(self.frame.speed);
                        break
                    }
                    self.audio_end = next;
                }
                if self.audio_buffer.len() >= count {break}
                let Some(frame) = frame else {continue};
                corrupted |= self.audio_position < covered && frame.corrupted
                    && frame.timestamp < covered && frame.timestamp + frame.duration > self.audio_position;
                let frame_end = frame.timestamp + frame.duration;
                consulted.start = consulted.start.min(frame.timestamp);
                consulted.end = consulted.end.max(frame_end);
                let end = self.frame.duration.map_or(frame_end, |end| frame_end.min(end));
                let input_count = frame.sound.samples.len() / frame.sound.channels as usize;
                let first = (self.audio_end.saturating_sub(frame.timestamp).as_secs_f64()
                    / frame.duration.as_secs_f64() * input_count as f64).round() as usize;
                let last = ((end - frame.timestamp).as_secs_f64()
                    / frame.duration.as_secs_f64() * input_count as f64).round() as usize;
                let first = first.min(input_count);
                let last = last.min(input_count);
                if first >= last {self.audio_end = end; continue}
                let rate = (frame.sample_rate() as f64 * self.frame.speed).round() as u32;
                assert!(rate > 0 && rate <= i32::MAX as u32);
                let layout = ffmpeg::ChannelLayout::default(i32::from(frame.sound.channels));
                let format = ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Packed);
                if self.resampler.as_ref().is_some_and(|resampler| {
                    resampler.input().rate != rate || resampler.input().channel_layout != layout
                }) {self.flush_audio()?;}
                if self.resampler.is_none() {
                    self.resampler = Some(ffmpeg::software::resampling::Context::get(
                        format, layout, rate, format,
                        ffmpeg::ChannelLayout::default(i32::from(channels)), sample_rate,
                    ).map_err(|error| error.to_string())?);
                }
                let mut input = ffmpeg::frame::Audio::new(format, last - first, layout);
                input.set_rate(rate);
                let source_channels = frame.sound.channels as usize;
                let samples = &frame.sound.as_ref()[first * source_channels..last * source_channels];
                input.plane_mut::<f32>(0).copy_from_slice(samples);
                let resampler = self.resampler.as_mut().unwrap();
                let delay = resampler.delay().map_or(0, |delay| delay.output.max(0) as usize);
                let capacity = ((last - first) as f64 * sample_rate as f64 / rate as f64).ceil() as usize;
                let mut output = ffmpeg::frame::Audio::new(
                    format, capacity + delay + 32, ffmpeg::ChannelLayout::default(i32::from(channels)),
                );
                resampler.run(&input, &mut output).map_err(|error| error.to_string())?;
                self.audio_buffer.extend(output.plane::<f32>(0).iter().copied().map(Sample));
                self.audio_history = frame.timestamp;
                self.audio_end = end;
            }
            let samples = self.audio_buffer.drain(..count).map(|sample| {
                Sample((sample.0 * self.frame.volume as f32).clamp(-1.0, 1.0))
            }).collect();
            self.audio_position = end;
            self.audio_end = self.audio_end.max(self.audio_position);
            if let Some(resampler) = &self.resampler {
                let delay = resampler.delay().map_or(0, |delay| delay.input.max(0));
                let history = Duration::from_secs_f64(delay as f64 / resampler.input().rate as f64)
                    .mul_f64(self.frame.speed);
                self.audio_history = self.audio_position.saturating_sub(history);
            } else if self.audio_buffer.is_empty() {self.audio_history = self.audio_position;}
            let start = if self.resampler.is_some() || !self.audio_buffer.is_empty() {
                self.audio_position.min(self.audio_history)
            } else {self.audio_position};
            let last = self.audio_end.max(start + Duration::from_nanos(1)) - Duration::from_nanos(1);
            let start = start - Duration::from_nanos((start.as_nanos() % 100_000_000) as u64);
            let last = last - Duration::from_nanos((last.as_nanos() % 100_000_000) as u64);
            let limit = available.max(self.audio_end);
            self.load_audio(start..(last + Duration::from_millis(100)).min(limit));
            let mut playhead = self.cursor.validate(
                self.revisions, consulted, Duration::ZERO..Duration::ZERO,
            );
            if playhead.as_ref().is_none_or(|playhead| playhead.rebased != self.rebased) {
                drop(playhead);
                let playhead = self.cursor.playhead();
                let position = playhead.rendered.unwrap_or_else(|| playhead.time(Instant::now()));
                drop(playhead);
                self.reset_audio(position);
                continue
            }
            let playhead = playhead.as_mut().unwrap();
            playhead.rendered = Some(end);
            self.frame.duration = (!playhead.live).then_some(playhead.end);
            self.output_position += duration;
            return Ok(AudioFrame {timestamp, duration, corrupted, sound: Sound {channels, samples}})
        }
    }

    fn refresh(&mut self) {
        loop {
            self.cursor.refresh();
            let playhead = self.cursor.playhead();
            self.frame.duration = (!playhead.live).then_some(playhead.end);
            self.frame.speed = playhead.speed;
            let (mut revisions, mut rebased) = (playhead.revisions, playhead.rebased);
            let position = playhead.time(Instant::now());
            drop(playhead);
            if revisions == self.revisions && rebased == self.rebased {return}
            let start = if self.resampler.is_some() || !self.audio_buffer.is_empty() {
                self.audio_position.min(self.audio_history)
            } else {self.audio_position};
            let end = self.audio_end.max(start);
            // Rendered audio ahead of the clock may still be waiting in a host output queue.
            let mut consulted = start.min(position)..end;
            for frame in self.audio_frames.iter().filter(|frame| {
                frame.timestamp < end && frame.timestamp + frame.duration > start
            }) {
                consulted.start = consulted.start.min(frame.timestamp);
                consulted.end = consulted.end.max(frame.timestamp + frame.duration);
            }
            let valid = self.cursor.validate(
                self.revisions, consulted.clone(), Duration::ZERO..Duration::ZERO,
            );
            let reset = valid.as_ref().is_none_or(|head| head.rebased != self.rebased);
            drop(valid);
            if reset {
                let playhead = self.cursor.playhead();
                revisions = playhead.revisions;
                rebased = playhead.rebased;
                self.frame.duration = (!playhead.live).then_some(playhead.end);
                let position = playhead.rendered.unwrap_or_else(|| playhead.time(Instant::now()));
                drop(playhead);
                self.reset_audio(position);
            } else {
                let audio = self.cursor.audio_frames(
                    self.audio_range.clone(), Duration::from_millis(100),
                );
                let Some(playhead) = self.cursor.validate(
                    revisions, consulted, Duration::ZERO..Duration::ZERO,
                ) else {continue};
                if playhead.rebased != rebased {continue}
                self.frame.duration = (!playhead.live).then_some(playhead.end);
                drop(playhead);
                self.audio_frames = audio;
            }
            self.revisions = revisions;
            self.rebased = rebased;
            self.video_range = Duration::ZERO..Duration::ZERO;
            self.video_frames = Arc::default();
            self.frame.video = None;
            return
        }
    }
    fn video_span(&self) -> Range<Duration> {
        let position = self.frame.position;
        let query = if self.frame.duration == Some(position) {
            position.saturating_sub(Duration::from_nanos(1))
        } else {position};
        self.video_frames.iter().find(|frame| {
            frame.timestamp <= query && frame.timestamp + frame.duration > query
        }).map_or(query..query + Duration::from_nanos(1), |frame| {
            frame.timestamp..frame.timestamp + frame.duration
        })
    }
    fn load_audio(&mut self, range: Range<Duration>) {
        if self.audio_range == range {return}
        self.audio_frames = self.cursor.audio_frames(range.clone(), Duration::from_millis(100));
        self.audio_range = range;
    }
    fn reset_audio(&mut self, position: Duration) {
        self.audio_revision += 1;
        self.audio_position = position;
        self.audio_end = position;
        self.audio_history = position;
        self.audio_range = Duration::ZERO..Duration::ZERO;
        self.audio_frames = Arc::default();
        self.audio_buffer.clear();
        self.resampler = None;
    }
    fn flush_audio(&mut self) -> Result<(), String> {
        let Some(mut resampler) = self.resampler.take() else {return Ok(())};
        loop {
            let delay = resampler.delay().map_or(0, |delay| delay.output.max(0) as usize);
            let mut output = ffmpeg::frame::Audio::new(
                resampler.output().format, delay + 32, resampler.output().channel_layout,
            );
            resampler.flush(&mut output).map_err(|error| error.to_string())?;
            if output.samples() == 0 {break}
            self.audio_buffer.extend(output.plane::<f32>(0).iter().copied().map(Sample));
        }
        Ok(())
    }
}
