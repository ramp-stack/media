# Media

`media2` provides owned audio, video, images, editable timelines, and playback primitives for Ramp.

```toml
[dependencies]
media = { package = "media2", version = "0.1.0" }
```

`Media`, `Audio`, and `Video` read seekable sources and decode frames lazily. Their timelines support
inserting and removing content. `Cursor` reads timestamped frames, and `Player` controls playback,
seeking, looping, live playback, speed, and volume. Platform adapters consume the resulting frames;
this crate does not open windows or audio devices.

`Image` reads and writes PNG, JPEG, WebP, GIF, BMP, ICO, TIFF, and QOI. `Encoder` exports supported
container and codec combinations, reusing compatible encoded packets when possible. `Decoder`
receives continuous encoded streams; streaming MP4, MOV, and M4A inputs must be fragmented.

Use `Media::write`, `Audio::write`, or `Video::write` to finalize a seekable file. Use `encode` with
`Decoder` for streaming: concatenating streaming WAV chunks does not finalize its file-length header.

Audio and video processing uses the `ramp-ffmpeg` fork with its portable build enabled. Building
requires Rust with edition 2024 support and the native build tools required by that dependency.

```sh
cargo check --locked
cargo test --locked
```

Backend changes require review by @CalebCouch. Frontend applications should consume the published
crate rather than modifying its implementation.

Licensed under BSD-3-Clause.
