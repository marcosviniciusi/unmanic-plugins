# Unmanic Plugins - marcosviniciusi

Custom plugin repository for [Unmanic](https://docs.unmanic.app/docs/) — an automated media library optimizer.

## Why This Repository Exists

Modern media libraries often contain files with a wide variety of codecs, audio formats, and subtitle languages. This creates two main problems:

### TV Compatibility (LG C/G Series and Others)

LG OLED TVs from the **C series** (C1, C2, C3, C4) and **G series** (G1, G2, G3, G4) — among the most popular high-end displays — have **limited native codec support**. They do not natively decode many common video and audio formats found in media libraries:

- **Video**: DTS-HD Master Audio, TrueHD, and certain AVC/HEVC profiles may not play or pass through correctly
- **Audio**: DTS and its variants (DTS-HD MA, DTS:X) are **not supported** on LG webOS. Only Dolby formats (DD, DD+, Atmos via DD+) and PCM/AAC are reliably supported
- **Subtitles**: Embedded PGS/ASS subtitles in unsupported languages add bloat and cause playback issues on some clients

### The Solution: Standardized Media

These plugins work together as an Unmanic pipeline to **equalize your entire media library** into a consistent, highly compatible format:

| Component | Target Format | Why |
|---|---|---|
| **Video** | HEVC (H.265) | Best compression-to-quality ratio, universally supported on modern TVs and devices |
| **Audio** | EAC3 5.1 (Dolby Digital Plus) | Native support on LG, Samsung, Sony TVs. Replaces DTS/FLAC/Opus which many TVs cannot decode |
| **Subtitles** | PT-BR only (SRT) | Keeps only Brazilian Portuguese subtitles, removes all other languages to reduce file size |

The result: every file in your library plays natively on LG C/G series TVs, Apple TV, Plex, Jellyfin, and virtually any modern media player — **without real-time transcoding on the server**.

## Plugins

### Video Processing

#### `video_transcoder` — Video Transcoder (HW Accelerated with Metadata)

Transcodes video streams to HEVC with full hardware acceleration support:

| Encoder | Platform | Notes |
|---|---|---|
| **VideoToolbox** | Apple Silicon (M1/M2/M3/M4) and Intel Macs (2017+) | Prioritizes speed; files may be 20-30% larger than other HW encoders |
| **NVENC** | NVIDIA GPUs | Excellent quality/speed balance |
| **QSV** | Intel CPUs with integrated graphics | Low power consumption, good for always-on servers |
| **VAAPI** | AMD/Intel GPUs on Linux | Broad Linux support |
| **LibSVT-AV1** | Software (CPU) | AV1 encoding for next-gen compatibility |
| **Libx264/Libx265** | Software (CPU) | Fallback for systems without HW acceleration |

Preserves metadata, chapter markers, and stream disposition during transcoding.

### Audio Processing

#### `audio_transcoder` — Audio Transcoder (EAC3 5.1 / Dolby Digital Plus)

Converts incompatible audio codecs to **EAC3 5.1 (Dolby Digital Plus)**:

- **Converts**: DTS, DTS-HD MA, DTS:X, FLAC, Opus, Vorbis → EAC3 5.1
- **Bitrate management**: Automatically selects bitrate based on source (448k for streams ≤768kb/s, 640k for higher)
- **Downmix**: Sources with more than 6 channels (e.g., 7.1 DTS-HD MA) are downmixed to 5.1
- **Preserves**: AAC, AC3, and existing EAC3 streams are kept untouched

#### `audio_transcode_create_stereo` — Surround Sound Downmix to Stereo

Creates a **stereo clone** of surround sound audio streams for devices that don't support multichannel audio (phones, laptops, Bluetooth headphones). The original surround track is preserved alongside the new stereo track.

### Subtitle Processing

#### `subtitles_transcode` — Keep PT-BR Subtitles Only

Designed for Brazilian Portuguese media libraries:

- **Keeps** only PT-BR subtitles embedded in the container
- **Converts** ASS/SSA subtitle formats to SRT automatically
- **Removes** all other subtitle languages from the container
- Reduces file size by removing unnecessary subtitle tracks

### Ignore / Filter Plugins

These plugins control which files enter the processing pipeline:

| Plugin | Purpose |
|---|---|
| `ignore_task_history` | Skip files already processed by Unmanic |
| `ignore_metadata_unmanic` | Skip files with `UNMANIC_STATUS=processed` metadata tag |
| `ignore_video_over_res` | Skip files exceeding a configured resolution limit |
| `ignore_video_under_res` | Skip files below a configured resolution limit |

### Observability

#### `postprocessor_otel_trace` — OpenTelemetry Task Log

Sends structured log records for every completed task to an **OpenTelemetry-compatible backend** (SigNoz, Jaeger, Grafana Tempo) via OTLP HTTP. Includes:

- Task result (success/failed), duration, timestamps
- Source and destination file paths, sizes
- Environment metadata (service name, hostname)
- Filterable attributes for dashboards and alerting

## Recommended Pipeline Order

For a fully equalized media library, configure the plugins in this order in Unmanic:

```
1. ignore_task_history          ← Skip already-processed files
2. ignore_metadata_unmanic      ← Skip files tagged as processed
3. ignore_video_over_res        ← Optional: skip 4K if targeting 1080p
4. ignore_video_under_res       ← Optional: skip low-res files
5. video_transcoder             ← Transcode video to HEVC
6. audio_transcoder             ← Convert audio to EAC3 5.1
7. audio_transcode_create_stereo ← Add stereo downmix track
8. subtitles_transcode          ← Keep PT-BR subtitles only
9. postprocessor_otel_trace     ← Log results to OTEL backend
```

## Installation

### Repository URL

```
https://raw.githubusercontent.com/marcosviniciusi/unmanic-plugins/repo/repo.json
```

Follow the [Unmanic documentation](https://docs.unmanic.app/docs/plugins/adding_a_custom_plugin_repo/) to add this custom repository to your Unmanic instance.

## Credits

- **Josh.5** — Original author of the base plugins (video_transcoder, audio_transcoder, audio_transcode_create_stereo, ignore plugins)
- **marcosviniciusi** — Fork maintainer: VideoToolbox encoder support, subtitle PT-BR plugin, OTEL observability plugin, plugin reorganization and standardization

## Links

- [Unmanic Documentation](https://docs.unmanic.app/docs/)
- [LG TV Supported Codecs](https://www.lg.com/us/support/help-library/lg-tv-supported-file-formats-for-usb-media-playback-CT10000018-1422182258498)

---

## License

This project is licensed under the GPL version 3.

See [CONTRIBUTING.md](docs/CONTRIBUTING.md) to learn how to contribute.
