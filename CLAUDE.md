# CLAUDE.md - Unmanic Plugins Repository (marcosviniciusi)

## Repository Overview

This is a custom Unmanic plugin repository maintained by **marcosviniciusi**.
Repository ID: `repository.vinicima` (defined in `config.json`).

## Plugin Inventory

### Processing Plugins

| Dir / Plugin ID | Name | Version | Original Author | Category |
|---|---|---|---|---|
| `vm_video_transcoder` | Video Transcoder - HW Accelerated with Metadata | 0.3.0 | Josh.5 | Video |
| `vm_audio_transcoder` | Audio Transcoder - EAC3 5.1 (Dolby Digital Plus) | 1.1.0 | Josh.5 | Audio |
| `vm_audio_transcode_create_stereo` | Audio Transcode Create Stereo - Surround Sound Downmix | 0.1.1 | Josh.5 | Audio |
| `vm_subtitles_transcode` | Subtitles Transcode - Keep PT-BR Only | 3.4.0 | marcosviniciusi | Subtitle |

### Post-Processor Plugins

| Dir / Plugin ID | Name | Version | Original Author | Category |
|---|---|---|---|---|
| `vm_postprocessor_otel_trace` | Post-Processor - OpenTelemetry Task Log | 0.2.0 | marcosviniciusi | Observability |

### Filter / Ignore Plugins

| Dir / Plugin ID | Name | Version | Original Author |
|---|---|---|---|
| `vm_ignore_task_history` | Ignore - Task History | 0.0.3 | Josh.5 |
| `vm_ignore_metadata_unmanic` | Ignore - Metadata Processed | 0.0.2 | marcosviniciusi |
| `vm_ignore_video_over_res` | Ignore - Video Over Resolution Limit | 0.0.4 | Josh.5 |
| `vm_ignore_video_under_res` | Ignore - Video Under Resolution Limit | 0.0.4 | Josh.5 |

## Naming Convention

Plugin IDs follow this pattern to avoid conflicts with the official Unmanic repository:

- **Video**: `video_transcoder` (kept original, unique due to custom repo)
- **Audio**: `audio_transcoder`, `audio_transcode_create_stereo` (category + action)
- **Subtitles**: `subtitles_transcode` (category + action)
- **Ignore/Filters**: `ignore_` prefix (e.g., `ignore_task_history`, `ignore_metadata_unmanic`, `ignore_video_over_res`)
- **Post-Processors**: `postprocessor_` prefix (e.g., `postprocessor_otel_trace`)

**Rule**: `{category}_{action}` or `ignore_{what_is_filtered}`

## ID Rename History

| Previous ID | Previous Dir | New ID / Dir |
|---|---|---|
| `dts_to_dd` | `audio_to_EAC3` | `vm_audio_transcoder` |
| `create_stereo_audio_clone` | `create_stereo_audio_clone` | `vm_audio_transcode_create_stereo` |
| `equalize_subtitles_ptbr` | `equalize_subtitles_ptbr` | `vm_subtitles_transcode` |
| `vm_video_transcoder` | `vm_video_transcoder` | `vm_video_transcoder` (unchanged) |
| `ignore_completed_tasks` | `ignore_completed_tasks` | `vm_ignore_task_history` |
| `vm_ignore_metadata_unmanic` | `vm_ignore_metadata_unmanic` | `vm_ignore_metadata_unmanic` (unchanged) |
| `ignore_video_file_over_resolution` | `ignore_video_file_over_resolution` | `vm_ignore_video_over_res` |
| `ignore_video_file_under_resolution` | `ignore_video_file_under_resolution` | `vm_ignore_video_under_res` |

## Project Structure

```
unmanic-plugins/
├── config.json                          # Repository metadata (id, name)
├── README.md                            # Main repository README
├── CLAUDE.md                            # This file - project documentation
├── .gitignore
├── .github/                             # GitHub workflows
├── docs/
│   └── CONTRIBUTING.md
└── source/
    ├── video_transcoder/                # Video processing (HW accel)
    │   ├── info.json
    │   ├── plugin.py
    │   ├── lib/encoders/               # libx, qsv, vaapi, nvenc, videotoolbox
    │   ├── lib/ffmpeg/                 # probe, parser, stream_mapper
    │   └── ...
    ├── audio_transcoder/               # Audio to EAC3 5.1
    │   ├── info.json
    │   ├── plugin.py
    │   ├── lib/ffmpeg/
    │   └── ...
    ├── audio_transcode_create_stereo/      # Surround to stereo downmix
    │   ├── info.json
    │   ├── plugin.py
    │   ├── lib/ffmpeg/
    │   └── ...
    ├── subtitles_transcode/            # Keep PT-BR subtitles only
    │   ├── info.json
    │   ├── plugin.py
    │   ├── lib/ffmpeg/
    │   └── ...
    ├── ignore_task_history/             # Filter: skip completed tasks
    ├── ignore_metadata_unmanic/        # Filter: skip processed metadata
    ├── ignore_video_over_res/          # Filter: skip high-res
    ├── ignore_video_under_res/         # Filter: skip low-res
    └── postprocessor_otel_trace/       # Post-processor: OTEL traces to SigNoz
```

## Key Files per Plugin

- `info.json` - Plugin metadata (id, name, version, author, compatibility, priorities, tags)
- `plugin.py` - Main plugin code with hooks (`on_library_management_file_test`, `on_worker_process`)
- `changelog.md` - Version history
- `description.md` - Plugin description shown in Unmanic UI
- `lib/ffmpeg/` - Bundled FFmpeg utilities (probe, parser, stream_mapper)

## Unmanic Plugin Rules

1. **Directory name must match `info.json` `id` field** - this is how Unmanic identifies plugins
2. **Plugin hooks**: `on_library_management_file_test` (file scanning), `on_worker_process` (processing), `on_postprocessor_task_results` (after task completes)
3. **Compatibility**: v1 (legacy) and/or v2 (current)
4. **Settings**: Managed via `PluginSettings` class with `settings_dict` and `form_settings`

## Tasks Completed (2026-03-12)

- [x] Explored and documented all 8 plugins
- [x] Renamed `audio_to_EAC3` (id: `dts_to_dd`) -> `audio_transcoder`
- [x] Renamed `create_stereo_audio_clone` -> `audio_transcode_create_stereo`
- [x] Renamed `equalize_subtitles_ptbr` -> `subtitles_transcode`
- [x] Kept `video_transcoder` with original ID
- [x] Renamed `ignore_completed_tasks` -> `ignore_task_history`
- [x] Renamed `ignore_video_file_over_resolution` -> `ignore_video_over_res`
- [x] Renamed `ignore_video_file_under_resolution` -> `ignore_video_under_res`
- [x] Updated all `info.json` files with new IDs, names, versions, and author attribution
- [x] Updated all `changelog.md` files preserving original history
- [x] Updated main `README.md` with organized plugin table
- [x] Created `CLAUDE.md` documentation
- [x] Created `postprocessor_otel_trace` plugin for OpenTelemetry task tracing (SigNoz/Jaeger/Tempo)

## In Progress: Audio Plugin Anti-Reprocessing (2026-03-13)

### Context

The audio plugins (`vm_audio_transcoder` and `vm_audio_transcode_create_stereo`) lack a robust mechanism to prevent reprocessing.
The `vm_video_transcoder` writes `-metadata unmanic_status=processed` at format level (line 100 of `source/vm_video_transcoder/lib/plugin_stream_mapper.py`),
which is detected by `vm_ignore_metadata_unmanic` plugin. Audio plugins don't do this.

### Current anti-reprocessing mechanisms

- **`vm_audio_transcoder`**: Checks codec type (DTS/FLAC/Opus/Vorbis → EAC3). After conversion, codec changes, so it won't reprocess.
  Risk: if `remove_original=False`, original DTS stream persists and plugin could create duplicate EAC3 streams.
- **`vm_audio_transcode_create_stereo`**: Checks stream title tag `[Stereo]` via `audio_tag_already_exists()`.
  Works but fragile — depends on title string matching.

### Planned improvement for `vm_audio_transcode_create_stereo`

**3-layer check in `on_library_management_file_test`** (ordered by priority — first match skips):

1. **Format-level metadata tag** (fastest exit): Check for custom tag (e.g., `unmanic_stereo=processed`) in file format tags via ffprobe.
   If found → skip file entirely, don't check further conditions.
2. **Channel count + language check**: If no tag, verify if a ≤2ch stream already exists for the same language as each surround stream.
   If yes → skip (already has a stereo version).
3. **Codec + channels check**: If no match above, check if a stream with the configured encoder codec (AAC/AC3) + 2 channels exists alongside the surround stream.
   If yes → skip.

**Writing the tag**: When the plugin DOES process, add `-metadata unmanic_stereo=processed` to FFmpeg advanced_options
so it's written to the output file at format level.

### Technical notes for implementation

- **Where to add the tag in FFmpeg args**: In `on_worker_process`, after confirming `streams_need_processing()`, call
  `mapper.set_ffmpeg_advanced_options(**{'-metadata': 'unmanic_stereo=processed'})` BEFORE `get_ffmpeg_args()`.
  This adds it to `self.advanced_options` which is placed in the command before stream mappings.
- **Tag detection in `on_library_management_file_test`**: Use ffprobe JSON output to check `format.tags` for the tag,
  similar to how `vm_ignore_metadata_unmanic/plugin.py` does it (subprocess call to ffprobe with `-print_format json -show_format`).
- **Previous failed attempt**: The tag wasn't being inserted because of how `__build_args` handles kwargs —
  if `-metadata` key already exists in options, it replaces rather than appending. Since the stereo plugin
  doesn't use `-metadata` in advanced_options currently, this should work. Stream-level metadata uses
  different keys (`-metadata:s:a:0`) so no conflict.
- **Key files to modify**:
  - `source/vm_audio_transcode_create_stereo/plugin.py` — add tag check in `on_library_management_file_test`, add tag writing in `on_worker_process`
  - `source/vm_audio_transcode_create_stereo/lib/ffmpeg/stream_mapper.py` — potentially no changes needed (tag goes in advanced_options, not stream_encoding)

### Also consider for `vm_audio_transcoder`

- Same pattern could be applied: write `unmanic_audio=processed` tag and check it first.
- Would fix the `remove_original=False` duplicate EAC3 issue.

## Future Tasks / Considerations

- [ ] Implement anti-reprocessing tag for `vm_audio_transcode_create_stereo` (see section above)
- [ ] Consider same pattern for `vm_audio_transcoder`
- [ ] Update `config.json` repository ID if needed (currently `repository.vinicima`)
- [ ] Add icons for plugins that are missing them (e.g., `subtitles_transcode`)
- [ ] Consider adding `platform` field to plugins that are missing it
- [ ] Review and update `description.md` files for consistency
- [ ] Test all plugins after rename to ensure no internal references break
- [ ] Verify GitHub Actions workflow works with new directory names
