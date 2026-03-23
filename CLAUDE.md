# CLAUDE.md - Unmanic Plugins Repository (marcosviniciusi)

## Repository Overview

This is a custom Unmanic plugin repository maintained by **marcosviniciusi**.
Repository ID: `repository.vinicima` (defined in `config.json`).

## Plugin Inventory

### Processing Plugins

| Dir / Plugin ID | Name | Version | Original Author | Category |
|---|---|---|---|---|
| `vm_video_transcoder` | Video Transcoder - HW Accelerated with Metadata | 0.3.1 | Josh.5 | Video |
| `vm_audio_transcoder` | Audio Transcoder - EAC3 5.1 (Dolby Digital Plus) | 1.1.0 | Josh.5 | Audio |
| `vm_audio_transcode_create_stereo` | Audio Transcode Create Stereo - Surround Sound Downmix | 0.2.0 | Josh.5 | Audio |
| `vm_audio_remove_duplicates` | Audio Remove Duplicates - Deduplicate Audio Streams | 0.1.0 | marcosviniciusi | Audio |
| `vm_subtitles_transcode` | Subtitles Transcode - Keep PT-BR Only | 3.5.0 | marcosviniciusi | Subtitle |
| `vm_subtitles_pgs_to_srt` | Subtitle Transcode - OCR PGS/ASS to SRT | 0.1.1 | marcosviniciusi | Subtitle |
| `vm_tag_pipeline_complete` | Tag Pipeline Complete - Write Full Pipeline Tag | 0.1.0 | marcosviniciusi | Pipeline |

### Post-Processor Plugins

| Dir / Plugin ID | Name | Version | Original Author | Category |
|---|---|---|---|---|
| `vm_postprocessor_otel_trace` | Post-Processor - OpenTelemetry Task Log | 0.4.0 | marcosviniciusi | Observability |

### Filter / Ignore Plugins

| Dir / Plugin ID | Name | Version | Original Author |
|---|---|---|---|
| `ignore_files_recently_modified` | Ignore files recently modified | 0.0.2 | Josh.5 |
| `vm_ignore_task_history` | Ignore - Task History | 0.0.3 | Josh.5 |
| `vm_ignore_metadata_unmanic` | Ignore - Full Pipeline Completed | 0.0.3 | marcosviniciusi |
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
    ├── audio_remove_duplicates/        # Remove duplicate audio streams
    │   ├── info.json
    │   ├── plugin.py
    │   ├── lib/ffmpeg/
    │   └── ...
    ├── subtitles_transcode/            # Keep PT-BR subtitles only
    │   ├── info.json
    │   ├── plugin.py
    │   ├── lib/ffmpeg/
    │   └── ...
    ├── ignore_files_recently_modified/  # Filter: skip recently modified files
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
5. **Every plugin directory MUST contain a `.gitignore` file** — the CI workflow (`plugin-repo-gen.yml`) checks for this and will fail if missing. Use the standard template from existing plugins (ignores `*.py[cod]`, `.idea/`, `.DS_Store`, `site-packages/**`)
6. **Required files per plugin (checked by CI)**: `.gitignore`, `info.json`, `LICENSE`, `plugin.py`. Must NOT contain: `site-packages/`, `settings.json`

## Unmanic Plugin Priority System

- **Lower priority number = runs FIRST** (min-heap style: priority 1 runs before priority 10)
- **Early exit behavior**: Once any plugin sets `add_file_to_pending_tasks = True` or requests ignore (`False`), subsequent file test plugins are **SKIPPED**
- **Best practice**: Place broader filters (ignore plugins) BEFORE specific ones (processing detection) to leverage early exit
- The UI-configured order in each library takes precedence over the `info.json` priority defaults

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

## Completed: Audio Plugin Anti-Reprocessing (2026-03-13)

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

### Stereo stream title fix

Current `generate_audio_stream_tag` produces generic titles like `EAC3 5.1 [Stereo]` which are confusing.
New title format should reflect the REAL output specs:
- **Title**: `{Language} {CODEC} stereo (Padrão)` — e.g., `Korean AAC stereo (Padrão)`
- **Specs shown by player**: `AAC Stereo / Advanced Audio Codec / 2.0 / 48 kHz / 128 kbps`
- Language comes from source stream's `tags.language`
- Codec comes from the configured encoder setting (aac/ac3)
- "stereo" and "(Padrão)" are fixed text

### FFmpeg command structure (single command does everything)

```
ffmpeg -hide_banner -loglevel info
  -i input.mkv
  -strict -2 -max_muxing_queue_size 4096
  -metadata unmanic_stereo=processed          ← format-level tag (in advanced_options)
  -map 0:v:0 -c:v:0 copy                     ← copy video
  -map 0:a:0 -c:a:0 copy                     ← copy surround original
  -map 0:a:0 -c:a:1 aac -ac 2                ← create stereo downmix
  -metadata:s:a:1 title="Korean AAC stereo (Padrão)"  ← stream-level title
  -metadata:s:a:1 language=kor                ← stream-level language
  -y output.mkv
```

### Files to modify

| File | Changes |
|------|---------|
| `source/vm_audio_transcode_create_stereo/plugin.py` | 3-layer check, tag writing, new title format |
| `source/vm_audio_transcode_create_stereo/info.json` | Bump version to 0.2.0 |
| `source/vm_audio_transcode_create_stereo/changelog.md` | Add 0.2.0 entry |
| `source/vm_audio_transcode_create_stereo/description.md` | Update description if needed |
| `README.md` | Update stereo plugin description |
| `CLAUDE.md` | Document completed work |

### Also consider for `vm_audio_transcoder`

- Same pattern could be applied: write `unmanic_audio=processed` tag and check it first.
- Would fix the `remove_original=False` duplicate EAC3 issue.

## Completed: New Plugin — vm_audio_remove_duplicates (2026-03-13)

### Purpose

Remove duplicate audio streams from media files. A "duplicate" is defined as two or more audio streams
where ALL of the following specs are identical:
- `codec_name` (aac, ac3, eac3, etc.)
- `channels` (2, 6, 8, etc.)
- `language` (por, eng, kor, etc.)
- `title` / tag
- `bit_rate` / bitrate

### Real-world example

File with 3x identical `English AAC stereo` streams (AAC Stereo / 2.0 / 48 kHz / 128 kbps).
Plugin keeps only the first one, removes the other 2.

### Logic

**`on_library_management_file_test`**:
1. Check `UNMANIC_FIX_AUDIO=processed` tag → skip if found
2. Probe all audio streams, group by (codec, channels, language, title, bitrate)
3. If any group has >1 stream → add to processing queue

**`on_worker_process`**:
1. Probe all streams
2. For each audio stream group, keep the first, skip mapping the rest
3. Copy all non-audio streams (video, subtitle, data, attachment) as-is
4. Write `-metadata unmanic_fix_audio=processed` to output
5. Build and execute FFmpeg command

### Plugin structure

- Dir: `source/vm_audio_remove_duplicates/`
- ID: `vm_audio_remove_duplicates`
- Name: `Audio Remove Duplicates - Deduplicate Audio Streams`
- Author: `marcosviniciusi`
- Version: `0.1.0`
- Hooks: `on_library_management_file_test` (priority 100, after audio plugins), `on_worker_process` (priority 6, after stereo)
- Tag: `unmanic_fix_audio=processed`
- Needs own lib/ffmpeg (copy from stereo plugin as base, only needs Probe and Parser)

### FFmpeg command structure

```
ffmpeg -hide_banner -loglevel info
  -i input.mkv
  -metadata unmanic_fix_audio=processed
  -map 0:v:0 -c:v copy                  ← copy all video
  -map 0:a:0 -c:a:0 copy                ← keep first English AAC stereo
  (skip 0:a:1 and 0:a:2 — duplicates)   ← removed
  -map 0:s:0 -c:s copy                  ← copy all subtitles
  -y output.mkv
```

### Pipeline order

Runs AFTER all audio processing plugins (which may create the duplicates):
```
7. vm_audio_transcode_create_stereo  ← may create duplicates
8. vm_audio_remove_duplicates        ← NEW: removes duplicates
9. vm_subtitles_transcode
```

## Fix: Video Transcoder Reprocessing Bug (2026-03-13)

### Root cause

The `vm_video_transcoder` had 3 bugs causing reprocessing:

1. **Metadata tag only written in `standard` mode** — `basic` and `advanced` modes never wrote `unmanic_status=processed`
   - Location: `source/vm_video_transcoder/lib/plugin_stream_mapper.py` line 92 (`if mode == 'standard'`)
   - Fix: Write tag in ALL modes (basic, standard, advanced)

2. **Metadata check only when `force_transcode=True`** — Normal mode never checked the tag
   - Location: `source/vm_video_transcoder/plugin.py` lines 240 and 299
   - Fix: Check tag FIRST in `on_library_management_file_test` and `on_worker_process`, regardless of `force_transcode`

3. **Smart filters override codec check** — `autocrop_black_bars` and `target_resolution` returned True
   before the codec check, causing infinite reprocessing even on already-processed HEVC files
   - Location: `source/vm_video_transcoder/lib/plugin_stream_mapper.py` lines 326-337
   - This is now moot because the metadata check runs first and skips the file entirely

### Fix approach

**RULE: If `UNMANIC_STATUS=processed` tag exists, the file is NEVER processed. No exceptions. All modes, all engines.**

- `on_library_management_file_test`: Check tag FIRST (before probe, before stream analysis). If found → return immediately
- `on_worker_process`: Check tag FIRST. If found → return immediately (no exec_command)
- `set_default_values` (plugin_stream_mapper.py): Write `-metadata unmanic_status=processed` in ALL modes, not just standard

### Files modified

| File | Changes |
|------|---------|
| `source/vm_video_transcoder/plugin.py` | Move metadata check to top, remove `force_transcode` condition |
| `source/vm_video_transcoder/lib/plugin_stream_mapper.py` | Write metadata tag in all modes |
| `source/vm_video_transcoder/info.json` | Bump version |
| `source/vm_video_transcoder/changelog.md` | Add entry |

## New: Full Pipeline Tag System (2026-03-13)

### Concept

A single, authoritative tag `UNMANIC_FULL_PIPELINE=processed` that guarantees a file has passed through
the entire pipeline. This replaces relying on individual plugin tags for the "should I process?" decision.

### Components

1. **`vm_ignore_metadata_unmanic`** (edited) — checks for `UNMANIC_FULL_PIPELINE=processed` tag.
   If found → `add_file_to_pending_tasks = False` → file never enters the queue.
   Runs at priority 2 (earliest).

2. **`vm_tag_pipeline_complete`** (new plugin) — last processing step (`on_worker_process`).
   Remuxes the file with `-metadata unmanic_full_pipeline=processed`.
   Runs with high priority number (after all other processing plugins).

### Pipeline order with new plugin

```
1. ignore_files_recently_modified    ← skip files still being written/downloaded (priority 2)
2. vm_ignore_metadata_unmanic        ← checks UNMANIC_FULL_PIPELINE tag (priority 2)
3. vm_ignore_task_history
4. vm_ignore_video_over_res
5. vm_ignore_video_under_res
6. vm_video_transcoder               ← on_worker_process priority 1
7. vm_audio_transcoder               ← on_worker_process priority 3
8. vm_audio_transcode_create_stereo  ← on_worker_process priority 5
9. vm_audio_remove_duplicates        ← on_worker_process priority 6
10. vm_subtitles_transcode           ← on_worker_process priority 7
11. vm_tag_pipeline_complete         ← writes UNMANIC_FULL_PIPELINE=processed (priority 99)
12. vm_postprocessor_otel_trace      ← post-processor
```

## Future Tasks / Considerations

- [x] Implement anti-reprocessing tag for `vm_audio_transcode_create_stereo` (v0.2.0)
- [ ] Consider same pattern for `vm_audio_transcoder`
- [ ] Update `config.json` repository ID if needed (currently `repository.vinicima`)
- [ ] Add icons for plugins that are missing them (e.g., `subtitles_transcode`)
- [ ] Consider adding `platform` field to plugins that are missing it
- [ ] Review and update `description.md` files for consistency
- [ ] Test all plugins after rename to ensure no internal references break
- [ ] Verify GitHub Actions workflow works with new directory names
