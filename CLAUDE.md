# CLAUDE.md - Unmanic Plugins Repository (marcosviniciusi)

## Repository Overview

This is a custom Unmanic plugin repository maintained by **marcosviniciusi**.
Repository ID: `repository.vinicima` (defined in `config.json`).

## Plugin Inventory

### Processing Plugins

| Dir / Plugin ID | Name | Version | Original Author | Category |
|---|---|---|---|---|
| `video_transcoder` | Video Transcoder - HW Accelerated with Metadata | 0.3.0 | Josh.5 | Video |
| `audio_transcoder` | Audio Transcoder - EAC3 5.1 (Dolby Digital Plus) | 1.1.0 | Josh.5 | Audio |
| `audio_transcode_create_stereo` | Audio Transcode Create Stereo - Surround Sound Downmix | 0.1.1 | Josh.5 | Audio |
| `subtitles_transcode` | Subtitles Transcode - Keep PT-BR Only | 3.4.0 | marcosviniciusi | Subtitle |

### Filter / Ignore Plugins

| Dir / Plugin ID | Name | Version | Original Author |
|---|---|---|---|
| `ignore_task_history` | Ignore - Task History | 0.0.3 | Josh.5 |
| `ignore_metadata_unmanic` | Ignore - Metadata Processed | 0.0.2 | marcosviniciusi |
| `ignore_video_over_res` | Ignore - Video Over Resolution Limit | 0.0.4 | Josh.5 |
| `ignore_video_under_res` | Ignore - Video Under Resolution Limit | 0.0.4 | Josh.5 |

## Naming Convention

Plugin IDs follow this pattern to avoid conflicts with the official Unmanic repository:

- **Video**: `video_transcoder` (kept original, unique due to custom repo)
- **Audio**: `audio_transcoder`, `audio_transcode_create_stereo` (category + action)
- **Subtitles**: `subtitles_transcode` (category + action)
- **Ignore/Filters**: `ignore_` prefix (e.g., `ignore_task_history`, `ignore_metadata_unmanic`, `ignore_video_over_res`)

**Rule**: `{category}_{action}` or `ignore_{what_is_filtered}`

## ID Rename History

| Previous ID | Previous Dir | New ID / Dir |
|---|---|---|
| `dts_to_dd` | `audio_to_EAC3` | `audio_transcoder` |
| `create_stereo_audio_clone` | `create_stereo_audio_clone` | `audio_transcode_create_stereo` |
| `equalize_subtitles_ptbr` | `equalize_subtitles_ptbr` | `subtitles_transcode` |
| `video_transcoder` | `video_transcoder` | `video_transcoder` (unchanged) |
| `ignore_completed_tasks` | `ignore_completed_tasks` | `ignore_task_history` |
| `ignore_metadata_unmanic` | `ignore_metadata_unmanic` | `ignore_metadata_unmanic` (unchanged) |
| `ignore_video_file_over_resolution` | `ignore_video_file_over_resolution` | `ignore_video_over_res` |
| `ignore_video_file_under_resolution` | `ignore_video_file_under_resolution` | `ignore_video_under_res` |

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
    └── ignore_video_under_res/         # Filter: skip low-res
```

## Key Files per Plugin

- `info.json` - Plugin metadata (id, name, version, author, compatibility, priorities, tags)
- `plugin.py` - Main plugin code with hooks (`on_library_management_file_test`, `on_worker_process`)
- `changelog.md` - Version history
- `description.md` - Plugin description shown in Unmanic UI
- `lib/ffmpeg/` - Bundled FFmpeg utilities (probe, parser, stream_mapper)

## Unmanic Plugin Rules

1. **Directory name must match `info.json` `id` field** - this is how Unmanic identifies plugins
2. **Plugin hooks**: `on_library_management_file_test` (file scanning), `on_worker_process` (processing)
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

## Future Tasks / Considerations

- [ ] Update `config.json` repository ID if needed (currently `repository.vinicima`)
- [ ] Add icons for plugins that are missing them (e.g., `subtitles_transcode`)
- [ ] Consider adding `platform` field to plugins that are missing it
- [ ] Review and update `description.md` files for consistency
- [ ] Test all plugins after rename to ensure no internal references break
- [ ] Verify GitHub Actions workflow works with new directory names
