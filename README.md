# Unmanic Plugins - marcosviniciusi

Custom plugins for [Unmanic](https://docs.unmanic.app/docs/).

## Included Plugins

### Video Processing

| Plugin ID | Name | Description |
|---|---|---|
| `video_transcoder` | Video Transcoder - HW Accelerated with Metadata | Transcode video streams with HW acceleration (VideoToolbox, NVENC, VAAPI, QSV) |

### Audio Processing

| Plugin ID | Name | Description |
|---|---|---|
| `audio_transcoder` | Audio Transcoder - EAC3 5.1 (Dolby Digital Plus) | Convert DTS, FLAC, Opus and Vorbis audio to EAC3 5.1 |
| `audio_transcode_create_stereo` | Audio Transcode Create Stereo - Surround Sound Downmix | Create stereo clone from surround sound audio streams |

### Subtitle Processing

| Plugin ID | Name | Description |
|---|---|---|
| `subtitles_transcode` | Subtitles Transcode - Keep PT-BR Only | Keep only PT-BR subtitles, remove all others |

### Ignore / Filter Plugins

| Plugin ID | Name | Description |
|---|---|---|
| `ignore_task_history` | Ignore - Task History | Skip files already in completed tasks list |
| `ignore_metadata_unmanic` | Ignore - Metadata Processed | Skip files with UNMANIC_STATUS=processed metadata |
| `ignore_video_over_res` | Ignore - Video Over Resolution Limit | Skip files exceeding configured resolution |
| `ignore_video_under_res` | Ignore - Video Under Resolution Limit | Skip files below configured resolution |

## Installation

### Repo URL:

```
https://raw.githubusercontent.com/marcosviniciusi/unmanic-plugins/repo/repo.json
```

Follow [Unmanic Documentation](https://docs.unmanic.app/docs/plugins/adding_a_custom_plugin_repo/) to add this repo to your Unmanic installation.

## Credits

- **Josh.5** - Original author of several base plugins (video_transcoder, create_stereo_audio_clone, dts_to_dd, ignore plugins)
- **marcosviniciusi** - Fork maintainer, VideoToolbox support, subtitle PT-BR plugin, plugin reorganization

## Links

- [Unmanic Documentation](https://docs.unmanic.app/docs/)

---

## License

This project is licensed under the GPL version 3.

See [CONTRIBUTING.md](docs/CONTRIBUTING.md) to learn how to contribute to Unmanic.
