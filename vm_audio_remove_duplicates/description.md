
---

### What it does

Scans media files for **duplicate audio streams** and removes them, keeping only one copy of each unique audio track.

Two audio streams are considered **duplicates** when ALL of the following specs are identical:
- **Codec** (AAC, AC3, EAC3, etc.)
- **Channels** (2, 6, 8, etc.)
- **Language** (eng, por, kor, etc.)
- **Title** (stream tag/title)
- **Bitrate** (128 kbps, 640 kbps, etc.)

### When to use

This plugin is useful when your pipeline creates duplicate audio tracks — for example, if a file is processed multiple times by audio transcoding or stereo downmix plugins.

#### Example

**Before** (3 duplicate stereo streams):
```
Stream #0:1 — English AAC stereo (AAC / 2.0 / 48 kHz / 128 kbps)
Stream #0:2 — English AAC stereo (AAC / 2.0 / 48 kHz / 128 kbps)  ← duplicate
Stream #0:3 — English AAC stereo (AAC / 2.0 / 48 kHz / 128 kbps)  ← duplicate
Stream #0:4 — English EAC3 5.1 (EAC3 / 5.1 / 48 kHz / 640 kbps)
```

**After** (duplicates removed):
```
Stream #0:1 — English AAC stereo (AAC / 2.0 / 48 kHz / 128 kbps)
Stream #0:2 — English EAC3 5.1 (EAC3 / 5.1 / 48 kHz / 640 kbps)
```

### Anti-reprocessing

After processing, the plugin writes a `UNMANIC_FIX_AUDIO=processed` metadata tag to the output file.
On subsequent scans, files with this tag are skipped immediately.

### Recommended pipeline position

Place this plugin **after** audio transcoding and stereo downmix plugins:

```
6. vm_audio_transcoder              ← Convert to EAC3
7. vm_audio_transcode_create_stereo ← Create stereo clones
8. vm_audio_remove_duplicates       ← Remove duplicates (this plugin)
9. vm_subtitles_transcode           ← Subtitle processing
```

---
