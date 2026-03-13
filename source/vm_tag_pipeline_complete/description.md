
---

### What it does

Writes the **`UNMANIC_FULL_PIPELINE=processed`** metadata tag to the output file. This tag signals that the file has completed the entire processing pipeline.

The plugin remuxes the file (copies all streams without re-encoding) and adds the metadata tag at the format level.

### When to use

Place this plugin as the **very last processing step** in your pipeline. It works together with the **Ignore - Full Pipeline Completed** (`vm_ignore_metadata_unmanic`) plugin, which checks for this tag and completely ignores files that have it.

### How it works

1. File enters this plugin after all other processing steps are done
2. FFmpeg remuxes the file with `-metadata unmanic_full_pipeline=processed`
3. All streams (video, audio, subtitles, data) are copied untouched
4. On next library scan, `vm_ignore_metadata_unmanic` detects the tag and blocks the file from re-entering the queue

### Pipeline position

This must be the **last** processing plugin:

```
 1. vm_video_transcoder              ← Video transcoding
 2. vm_audio_transcoder              ← Audio to EAC3 5.1
 3. vm_audio_transcode_create_stereo ← Stereo downmix
 4. vm_audio_remove_duplicates       ← Remove duplicates
 5. vm_subtitles_transcode           ← Subtitle processing
 6. vm_tag_pipeline_complete         ← Write pipeline tag (THIS — LAST)
```

### Ignore plugin setup

The **Ignore - Full Pipeline Completed** plugin (`vm_ignore_metadata_unmanic`) must be enabled with priority 999 (runs last among file test plugins). It checks for `UNMANIC_FULL_PIPELINE=processed` and completely blocks any file that has this tag from being added to the processing queue.

---
