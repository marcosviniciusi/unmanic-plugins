**<span style="color:#56adda">0.2.0</span>** *(marcosviniciusi)*
- Changed tag from `UNMANIC_FULL_PIPELINE` to `UNMANIC_FULL_PIPELINE_V2` to force reprocessing of files from pipeline v1 (before OCR subtitle plugin)

**<span style="color:#56adda">0.1.0</span>** *(marcosviniciusi)*
- Initial version
- Remuxes file to add `UNMANIC_FULL_PIPELINE=processed` format-level metadata tag
- Copies all streams untouched (video, audio, subtitles, data)
- Designed to run as the LAST processing step in the pipeline
- Works with `vm_ignore_metadata_unmanic` to prevent reprocessing
