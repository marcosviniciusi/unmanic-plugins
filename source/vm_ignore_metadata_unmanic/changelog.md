**<span style="color:#56adda">0.0.6</span>** *(marcosviniciusi)*
- Allow re-processing of files that have V2 tag but still contain bitmap subtitles (PGS/VOBSUB)
- Fixes files processed before OCR plugin was installed/fixed — they had the V2 tag but PGS was never converted to SRT
- Second ffprobe call (streams only, subtitle filter) runs only when V2 tag is found

**<span style="color:#56adda">0.0.5</span>** *(marcosviniciusi)*
- Add files without UNMANIC_FULL_PIPELINE_V2 tag to the processing queue (sets True when tag is missing)
- Fixes ~2000 files stuck in limbo — processed by pipeline V1 but never entering V2 queue

**<span style="color:#56adda">0.0.4</span>** *(marcosviniciusi)*
- Changed tag from `UNMANIC_FULL_PIPELINE` to `UNMANIC_FULL_PIPELINE_V2` to match pipeline v2 (with OCR subtitle plugin)
- Files processed by pipeline v1 will re-enter the queue and pass through the new OCR plugin

**<span style="color:#56adda">0.0.3</span>** *(marcosviniciusi)*
- Changed to check `UNMANIC_FULL_PIPELINE=processed` tag (full pipeline completion)
- Changed priority to 0 (runs LAST — lower number = runs later in Unmanic) so it overrides all other plugins' decisions
- If tag exists, file is completely ignored — no other plugin can re-add it
- Renamed to "Ignore - Full Pipeline Completed"

**<span style="color:#56adda">0.0.2</span>** *(marcosviniciusi)*
- Standardized plugin name to "Ignore - Metadata Processed" for consistent naming
- Updated description with metadata tag details

**<span style="color:#56adda">0.0.1</span>**
- Initial version (marcosviniciusi)
