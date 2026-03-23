**<span style="color:#56adda">0.1.1</span>** *(marcosviniciusi)* - Fix missing lib files
- FIX: Added missing mimetype_overrides.py and tools.py to lib/ffmpeg (plugin failed to load without them)

**<span style="color:#56adda">0.1.0</span>** *(marcosviniciusi)* - Initial release
- NEW: Convert PGS/VOBSUB bitmap subtitles to SRT via OCR (pgsrip + tesseract)
- NEW: Convert ASS/SSA styled text subtitles to SRT via FFmpeg
- NEW: Graceful fallback when dependencies (tesseract/pgsrip) are not available
- NEW: Format-level metadata tag `UNMANIC_SUBS_TO_SRT=processed` to prevent reprocessing
- Runs AFTER `vm_subtitles_transcode` (processes only PT-BR tracks that survived filtering)
