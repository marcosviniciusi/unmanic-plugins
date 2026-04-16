**<span style="color:#56adda">0.2.0</span>** *(marcosviniciusi)* - Use pgsrip CLI, fix tag-on-failure bug
- FIX: Use pgsrip CLI subprocess instead of Python import to avoid importlib_metadata conflict with otel plugin. The otel plugin's bundled importlib_metadata backport corrupts the import system, making `import pgsrip` fail with `ModuleNotFoundError` inside Unmanic.
- FIX: Only write `UNMANIC_SUBS_TO_SRT=processed` tag when ALL bitmap subtitles are successfully OCR'd. Previously the tag was written even when OCR failed, permanently blocking retries.
- FIX: `_is_already_processed()` now re-processes files that have the processed tag but still contain bitmap (PGS/VOBSUB) subtitles (from previous failed OCR runs).
- Removed unused Python module imports (json, babelfish) and duplicate PATH setup code.

**<span style="color:#56adda">0.1.5</span>** *(marcosviniciusi)* - Fix pgsrip API call
- FIX: Use `Sup(path)` object instead of raw string path when calling `pgsrip.rip()`. Fixes `'str' object has no attribute 'get_pgs_medias'` error.

**<span style="color:#56adda">0.1.4</span>** *(marcosviniciusi)* - Fix PATH injection at module level
- FIX: Inject /opt/homebrew/bin and /usr/local/bin into os.environ['PATH'] at module load time (not just shutil.which). Ensures mkvextract and tesseract are found when Unmanic daemon starts with minimal PATH.

**<span style="color:#56adda">0.1.3</span>** *(marcosviniciusi)* - Fix PATH for macOS Homebrew
- FIX: Add /opt/homebrew/bin and /usr/local/bin to PATH at module load. Fixes mkvextract/tesseract not found on macOS when Unmanic runs with minimal PATH.

**<span style="color:#56adda">0.1.2</span>** *(marcosviniciusi)* - Fix importlib_metadata conflict with otel plugin
- FIX: pgsrip import fails with KeyError 'home_page' when vm_postprocessor_otel_trace bundles old importlib_metadata in its site-packages. Clean up sys.path and sys.modules before importing pgsrip.

**<span style="color:#56adda">0.1.1</span>** *(marcosviniciusi)* - Fix missing lib files
- FIX: Added missing mimetype_overrides.py and tools.py to lib/ffmpeg (plugin failed to load without them)

**<span style="color:#56adda">0.1.0</span>** *(marcosviniciusi)* - Initial release
- NEW: Convert PGS/VOBSUB bitmap subtitles to SRT via OCR (pgsrip + tesseract)
- NEW: Convert ASS/SSA styled text subtitles to SRT via FFmpeg
- NEW: Graceful fallback when dependencies (tesseract/pgsrip) are not available
- NEW: Format-level metadata tag `UNMANIC_SUBS_TO_SRT=processed` to prevent reprocessing
- Runs AFTER `vm_subtitles_transcode` (processes only PT-BR tracks that survived filtering)
