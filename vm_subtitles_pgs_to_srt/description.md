Converts all non-SRT subtitle formats to SubRip (SRT):

- **PGS / VOBSUB (bitmap)**: Extracted and converted to SRT via OCR (pgsrip + tesseract)
- **ASS / SSA (styled text)**: Converted to SRT via FFmpeg (removes styling — fixes small/yellow subtitles in Emby/Jellyfin)
- **SRT (already text)**: Copied as-is, no conversion needed

All subtitles remain embedded in the MKV container.

**Requirements** (must be installed on the host or container):
- `tesseract-ocr` + language packs (e.g., `tesseract-ocr-por`)
- `pgsrip` (Python package)
- `mkvtoolnix` (for mkvextract)

Use the `marcosviniciusi/unmanic:ocr` Docker image which includes all dependencies.
