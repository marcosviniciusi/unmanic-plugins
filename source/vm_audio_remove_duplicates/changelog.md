**<span style="color:#56adda">0.1.2</span>** *(marcosviniciusi)*
- Fix: Skip invalid `-c:t:N` and `-c:d:N` FFmpeg specifiers for attachment and data streams
- Prevents FFmpeg errors when processing files with embedded fonts (e.g., anime MKV with TTF attachments)

**<span style="color:#56adda">0.1.0</span>** *(marcosviniciusi)*
- Initial version
- Detects and removes duplicate audio streams based on all specs (codec, channels, language, title, bitrate)
- Keeps only the first occurrence of each unique audio stream
- Anti-reprocessing via format-level metadata tag `UNMANIC_FIX_AUDIO=processed`
- Copies all non-audio streams (video, subtitles, data) untouched
