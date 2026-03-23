**<span style="color:#56adda">0.1.0</span>** *(marcosviniciusi)*
- Initial version
- Detects and removes duplicate audio streams based on all specs (codec, channels, language, title, bitrate)
- Keeps only the first occurrence of each unique audio stream
- Anti-reprocessing via format-level metadata tag `UNMANIC_FIX_AUDIO=processed`
- Copies all non-audio streams (video, subtitles, data) untouched
