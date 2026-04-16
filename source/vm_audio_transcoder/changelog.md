**<span style="color:#56adda">1.2.1</span>** *(marcosviniciusi)*
- Fix: Skip invalid `-c:t:N` and `-c:d:N` FFmpeg specifiers for attachment and data streams
- Prevents FFmpeg errors when processing files with embedded fonts (e.g., anime MKV with TTF attachments)

**<span style="color:#56adda">1.1.0</span>** *(marcosviniciusi)*
- Renamed plugin from `dts_to_dd` to `audio_transcoder` for unique identification
- Updated plugin ID, name and tags to avoid conflicts with official repository
- Standardized author attribution

**<span style="color:#56adda">1.0.0</span>**
- Rewritten based on Josh.5's original dts_to_dd plugin
- Converts DTS, FLAC, Opus and Vorbis to EAC3 5.1 (Dolby Digital Plus)
- Sources with > 6 channels are downmixed to 5.1
- Sources with <= 6 channels preserve channel count
- Configurable EAC3 bitrate (768k / 1024k / 1536k)
- Option to remove original stream after conversion
- AC3, EAC3, TrueHD and AAC streams are passed through untouched
- Bundled lib/ffmpeg for portability

**<span style="color:#56adda">0.0.3</span>**
- Original Josh.5 release (DTS to AC3 only)
