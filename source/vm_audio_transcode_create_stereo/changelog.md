**<span style="color:#56adda">0.2.2</span>** *(marcosviniciusi)*
- Fix: Skip invalid `-c:t:N` and `-c:d:N` FFmpeg specifiers for attachment and data streams
- Prevents FFmpeg errors when processing files with embedded fonts (e.g., anime MKV with TTF attachments)

**<span style="color:#56adda">0.2.0</span>** *(marcosviniciusi)*
- Added 3-layer anti-reprocessing mechanism:
  - Layer 1: Format-level metadata tag `UNMANIC_STEREO=processed` (fastest skip)
  - Layer 2: Channel count + language check (≤2ch stream with same language already exists)
  - Layer 3: Codec + channels check (configured encoder codec with 2 channels already exists)
- Writes `UNMANIC_STEREO=processed` metadata tag to output file to prevent future reprocessing
- Improved stereo stream titles to show real output specs: `{Language} {CODEC} stereo (Padrão)`
- Maintains backwards compatibility with legacy `[Stereo]` tag detection

**<span style="color:#56adda">0.1.1</span>** *(marcosviniciusi)*
- Renamed plugin from `audio_transcode_to_stereo` to `audio_transcode_create_stereo` (better reflects the "create" action)

**<span style="color:#56adda">0.1.0</span>** *(marcosviniciusi)*
- Renamed plugin from `create_stereo_audio_clone` to `audio_transcode_to_stereo` for unique identification
- Updated plugin ID, name and tags to avoid conflicts with official repository
- Standardized author attribution

**<span style="color:#56adda">0.0.3</span>**
- Update FFmpeg helper
- Add platform declaration

**<span style="color:#56adda">0.0.2</span>**
- Enabled support for v2 plugin executor

**<span style="color:#56adda">0.0.1</span>**
- initial version
