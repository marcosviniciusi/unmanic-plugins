
**<span style="color:#56adda">1.0.4</span>**
- Improved audio track titles with proper channel detection
- Mono shows as "AC3 Mono (256k)" instead of "AC3 1.0"
- Stereo shows as "AC3 Stereo (256k)" instead of "AC3 2.0"
- 5.1 shows as "AC3 5.1 (640k)" (6 channels detected)
- 7.1 shows as "AC3 7.1 (640k)" (8 channels detected)
- Preserves original language metadata (Korean, English, etc)

**<span style="color:#56adda">1.0.3</span>**
- Added descriptive titles to audio tracks for easy identification
- AC3 tracks now show: "AC3 5.1 (640k)" instead of copying original title
- EAC3 tracks now show: "EAC3 5.1 (1536k)" instead of copying original title

**<span style="color:#56adda">1.0.2</span>**
- CRITICAL FIX: Added missing lib/ffmpeg helper module
- Plugin will now load correctly and show configuration options
- Fixed import path to use plugin's own ffmpeg module

**<span style="color:#56adda">1.0.1</span>**
- Fixed: Added missing input_type="checkbox" for all boolean settings
- Now checkboxes will display correctly in UI

**<span style="color:#56adda">1.0.0</span>**
- Initial release
- Convert DTS (all variants) to AC3 and/or EAC3
- Convert FLAC (all channel configs) to AC3 and/or EAC3
- Configurable bitrates for both AC3 and EAC3
- Option to create dual tracks (AC3 + EAC3)
- Option to remove original DTS/FLAC after conversion
- Automatic downmix of 7.1+ to 5.1
- Ignores TrueHD, Atmos, and other codecs
