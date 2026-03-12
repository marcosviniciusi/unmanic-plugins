
This plugin converts **DTS** and **FLAC** audio tracks to **AC3** (Dolby Digital) and/or **EAC3** (Dolby Digital Plus) for maximum device compatibility.

---

## Why convert DTS/FLAC?

**Problem:**
- Modern TVs support AC3/EAC3 but NOT DTS (no passthrough)
- Soundbars often support AC3 but NOT EAC3
- Streaming devices have varying codec support

**Solution:**
- **AC3** = Universal fallback (everything plays it)
- **EAC3** = Better quality for modern devices
- **Dual tracks** = Player automatically selects best supported codec

---

## What gets converted?

### ✅ Processed:
- **All DTS variants:**
  - DTS (Core)
  - DTS-HD Master Audio
  - DTS-HD High Resolution Audio
  - DTS:X (metadata is lost, converts to regular DTS)
  - DTS Express

- **All FLAC:**
  - Any channel configuration (stereo, 5.1, 7.1, etc)

### ❌ Ignored:
- TrueHD / TrueHD Atmos
- AC3 (already compatible)
- EAC3 (already compatible)
- AAC, MP3, Opus, etc.

---

## Output Example

**Before:**
```
Movie.mkv
├── Video: HEVC
├── Audio: DTS 5.1
└── Subtitles
```

**After:**
```
Movie.mkv
├── Video: HEVC
├── Audio 1: AC3 5.1 (640k) - for old devices
├── Audio 2: EAC3 5.1 (1536k) - for modern TVs
└── Subtitles
```

---

## Configuration Options

### Create AC3 track
Enable/disable AC3 conversion. AC3 has maximum compatibility but lower quality ceiling (640 kbps max).

**Bitrate recommendations:**
- **2.0 stereo:** 192-256 kbps
- **5.1 surround:** 384-640 kbps (640k recommended)

### Create EAC3 track
Enable/disable EAC3 conversion. EAC3 offers better quality at higher bitrates (up to 1536 kbps).

**Bitrate recommendations:**
- **2.0 stereo:** 256-384 kbps
- **5.1 surround:** 768-1536 kbps (1536k recommended)

### Remove originals
When enabled, DTS/FLAC tracks are removed after conversion to save space.

**Space savings:**
- DTS-HD MA 5.1: ~3-4 GB → AC3+EAC3: ~1-1.5 GB
- Typical savings: 50-70% per audio track

---

## Channel Handling

**5.1 and below:** Channels preserved
```
DTS 5.1 → AC3 5.1 + EAC3 5.1
FLAC 2.0 → AC3 2.0 + EAC3 2.0
```

**7.1 and above:** Automatic downmix to 5.1
```
DTS 7.1 → AC3 5.1 + EAC3 5.1
```

**Why?** FFmpeg's AC3/EAC3 encoders only support up to 5.1 channels. Most home theater setups are 5.1 anyway.

---

## Quality Notes

### DTS-HD MA (lossless) → AC3/EAC3 (lossy)
Yes, you lose quality. But:
- Most people can't hear the difference on typical setups
- Compatibility is usually more important than audiophile quality
- If you need lossless, keep the original DTS-HD MA

### FLAC (lossless) → AC3/EAC3 (lossy)
- FLAC is typically used for music or high-quality audio
- Converting to lossy reduces quality but improves compatibility
- Consider keeping originals if quality is critical

---

## Use Cases

**Perfect for:**
- Media servers (Plex, Jellyfin, Emby) with mixed device compatibility
- Smart TVs that don't support DTS
- Soundbars with limited codec support
- Reducing storage space while maintaining compatibility

**Not recommended for:**
- Audiophile archival collections
- If all your devices already support DTS/FLAC natively
- Professional audio mastering workflows

---

## Technical Details

- Uses FFmpeg native AC3 and EAC3 encoders
- No external dependencies required
- Preserves video, subtitle, and metadata streams
- Container format unchanged (MKV stays MKV, MP4 stays MP4)

---

## Links

- [FFmpeg AC3 Encoder](https://ffmpeg.org/ffmpeg-codecs.html#ac3)
- [FFmpeg EAC3 Encoder](https://ffmpeg.org/ffmpeg-codecs.html#eac3)
- [Dolby Digital Specification](https://professional.dolby.com/tv/dolby-digital/)
