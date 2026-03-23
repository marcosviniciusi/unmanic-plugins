#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Subtitles to SRT - Convert All Formats to SubRip

Converts non-SRT subtitles to SRT:
- PGS/VOBSUB (bitmap) → SRT via OCR (pgsrip + tesseract)
- ASS/SSA (styled text) → SRT via FFmpeg
- SRT → copied as-is

Runs AFTER vm_subtitles_transcode so only PT-BR tracks remain.
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile

from unmanic.libs.unplugins.settings import PluginSettings

from vm_subtitles_pgs_to_srt.lib.ffmpeg import Probe, Parser

logger = logging.getLogger("Unmanic.Plugin.vm_subtitles_pgs_to_srt")

# Ensure common tool paths are in PATH (macOS Homebrew, Linux)
_EXTRA_PATHS = ['/opt/homebrew/bin', '/usr/local/bin', '/usr/bin']
_current_path = os.environ.get('PATH', '')
for _p in _EXTRA_PATHS:
    if _p not in _current_path:
        os.environ['PATH'] = _p + ':' + os.environ.get('PATH', '')

PROCESSED_TAG = 'unmanic_subs_to_srt'

# Subtitle codecs that need conversion
BITMAP_CODECS = frozenset(['hdmv_pgs_subtitle', 'dvd_subtitle', 'dvdsub', 'xsub'])
STYLED_TEXT_CODECS = frozenset(['ass', 'ssa'])
SRT_CODECS = frozenset(['subrip', 'srt'])


class Settings(PluginSettings):
    settings = {}


def _ensure_path():
    """Ensure common binary paths are in PATH (macOS homebrew, Linux)."""
    import os
    extra_paths = ['/opt/homebrew/bin', '/usr/local/bin', '/snap/bin']
    current = os.environ.get('PATH', '')
    for p in extra_paths:
        if p not in current and os.path.isdir(p):
            current = p + ':' + current
    os.environ['PATH'] = current


# Ensure PATH is set at module load time
_ensure_path()


def _has_pgsrip():
    """Check if pgsrip and tesseract are available."""
    try:
        # Fix: other Unmanic plugins (e.g., otel) may bundle an old
        # importlib_metadata backport in their site-packages that overrides
        # the stdlib importlib.metadata and breaks pgsrip's metadata lookup
        # (KeyError: 'home_page'). Clean up before importing pgsrip.
        import sys
        # Remove otel plugin's site-packages from sys.path
        clean_path = [p for p in sys.path if '/vm_postprocessor_otel_trace/' not in p]
        original_path = sys.path[:]
        sys.path = clean_path
        # Remove stale importlib_metadata from sys.modules if loaded from wrong path
        stale_modules = [k for k in sys.modules
                         if 'importlib_metadata' in k
                         and hasattr(sys.modules[k], '__file__')
                         and sys.modules[k].__file__
                         and 'vm_postprocessor_otel_trace' in str(sys.modules[k].__file__)]
        for mod in stale_modules:
            del sys.modules[mod]
        try:
            import pgsrip  # noqa: F401
        finally:
            sys.path = original_path
        if shutil.which('tesseract') is None:
            return False
        return True
    except (ImportError, KeyError):
        return False


def _has_mkvextract():
    """Check if mkvextract is available."""
    return shutil.which('mkvextract') is not None


def _is_already_processed(probe):
    """Check if file has already been processed by this plugin."""
    format_tags = probe.get('format', {}).get('tags', {})
    for key, value in format_tags.items():
        if key.lower() == PROCESSED_TAG:
            if str(value).lower() in ('true', '1', 'yes', 'processed'):
                return True
    return False


def _get_subtitle_streams(probe):
    """Get all subtitle streams from probe data."""
    streams = probe.get('streams', [])
    subs = []
    sub_idx = 0
    for s in streams:
        if s.get('codec_type', '').lower() == 'subtitle':
            s['_sub_index'] = sub_idx
            subs.append(s)
            sub_idx += 1
    return subs


def _needs_conversion(subtitle_streams):
    """Check if any subtitle stream needs conversion (non-SRT)."""
    for s in subtitle_streams:
        codec = s.get('codec_name', '').lower()
        if codec not in SRT_CODECS:
            return True
    return False


def _extract_pgs_to_sup(input_file, sub_index, output_sup):
    """Extract a PGS subtitle track to a .sup file using mkvextract."""
    # mkvextract uses absolute stream index, not per-type index
    # We need to find the absolute index from the probe data
    abs_index = sub_index
    cmd = ['mkvextract', 'tracks', input_file, '{}:{}'.format(abs_index, output_sup)]
    logger.info("[OCR] Extracting PGS track %d: %s", abs_index, ' '.join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        logger.error("[OCR] mkvextract failed: %s", result.stderr)
        return False
    return os.path.exists(output_sup) and os.path.getsize(output_sup) > 0


def _ocr_sup_to_srt(sup_file, srt_file, language='por'):
    """Run OCR on a .sup file to produce a .srt file using pgsrip."""
    try:
        from pgsrip import pgsrip, Options, Sup
        from babelfish import Language

        lang = Language(language)
        options = Options(languages={lang}, overwrite=True)
        sup_media = Sup(sup_file)
        pgsrip.rip(sup_media, options)

        # pgsrip writes .srt alongside the .sup file
        expected_srt = os.path.splitext(sup_file)[0] + '.srt'
        if os.path.exists(expected_srt) and os.path.getsize(expected_srt) > 0:
            if expected_srt != srt_file:
                shutil.move(expected_srt, srt_file)
            logger.info("[OCR] Successfully converted: %s", srt_file)
            return True
        else:
            logger.warning("[OCR] pgsrip produced no output for: %s", sup_file)
            return False
    except Exception as e:
        logger.error("[OCR] pgsrip failed: %s", str(e))
        return False


def _map_language_to_tesseract(lang_code):
    """Map ISO 639-2/B language codes to tesseract language codes."""
    mapping = {
        'por': 'por', 'pt': 'por', 'pob': 'por',
        'eng': 'eng', 'en': 'eng',
        'spa': 'spa', 'es': 'spa',
        'fre': 'fra', 'fra': 'fra', 'fr': 'fra',
        'ger': 'deu', 'deu': 'deu', 'de': 'deu',
        'ita': 'ita', 'it': 'ita',
        'jpn': 'jpn', 'ja': 'jpn',
        'kor': 'kor', 'ko': 'kor',
        'chi': 'chi_sim', 'zho': 'chi_sim',
        'rus': 'rus', 'ru': 'rus',
        'ara': 'ara', 'ar': 'ara',
        'dut': 'nld', 'nld': 'nld', 'nl': 'nld',
        'dan': 'dan', 'da': 'dan',
        'fin': 'fin', 'fi': 'fin',
        'nor': 'nor', 'no': 'nor',
        'swe': 'swe', 'sv': 'swe',
        'pol': 'pol', 'pl': 'pol',
        'tur': 'tur', 'tr': 'tur',
        'tha': 'tha', 'th': 'tha',
    }
    return mapping.get(lang_code, 'eng')


# --- Library file test ---

def on_library_management_file_test(data):
    abspath = data.get('path')
    logger.info("[TEST] Checking: %s", abspath)

    probe = Probe(logger, allowed_mimetypes=['video'])
    if not probe.file(abspath):
        logger.info("[TEST] Not a video file, skipping: %s", abspath)
        return data

    file_probe = probe.get_probe()

    if _is_already_processed(file_probe):
        logger.info("[TEST] Already processed (tag found), skipping: %s", abspath)
        return data

    subtitle_streams = _get_subtitle_streams(file_probe)
    if not subtitle_streams:
        logger.info("[TEST] No subtitle streams, skipping: %s", abspath)
        return data

    if _needs_conversion(subtitle_streams):
        # Check if we have the tools to actually do the conversion
        has_bitmap = any(s.get('codec_name', '').lower() in BITMAP_CODECS for s in subtitle_streams)
        has_styled = any(s.get('codec_name', '').lower() in STYLED_TEXT_CODECS for s in subtitle_streams)

        if has_bitmap and not (_has_pgsrip() and _has_mkvextract()):
            logger.warning("[TEST] Bitmap subtitles found but pgsrip/mkvextract not available, skipping: %s", abspath)
            if not has_styled:
                return data
            # Still process if there are styled text subs (FFmpeg can handle those)

        data['add_file_to_pending_tasks'] = True
        logger.info("[TEST] Added to pending tasks: %s", abspath)
    else:
        logger.info("[TEST] All subtitles already SRT, skipping: %s", abspath)

    return data


# --- Worker process ---

def on_worker_process(data):
    data['exec_command'] = []
    data['repeat'] = False

    abspath = data.get('file_in')

    probe = Probe(logger, allowed_mimetypes=['video'])
    if not probe.file(abspath):
        return data

    file_probe = probe.get_probe()

    if _is_already_processed(file_probe):
        logger.info("[WORKER] Already processed, skipping: %s", abspath)
        return data

    subtitle_streams = _get_subtitle_streams(file_probe)
    if not subtitle_streams:
        logger.info("[WORKER] No subtitle streams: %s", abspath)
        return data

    if not _needs_conversion(subtitle_streams):
        logger.info("[WORKER] All subtitles already SRT: %s", abspath)
        return data

    all_streams = file_probe.get('streams', [])
    has_pgsrip = _has_pgsrip()
    has_mkvextract = _has_mkvextract()

    # Temp directory for OCR work
    tmp_dir = tempfile.mkdtemp(prefix='unmanic_ocr_')
    temp_srt_files = []  # list of (srt_path, lang, title) for bitmap tracks
    bitmap_failed_indices = set()  # sub indices where OCR failed — keep original

    try:
        # Phase 1: OCR bitmap subtitles
        for s in subtitle_streams:
            codec = s.get('codec_name', '').lower()
            if codec not in BITMAP_CODECS:
                continue

            sub_idx = s['_sub_index']
            abs_idx = s.get('index')
            lang = s.get('tags', {}).get('language', 'und')
            title = s.get('tags', {}).get('title', '')

            if not (has_pgsrip and has_mkvextract):
                logger.warning("[WORKER] No OCR tools for bitmap sub s:%d, keeping as-is", sub_idx)
                bitmap_failed_indices.add(sub_idx)
                continue

            sup_file = os.path.join(tmp_dir, 'track_{}.sup'.format(sub_idx))
            srt_file = os.path.join(tmp_dir, 'track_{}.srt'.format(sub_idx))

            # Extract PGS/VOBSUB to .sup using absolute stream index
            if not _extract_pgs_to_sup(abspath, abs_idx, sup_file):
                logger.warning("[WORKER] Failed to extract sub s:%d, keeping as-is", sub_idx)
                bitmap_failed_indices.add(sub_idx)
                continue

            # OCR the .sup to .srt
            tess_lang = _map_language_to_tesseract(lang)
            if not _ocr_sup_to_srt(sup_file, srt_file, tess_lang):
                logger.warning("[WORKER] OCR failed for sub s:%d, keeping as-is", sub_idx)
                bitmap_failed_indices.add(sub_idx)
                continue

            temp_srt_files.append({
                'srt_path': srt_file,
                'sub_index': sub_idx,
                'lang': lang,
                'title': title,
            })

        # Phase 2: Build FFmpeg command
        ffmpeg_args = [
            '-hide_banner',
            '-loglevel', 'info',
        ]

        # Input 0: original file
        ffmpeg_args += ['-i', abspath]

        # Additional inputs: OCR'd SRT files
        for i, srt_info in enumerate(temp_srt_files):
            ffmpeg_args += ['-i', srt_info['srt_path']]

        # Advanced options
        ffmpeg_args += ['-strict', '-2', '-max_muxing_queue_size', '4096']

        # Metadata tag
        ffmpeg_args += ['-metadata', '{}=processed'.format(PROCESSED_TAG)]

        # Map all non-subtitle streams (video, audio, etc) as copy
        for s in all_streams:
            ct = s.get('codec_type', '').lower()
            idx = s.get('index')
            if ct == 'video':
                ffmpeg_args += ['-map', '0:{}'.format(idx)]
            elif ct == 'audio':
                ffmpeg_args += ['-map', '0:{}'.format(idx)]

        # Copy all video and audio codecs
        ffmpeg_args += ['-c:v', 'copy', '-c:a', 'copy']

        # Map subtitle streams with conversion
        output_sub_idx = 0
        ocr_input_idx = 1  # first OCR'd SRT is input 1

        for s in subtitle_streams:
            codec = s.get('codec_name', '').lower()
            sub_idx = s['_sub_index']
            abs_idx = s.get('index')
            lang = s.get('tags', {}).get('language', '')
            title = s.get('tags', {}).get('title', '')

            if codec in SRT_CODECS:
                # Already SRT — copy as-is
                ffmpeg_args += ['-map', '0:{}'.format(abs_idx)]
                ffmpeg_args += ['-c:s:{}'.format(output_sub_idx), 'copy']
                if lang:
                    ffmpeg_args += ['-metadata:s:s:{}'.format(output_sub_idx), 'language={}'.format(lang)]
                if title:
                    ffmpeg_args += ['-metadata:s:s:{}'.format(output_sub_idx), 'title={}'.format(title)]
                output_sub_idx += 1

            elif codec in STYLED_TEXT_CODECS:
                # ASS/SSA — convert to SRT via FFmpeg
                ffmpeg_args += ['-map', '0:{}'.format(abs_idx)]
                ffmpeg_args += ['-c:s:{}'.format(output_sub_idx), 'srt']
                if lang:
                    ffmpeg_args += ['-metadata:s:s:{}'.format(output_sub_idx), 'language={}'.format(lang)]
                if title:
                    # Strip ASS styling info from title if present
                    clean_title = title.replace('(ASS)', '').replace('(SSA)', '').strip()
                    if clean_title:
                        ffmpeg_args += ['-metadata:s:s:{}'.format(output_sub_idx), 'title={}'.format(clean_title)]
                output_sub_idx += 1

            elif codec in BITMAP_CODECS:
                if sub_idx in bitmap_failed_indices:
                    # OCR failed — copy bitmap as-is (fallback)
                    ffmpeg_args += ['-map', '0:{}'.format(abs_idx)]
                    ffmpeg_args += ['-c:s:{}'.format(output_sub_idx), 'copy']
                    if lang:
                        ffmpeg_args += ['-metadata:s:s:{}'.format(output_sub_idx), 'language={}'.format(lang)]
                    if title:
                        ffmpeg_args += ['-metadata:s:s:{}'.format(output_sub_idx), 'title={}'.format(title)]
                    output_sub_idx += 1
                else:
                    # OCR succeeded — map from temp SRT file
                    ffmpeg_args += ['-map', '{}:0'.format(ocr_input_idx)]
                    ffmpeg_args += ['-c:s:{}'.format(output_sub_idx), 'srt']
                    if lang:
                        ffmpeg_args += ['-metadata:s:s:{}'.format(output_sub_idx), 'language={}'.format(lang)]
                    if title:
                        ffmpeg_args += ['-metadata:s:s:{}'.format(output_sub_idx), 'title={}'.format(title)]
                    ocr_input_idx += 1
                    output_sub_idx += 1

            else:
                # Unknown codec — copy as-is
                ffmpeg_args += ['-map', '0:{}'.format(abs_idx)]
                ffmpeg_args += ['-c:s:{}'.format(output_sub_idx), 'copy']
                output_sub_idx += 1

        # Output file
        ffmpeg_args += ['-y', data.get('file_out')]

        data['exec_command'] = ['ffmpeg'] + ffmpeg_args

        logger.info("[WORKER] FFmpeg command: %s", ' '.join(data['exec_command']))

        parser = Parser(logger)
        parser.set_probe(probe)
        data['command_progress_parser'] = parser.parse_progress

    except Exception as e:
        logger.error("[WORKER] Error building command: %s", str(e))
        # Don't clean up temp dir yet — Unmanic hasn't run the command
        data['exec_command'] = []

    # NOTE: temp_dir cleanup happens after Unmanic runs the command.
    # Since we can't hook into post-exec, the temp files will be cleaned
    # up by the OS or on next run. The temp dir is small (SRT files only).

    return data
