#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Equalize Subtitles PT-BR

- Keeps only PT-BR subtitles embedded in the MKV (copied as-is, no conversion)
- Removes all non-PT-BR subtitle tracks from the container
- Tags files as processed to avoid reprocessing
"""

import logging
import os

from unmanic.libs.unplugins.settings import PluginSettings

from vm_subtitles_transcode.lib.ffmpeg import StreamMapper, Probe, Parser

logger = logging.getLogger("Unmanic.Plugin.vm_subtitles_transcode")


class Settings(PluginSettings):
    settings = {}


# --- Language helpers ---

PT_CODES = frozenset([
    'por', 'pt', 'pt-br', 'pt-pt', 'pob', 'pb', 'ptbr',
    'portuguese', 'portugues', 'brazilian', 'brasil', 'brazil',
])



def is_portuguese(lang):
    return lang.lower() in PT_CODES if lang else False


def classify_subtitle(lang):
    """
    Returns:
      'embed'  - Keep as-is (PT-BR or untagged)
      'remove' - Remove from container
    Applies to ALL subtitle streams regardless of codec.
    """
    if is_portuguese(lang) or not lang:
        return 'embed'
    return 'remove'


# --- StreamMapper subclass (same pattern as official plugins) ---

class PluginStreamMapper(StreamMapper):
    def __init__(self):
        super(PluginStreamMapper, self).__init__(logger, ['subtitle'])

    def test_stream_needs_processing(self, stream_info: dict):
        if stream_info.get('codec_type', '').lower() != 'subtitle':
            return False
        lang = stream_info.get('tags', {}).get('language', '')
        return classify_subtitle(lang) == 'remove'

    def custom_stream_mapping(self, stream_info: dict, stream_id: int):
        codec = stream_info.get('codec_name', '').lower()
        lang = stream_info.get('tags', {}).get('language', '')

        if classify_subtitle(lang) == 'remove':
            logger.info("[WORKER] Removing subtitle: lang=%s codec=%s (input s:%d)", lang, codec, stream_id)
            return {
                'stream_mapping': [],
                'stream_encoding': [],
            }

        # Should not be reached (PT-BR goes through framework default copy path),
        # but kept as a safe fallback.
        logger.info("[WORKER] Keeping subtitle: lang=%s codec=%s (input s:%d, output s:%d)",
                     lang, codec, stream_id, self.subtitle_output_count)
        return {
            'stream_mapping': ['-map', '0:s:{}'.format(stream_id)],
            'stream_encoding': ['-c:s:{}'.format(self.subtitle_output_count), 'copy'],
        }


# --- Library file test ---

def on_library_management_file_test(data):
    abspath = data.get('path')
    logger.info("[TEST] Checking: %s", abspath)

    probe = Probe(logger, allowed_mimetypes=['video'])
    if not probe.file(abspath):
        logger.info("[TEST] Not a video file, skipping: %s", abspath)
        return data

    file_probe = probe.get_probe()
    format_tags = file_probe.get('format', {}).get('tags', {})
    for tag_key, tag_value in format_tags.items():
        if tag_key.lower() in ('unmanic-equalize-subtitles-v2', 'unmanic_equalize_subtitles_v2'):
            if str(tag_value).lower() in ('true', '1', 'yes'):
                logger.info("[TEST] Already processed, skipping: %s", abspath)
                return data

    mapper = PluginStreamMapper()
    mapper.set_probe(probe)

    if mapper.streams_need_processing():
        data['add_file_to_pending_tasks'] = True
        logger.info("[TEST] Added to pending tasks: %s", abspath)
    else:
        logger.info("[TEST] No processing needed: %s", abspath)

    return data


# --- Worker process ---

def _check_already_processed(path):
    """Check if file has unmanic-equalize-subtitles-v2 tag."""
    try:
        import subprocess, json as _json
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', path],
            capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return False
        probe_data = _json.loads(result.stdout)
        for k, v in probe_data.get('format', {}).get('tags', {}).items():
            if k.lower() in ('unmanic-equalize-subtitles-v2', 'unmanic_equalize_subtitles_v2'):
                if str(v).lower() in ('true', '1', 'yes'):
                    return True
        return False
    except Exception:
        return False


def on_worker_process(data):
    data['exec_command'] = []
    data['repeat'] = False

    abspath = data.get('file_in')

    # Check metadata tag first — skip if already processed
    if _check_already_processed(abspath):
        logger.info("[WORKER] Already processed (tag found), skipping: %s", abspath)
        return data

    probe = Probe(logger, allowed_mimetypes=['video'])
    if not probe.file(abspath):
        return data

    mapper = PluginStreamMapper()
    mapper.set_probe(probe)

    if not mapper.streams_need_processing():
        logger.info("[WORKER] No work needed: %s", abspath)
        return data

    mapper.set_input_file(abspath)
    mapper.set_output_file(data.get('file_out'))

    ffmpeg_args = mapper.get_ffmpeg_args()

    # Inject metadata tag before the output file (last 2 args are -y output)
    ffmpeg_args = ffmpeg_args[:-2] + ['-metadata', 'unmanic-equalize-subtitles-v2=true'] + ffmpeg_args[-2:]

    data['exec_command'] = ['ffmpeg']
    data['exec_command'] += ffmpeg_args

    logger.info("[WORKER] FFmpeg command: %s", ' '.join(data['exec_command']))

    parser = Parser(logger)
    parser.set_probe(probe)
    data['command_progress_parser'] = parser.parse_progress

    return data
