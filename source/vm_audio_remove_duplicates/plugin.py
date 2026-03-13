#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    unmanic-plugins.plugin.py

    Written by:               Marcos Gabriel
    Date:                     13 Mar 2026

    Copyright:
        Copyright (C) 2026 Marcos Gabriel

        This program is free software: you can redistribute it and/or modify it under the terms of the GNU General
        Public License as published by the Free Software Foundation, version 3.

        This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
        implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
        for more details.

        You should have received a copy of the GNU General Public License along with this program.
        If not, see <https://www.gnu.org/licenses/>.

"""
import json
import logging
import os
import subprocess

from vm_audio_remove_duplicates.lib.ffmpeg import Probe, Parser

# Configure plugin logger
logger = logging.getLogger("Unmanic.Plugin.vm_audio_remove_duplicates")

METADATA_TAG_KEY = 'UNMANIC_FIX_AUDIO'
METADATA_TAG_VALUE = 'processed'


def check_file_has_tag(path):
    """
    Check if file has UNMANIC_FIX_AUDIO=processed format-level metadata tag.

    :param path: Path to the file
    :return: True if tag exists, False otherwise
    """
    try:
        command = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            path
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            logger.warning("ffprobe failed for file: '{}'".format(path))
            return False
        probe_data = json.loads(result.stdout)
        if 'format' in probe_data and 'tags' in probe_data['format']:
            for key, value in probe_data['format']['tags'].items():
                if key.upper() == METADATA_TAG_KEY and value == METADATA_TAG_VALUE:
                    logger.debug("Found {}={} in file: '{}'".format(
                        METADATA_TAG_KEY, METADATA_TAG_VALUE, path))
                    return True
        return False
    except subprocess.TimeoutExpired:
        logger.error("ffprobe timed out for file: '{}'".format(path))
        return False
    except json.JSONDecodeError:
        logger.error("Failed to parse ffprobe output for file: '{}'".format(path))
        return False
    except Exception as e:
        logger.error("Error checking metadata for file '{}': {}".format(path, str(e)))
        return False


def get_audio_stream_fingerprint(stream_info):
    """
    Generate a fingerprint tuple for an audio stream based on all relevant specs.
    Two streams are considered duplicates if their fingerprints are identical.

    :param stream_info: Stream info dict from ffprobe
    :return: tuple of (codec_name, channels, language, title, bit_rate)
    """
    codec_name = stream_info.get('codec_name', '').lower()
    channels = str(stream_info.get('channels', ''))
    language = stream_info.get('tags', {}).get('language', '')
    title = stream_info.get('tags', {}).get('title', '')
    bit_rate = stream_info.get('bit_rate', stream_info.get('tags', {}).get('BPS', ''))
    return (codec_name, channels, language, title, str(bit_rate))


def find_duplicate_audio_streams(probe):
    """
    Analyze all audio streams and identify duplicates.

    :param probe: Probe object with file data
    :return: tuple (has_duplicates: bool, streams_to_keep: set of stream indices, all_audio_indices: list)
    """
    file_probe_streams = probe.get('streams')
    if not file_probe_streams:
        return False, set(), []

    # Collect audio streams with their absolute indices
    audio_streams = []
    for stream_info in file_probe_streams:
        if stream_info.get('codec_type', '').lower() == 'audio':
            audio_streams.append(stream_info)

    if not audio_streams:
        return False, set(), []

    # Group by fingerprint
    seen = {}
    streams_to_keep = set()
    has_duplicates = False

    for stream_info in audio_streams:
        fingerprint = get_audio_stream_fingerprint(stream_info)
        abs_index = stream_info.get('index')

        if fingerprint not in seen:
            seen[fingerprint] = abs_index
            streams_to_keep.add(abs_index)
        else:
            has_duplicates = True
            logger.debug(
                "Duplicate audio stream found: index={}, fingerprint={} "
                "(duplicate of stream {})".format(abs_index, fingerprint, seen[fingerprint]))

    all_audio_indices = [s.get('index') for s in audio_streams]
    return has_duplicates, streams_to_keep, all_audio_indices


def on_library_management_file_test(data):
    """
    Runner function - enables additional actions during the library management file tests.

    The 'data' object argument includes:
        path                            - String containing the full path to the file being tested.
        issues                          - List of currently found issues for not processing the file.
        add_file_to_pending_tasks       - Boolean, is the file currently marked to be added to the queue for processing.

    :param data:
    :return:
    """
    abspath = data.get('path')

    # Layer 1: Check format-level metadata tag (fastest exit)
    if check_file_has_tag(abspath):
        logger.debug("File '{}' already has {}={} tag. Skipping.".format(
            abspath, METADATA_TAG_KEY, METADATA_TAG_VALUE))
        return data

    # Probe the file
    probe = Probe(logger, allowed_mimetypes=['audio', 'video'])
    if not probe.file(abspath):
        return data

    # Check for duplicate audio streams
    has_duplicates, _, _ = find_duplicate_audio_streams(probe)

    if has_duplicates:
        data['add_file_to_pending_tasks'] = True
        logger.debug("File '{}' has duplicate audio streams. Adding to task list.".format(abspath))
    else:
        logger.debug("File '{}' has no duplicate audio streams.".format(abspath))

    return data


def on_worker_process(data):
    """
    Runner function - enables additional configured processing jobs during the worker stages of a task.

    The 'data' object argument includes:
        exec_command            - A command that Unmanic should execute. Can be empty.
        command_progress_parser - A function that Unmanic can use to parse the STDOUT of the command to collect progress stats. Can be empty.
        file_in                 - The source file to be processed by the command.
        file_out                - The destination that the command should output (may be the same as the file_in if necessary).
        original_file_path      - The absolute path to the original file.
        repeat                  - Boolean, should this runner be executed again once completed with the same variables.

    :param data:
    :return:
    """
    data['exec_command'] = []
    data['repeat'] = False

    abspath = data.get('file_in')

    # Probe the file
    probe = Probe(logger, allowed_mimetypes=['audio', 'video'])
    if not probe.file(abspath):
        return data

    # Check for duplicate audio streams
    has_duplicates, streams_to_keep, all_audio_indices = find_duplicate_audio_streams(probe)

    if not has_duplicates:
        logger.debug("File '{}' has no duplicate audio streams. Nothing to do.".format(abspath))
        return data

    # Build FFmpeg command manually (not using StreamMapper since we need selective stream mapping)
    file_probe_streams = probe.get('streams')

    ffmpeg_args = [
        '-hide_banner',
        '-loglevel', 'info',
        '-i', os.path.abspath(abspath),
    ]

    # Add metadata tag to prevent reprocessing
    ffmpeg_args += ['-metadata', '{}={}'.format(METADATA_TAG_KEY.lower(), METADATA_TAG_VALUE)]

    # Map streams selectively
    stream_mapping = []
    stream_encoding = []

    # Track counts per codec type for encoding indices
    video_count = 0
    audio_count = 0
    subtitle_count = 0

    removed_count = 0

    for stream_info in file_probe_streams:
        codec_type = stream_info.get('codec_type', '').lower()
        abs_index = stream_info.get('index')

        if codec_type == 'video':
            stream_mapping += ['-map', '0:{}'.format(abs_index)]
            stream_encoding += ['-c:v:{}'.format(video_count), 'copy']
            video_count += 1

        elif codec_type == 'audio':
            if abs_index in streams_to_keep:
                stream_mapping += ['-map', '0:{}'.format(abs_index)]
                stream_encoding += ['-c:a:{}'.format(audio_count), 'copy']
                audio_count += 1
            else:
                removed_count += 1
                logger.info("Removing duplicate audio stream index={} fingerprint={}".format(
                    abs_index, get_audio_stream_fingerprint(stream_info)))

        elif codec_type == 'subtitle':
            stream_mapping += ['-map', '0:{}'.format(abs_index)]
            stream_encoding += ['-c:s:{}'.format(subtitle_count), 'copy']
            subtitle_count += 1

        # Skip data/attachment streams (copy them if needed)
        elif codec_type in ('data', 'attachment'):
            stream_mapping += ['-map', '0:{}'.format(abs_index)]

    logger.info("Removing {} duplicate audio stream(s) from '{}'".format(removed_count, abspath))

    # Build complete FFmpeg args
    ffmpeg_args += ['-strict', '-2', '-max_muxing_queue_size', '4096']
    ffmpeg_args += stream_mapping
    ffmpeg_args += stream_encoding

    # Keep same container extension
    split_file_in = os.path.splitext(abspath)
    split_file_out = os.path.splitext(data.get('file_out'))
    output_file = "{}{}".format(split_file_out[0], split_file_in[1])

    ffmpeg_args += ['-y', output_file]

    # Apply ffmpeg args to command
    data['exec_command'] = ['ffmpeg'] + ffmpeg_args

    # Set the parser
    parser = Parser(logger)
    parser.set_probe(probe)
    data['command_progress_parser'] = parser.parse_progress

    return data
