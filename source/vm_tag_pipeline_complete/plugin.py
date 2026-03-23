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

from vm_tag_pipeline_complete.lib.ffmpeg import Probe, Parser

# Configure plugin logger
logger = logging.getLogger("Unmanic.Plugin.vm_tag_pipeline_complete")

METADATA_TAG_KEY = 'UNMANIC_FULL_PIPELINE_V2'
METADATA_TAG_VALUE = 'processed'


def check_file_has_pipeline_tag(path):
    """Check if file has UNMANIC_FULL_PIPELINE_V2=processed format-level metadata tag."""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', path],
            capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return False
        probe_data = json.loads(result.stdout)
        for key, value in probe_data.get('format', {}).get('tags', {}).items():
            if key.upper() == METADATA_TAG_KEY and value == METADATA_TAG_VALUE:
                return True
        return False
    except Exception:
        return False


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

    # Check metadata tag first — skip if already processed
    if check_file_has_pipeline_tag(abspath):
        logger.info("[WORKER] Already has {}={} tag, skipping: {}".format(
            METADATA_TAG_KEY, METADATA_TAG_VALUE, abspath))
        return data

    # Probe the file to validate it's a media file
    probe = Probe(logger, allowed_mimetypes=['audio', 'video'])
    if not probe.file(abspath):
        logger.warning("File '{}' is not a valid media file. Skipping.".format(abspath))
        return data

    # Build FFmpeg command to remux with the pipeline tag
    ffmpeg_args = [
        '-hide_banner',
        '-loglevel', 'info',
        '-i', os.path.abspath(abspath),
    ]

    # Add the full pipeline metadata tag
    ffmpeg_args += ['-metadata', '{}={}'.format(METADATA_TAG_KEY.lower(), METADATA_TAG_VALUE)]

    # Map ALL streams (copy everything untouched)
    ffmpeg_args += ['-map', '0', '-c', 'copy']

    # Additional safety options
    ffmpeg_args += ['-strict', '-2', '-max_muxing_queue_size', '4096']

    # Keep same container extension
    split_file_in = os.path.splitext(abspath)
    split_file_out = os.path.splitext(data.get('file_out'))
    output_file = "{}{}".format(split_file_out[0], split_file_in[1])

    ffmpeg_args += ['-y', output_file]

    # Apply ffmpeg args to command
    data['exec_command'] = ['ffmpeg'] + ffmpeg_args

    logger.info("Writing {}={} tag to '{}'".format(
        METADATA_TAG_KEY, METADATA_TAG_VALUE, abspath))

    # Set the parser
    parser = Parser(logger)
    parser.set_probe(probe)
    data['command_progress_parser'] = parser.parse_progress

    return data
