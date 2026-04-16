#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    unmanic-plugins.plugin.py

    Written by:               Marcos Gabriel <marcosviniciusi@gmail.com@gmail.com>
    Date:                     18 Jan 2025, (23:09 PM)

    Copyright:
        Copyright (C) 2021 Josh Sunnex

        This program is free software: you can redistribute it and/or modify it under the terms of the GNU General
        Public License as published by the Free Software Foundation, version 3.

        This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
        implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
        for more details.

        You should have received a copy of the GNU General Public License along with this program.
        If not, see <https://www.gnu.org/licenses/>.

"""
import logging
import subprocess
import json

# Configure plugin logger
logger = logging.getLogger("Unmanic.Plugin.vm_ignore_metadata_unmanic")

METADATA_TAG_KEY = 'UNMANIC_FULL_PIPELINE_V2'
METADATA_TAG_VALUE = 'processed'

# Bitmap subtitle codecs that still need OCR conversion
BITMAP_SUB_CODECS = {'hdmv_pgs_subtitle', 'dvd_subtitle', 'dvdsub', 'xsub'}


def check_file_has_pipeline_tag(path):
    """
    Check if file has UNMANIC_FULL_PIPELINE_V2=processed format-level metadata tag
    AND no bitmap subtitles remain (PGS/VOBSUB).

    If the tag exists but bitmap subs are still present, the file needs re-processing
    (e.g., OCR plugin was added after the file was first processed).

    :param path: Path to the file
    :return: True if tag exists and no bitmap subs remain, False otherwise
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

        # Check for V2 pipeline tag
        has_tag = False
        if 'format' in probe_data and 'tags' in probe_data['format']:
            for key, value in probe_data['format']['tags'].items():
                if key.upper() == METADATA_TAG_KEY and value == METADATA_TAG_VALUE:
                    has_tag = True
                    break

        if not has_tag:
            return False

        # Tag found — but check if bitmap subtitles still exist
        # (files processed before OCR plugin was installed/fixed)
        probe_streams_cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_streams',
            '-select_streams', 's',
            path
        ]
        streams_result = subprocess.run(probe_streams_cmd, capture_output=True, text=True, timeout=30)
        if streams_result.returncode == 0:
            streams_data = json.loads(streams_result.stdout)
            for s in streams_data.get('streams', []):
                codec = s.get('codec_name', '').lower()
                if codec in BITMAP_SUB_CODECS:
                    logger.info("File has V2 tag but still contains bitmap subtitle ({}), allowing re-process: '{}'".format(
                        codec, path))
                    return False

        logger.debug("Found {}={} in file: '{}'".format(
            METADATA_TAG_KEY, METADATA_TAG_VALUE, path))
        return True

    except subprocess.TimeoutExpired:
        logger.error("ffprobe timed out for file: '{}'".format(path))
        return False
    except json.JSONDecodeError:
        logger.error("Failed to parse ffprobe output for file: '{}'".format(path))
        return False
    except Exception as e:
        logger.error("Error checking metadata for file '{}': {}".format(path, str(e)))
        return False


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

    if check_file_has_pipeline_tag(abspath):
        data['add_file_to_pending_tasks'] = False
        logger.info("File has {}={} tag — IGNORING: '{}'".format(
            METADATA_TAG_KEY, METADATA_TAG_VALUE, abspath))
    else:
        data['add_file_to_pending_tasks'] = True
        logger.info("File missing {}={} tag — adding to queue: '{}'".format(
            METADATA_TAG_KEY, METADATA_TAG_VALUE, abspath))

    return data