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

METADATA_TAG_KEY = 'UNMANIC_FULL_PIPELINE'
METADATA_TAG_VALUE = 'processed'


def check_file_has_pipeline_tag(path):
    """
    Check if file has UNMANIC_FULL_PIPELINE=processed format-level metadata tag.
    If this tag exists, the file has already been through the full processing pipeline.

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
        # File already completed the full pipeline — IGNORE COMPLETELY
        # This runs with high priority (last) so it overrides any other plugin's decision
        data['add_file_to_pending_tasks'] = False
        logger.info("File has {}={} tag — full pipeline already completed. "
                     "IGNORING completely: '{}'".format(
                         METADATA_TAG_KEY, METADATA_TAG_VALUE, abspath))

    return data