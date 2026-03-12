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
logger = logging.getLogger("Unmanic.Plugin.ignore_files_with_processed_metadata")


def check_file_has_metadata(path):
    """
    Check if file has UNMANIC_STATUS=processed metadata using ffprobe
    
    :param path: Path to the file
    :return: True if metadata exists with matching value, False otherwise
    """
    try:
        # Use ffprobe to get file metadata in JSON format
        command = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            path
        ]
        
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            logger.warning(f"ffprobe failed for file: {path}")
            return False
        
        # Parse JSON output
        probe_data = json.loads(result.stdout)
        
        # Check if format tags exist
        if 'format' in probe_data and 'tags' in probe_data['format']:
            tags = probe_data['format']['tags']
            
            # Check for UNMANIC_STATUS metadata (case-insensitive)
            for key, value in tags.items():
                if key.upper() == 'UNMANIC_STATUS':
                    if value == 'processed':
                        logger.debug(f"Found UNMANIC_STATUS=processed in file: {path}")
                        return True
        
        return False
        
    except subprocess.TimeoutExpired:
        logger.error(f"ffprobe timed out for file: {path}")
        return False
    except json.JSONDecodeError:
        logger.error(f"Failed to parse ffprobe output for file: {path}")
        return False
    except Exception as e:
        logger.error(f"Error checking metadata for file {path}: {str(e)}")
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
    if check_file_has_metadata(data.get('path')):
        # File already has the processed metadata, ignore it
        data['add_file_to_pending_tasks'] = False
        logger.info(f"File already processed (has UNMANIC_STATUS=processed). Ignoring: {data.get('path')}")

    return data