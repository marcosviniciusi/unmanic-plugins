#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    unmanic-plugins.plugin.py

    Written by:               Josh.5 <jsunnex@gmail.com>
    Modified by:              Marcos Gabriel
    Date:                     23 Feb 2025

    Copyright:
        Copyright (C) 2021 Josh Sunnex
        Copyright (C) 2025 Marcos Gabriel

        This program is free software: you can redistribute it and/or modify it under the terms of the GNU General
        Public License as published by the Free Software Foundation, version 3.

        This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
        implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
        for more details.

        You should have received a copy of the GNU General Public License along with this program.
        If not, see <https://www.gnu.org/licenses/>.

"""
import logging
import os

from unmanic.libs.unplugins.settings import PluginSettings

from dts_flac_to_dd.lib.ffmpeg import StreamMapper, Probe, Parser

# Configure plugin logger
logger = logging.getLogger("Unmanic.Plugin.dts_flac_to_dd")


class Settings(PluginSettings):
    settings = {
        'create_ac3':       True,
        'ac3_bitrate':      '640k',
        'create_eac3':      True,
        'eac3_bitrate':     '1536k',
        'remove_original':  True,
    }
    
    form_settings = {
        "create_ac3": {
            "label": "Create AC3 track (maximum compatibility)",
            "input_type": "checkbox",
        },
        "ac3_bitrate": {
            "label":      "AC3 bitrate",
            "sub_setting": True,
            "input_type": "select",
            "select_options": [
                {
                    'value': "192k",
                    'label': "192 kbps (2.0 stereo)",
                },
                {
                    'value': "224k",
                    'label': "224 kbps",
                },
                {
                    'value': "256k",
                    'label': "256 kbps (2.0 recommended)",
                },
                {
                    'value': "320k",
                    'label': "320 kbps",
                },
                {
                    'value': "384k",
                    'label': "384 kbps (5.1 balanced)",
                },
                {
                    'value': "448k",
                    'label': "448 kbps (5.1 good)",
                },
                {
                    'value': "640k",
                    'label': "640 kbps (5.1 maximum)",
                },
            ],
        },
        "create_eac3": {
            "label": "Create EAC3 track (better quality)",
            "input_type": "checkbox",
        },
        "eac3_bitrate": {
            "label":      "EAC3 bitrate",
            "sub_setting": True,
            "input_type": "select",
            "select_options": [
                {
                    'value': "256k",
                    'label': "256 kbps (2.0 stereo)",
                },
                {
                    'value': "384k",
                    'label': "384 kbps",
                },
                {
                    'value': "512k",
                    'label': "512 kbps",
                },
                {
                    'value': "768k",
                    'label': "768 kbps (5.1 balanced)",
                },
                {
                    'value': "1024k",
                    'label': "1024 kbps (5.1 good)",
                },
                {
                    'value': "1536k",
                    'label': "1536 kbps (5.1 maximum)",
                },
            ],
        },
        "remove_original": {
            "label": "Remove DTS/FLAC originals after conversion",
            "input_type": "checkbox",
        },
    }


class PluginStreamMapper(StreamMapper):
    def __init__(self):
        super(PluginStreamMapper, self).__init__(logger, ['audio'])
        self.settings = None
        self.stream_encoding_map = {}

    def set_settings(self, settings):
        self.settings = settings

    def should_process_stream(self, probe_stream):
        """
        Check if this stream should be processed (DTS or FLAC)
        """
        codec_name = probe_stream.get('codec_name', '').lower()
        
        # Process all DTS variants
        if codec_name == 'dts':
            return True
            
        # Process all FLAC
        if codec_name == 'flac':
            return True
        
        return False

    def get_output_bitrate(self, codec_type, channels):
        """
        Get appropriate bitrate based on codec and channel count
        
        Args:
            codec_type: 'ac3' or 'eac3'
            channels: number of channels
        
        Returns:
            bitrate string (e.g., '640k')
        """
        if codec_type == 'ac3':
            bitrate = self.settings.get_setting('ac3_bitrate')
        else:  # eac3
            bitrate = self.settings.get_setting('eac3_bitrate')
        
        return bitrate

    def test_stream_needs_processing(self, stream_info: dict):
        """
        Test if this stream needs processing
        """
        if self.should_process_stream(stream_info):
            return True
        return False

    def custom_stream_mapping(self, stream_info: dict, stream_id: int):
        """
        Create custom stream mapping for DTS/FLAC conversion
        
        This will create AC3 and/or EAC3 tracks based on settings
        """
        codec_name = stream_info.get('codec_name', '').lower()
        channels = stream_info.get('channels', 2)
        
        logger.info(f"Processing {codec_name.upper()} stream (ID: {stream_id}, Channels: {channels})")
        
        # Store original stream info for later removal decision
        stream_mapping = []
        stream_encoding = []
        
        # Track counter for this source stream
        output_track_count = 0
        
        # Create AC3 track if enabled
        if self.settings.get_setting('create_ac3'):
            ac3_bitrate = self.get_output_bitrate('ac3', channels)
            
            # Get original language if available
            original_lang = stream_info.get('tags', {}).get('language', '')
            
            # Map the source stream
            stream_mapping += ['-map', f'0:a:{stream_id}']
            
            # Encode to AC3
            output_stream_id = stream_id + output_track_count
            
            # Build title based on channels
            if channels == 1:
                title = f'AC3 Mono ({ac3_bitrate})'
            elif channels == 2:
                title = f'AC3 Stereo ({ac3_bitrate})'
            elif channels == 6:
                title = f'AC3 5.1 ({ac3_bitrate})'
            elif channels == 8:
                title = f'AC3 7.1 ({ac3_bitrate})'
            else:
                title = f'AC3 {channels}.0 ({ac3_bitrate})'
            
            stream_encoding += [
                f'-c:a:{output_stream_id}', 'ac3',
                f'-b:a:{output_stream_id}', ac3_bitrate,
                f'-metadata:s:a:{output_stream_id}', f'title={title}',
            ]
            
            # Preserve language if available
            if original_lang:
                stream_encoding += [f'-metadata:s:a:{output_stream_id}', f'language={original_lang}']
            
            logger.info(f"Creating AC3 track: {ac3_bitrate}")
            output_track_count += 1
        
        # Create EAC3 track if enabled
        if self.settings.get_setting('create_eac3'):
            eac3_bitrate = self.get_output_bitrate('eac3', channels)
            
            # Get original language if available
            original_lang = stream_info.get('tags', {}).get('language', '')
            
            # Map the source stream again
            stream_mapping += ['-map', f'0:a:{stream_id}']
            
            # Encode to EAC3
            output_stream_id = stream_id + output_track_count
            
            # Build title based on channels
            if channels == 1:
                title = f'EAC3 Mono ({eac3_bitrate})'
            elif channels == 2:
                title = f'EAC3 Stereo ({eac3_bitrate})'
            elif channels == 6:
                title = f'EAC3 5.1 ({eac3_bitrate})'
            elif channels == 8:
                title = f'EAC3 7.1 ({eac3_bitrate})'
            else:
                title = f'EAC3 {channels}.0 ({eac3_bitrate})'
            
            stream_encoding += [
                f'-c:a:{output_stream_id}', 'eac3',
                f'-b:a:{output_stream_id}', eac3_bitrate,
                f'-metadata:s:a:{output_stream_id}', f'title={title}',
            ]
            
            # Preserve language if available
            if original_lang:
                stream_encoding += [f'-metadata:s:a:{output_stream_id}', f'language={original_lang}']
            
            logger.info(f"Creating EAC3 track: {eac3_bitrate}")
            output_track_count += 1
        
        # If remove_original is disabled, also map the original stream
        if not self.settings.get_setting('remove_original'):
            stream_mapping += ['-map', f'0:a:{stream_id}']
            
            # Copy original codec
            output_stream_id = stream_id + output_track_count
            stream_encoding += [f'-c:a:{output_stream_id}', 'copy']
            
            logger.info(f"Keeping original {codec_name.upper()} track")
        else:
            logger.info(f"Removing original {codec_name.upper()} track")
        
        return {
            'stream_mapping':  stream_mapping,
            'stream_encoding': stream_encoding,
        }


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
    # Get the path to the file
    abspath = data.get('path')

    # Get file probe
    probe = Probe(logger, allowed_mimetypes=['audio', 'video'])
    if not probe.file(abspath):
        # File probe failed, skip the rest of this test
        return data

    # Configure settings object (maintain compatibility with v1 plugins)
    if data.get('library_id'):
        settings = Settings(library_id=data.get('library_id'))
    else:
        settings = Settings()

    # Get stream mapper
    mapper = PluginStreamMapper()
    mapper.set_settings(settings)
    mapper.set_probe(probe)

    if mapper.streams_need_processing():
        # Mark this file to be added to the pending tasks
        data['add_file_to_pending_tasks'] = True
        logger.debug("File '{}' should be added to task list. Probe found streams require processing.".format(abspath))
    else:
        logger.debug("File '{}' does not contain streams require processing.".format(abspath))

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

    DEPRECIATED 'data' object args passed for legacy Unmanic versions:
        exec_ffmpeg             - Boolean, should Unmanic run FFMPEG with the data returned from this plugin.
        ffmpeg_args             - A list of Unmanic's default FFMPEG args.

    :param data:
    :return:

    """
    # Default to no FFMPEG command required. This prevents the FFMPEG command from running if it is not required
    data['exec_command'] = []
    data['repeat'] = False
    # DEPRECIATED: 'exec_ffmpeg' kept for legacy Unmanic versions
    data['exec_ffmpeg'] = False

    # Get the path to the file
    abspath = data.get('file_in')

    # Get file probe
    probe = Probe(logger, allowed_mimetypes=['audio', 'video'])
    if not probe.file(abspath):
        # File probe failed, skip the rest of this test
        return data

    # Configure settings object (maintain compatibility with v1 plugins)
    if data.get('library_id'):
        settings = Settings(library_id=data.get('library_id'))
    else:
        settings = Settings()

    # Get stream mapper
    mapper = PluginStreamMapper()
    mapper.set_settings(settings)
    mapper.set_probe(probe)

    if mapper.streams_need_processing():
        # Set the input file
        mapper.set_input_file(abspath)

        # Set the output file
        # Do not remux the file. Keep the file out in the same container
        split_file_in = os.path.splitext(abspath)
        split_file_out = os.path.splitext(data.get('file_out'))
        mapper.set_output_file("{}{}".format(split_file_out[0], split_file_in[1]))

        # Get generated ffmpeg args
        ffmpeg_args = mapper.get_ffmpeg_args()

        # Apply ffmpeg args to command
        data['exec_command'] = ['ffmpeg']
        data['exec_command'] += ffmpeg_args
        # DEPRECIATED: 'ffmpeg_args' kept for legacy Unmanic versions
        data['ffmpeg_args'] = ffmpeg_args

        # Set the parser
        parser = Parser(logger)
        parser.set_probe(probe)
        data['command_progress_parser'] = parser.parse_progress

    return data
