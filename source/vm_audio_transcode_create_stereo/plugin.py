#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    plugins.__init__.py

    Written by:               Josh.5 <jsunnex@gmail.com>
    Date:                     28 Sep 2021, (9:20 PM)

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
import json
import re
import subprocess

from unmanic.libs.unplugins.settings import PluginSettings

from vm_audio_transcode_create_stereo.lib.ffmpeg import StreamMapper, Probe, Parser

# Configure plugin logger
logger = logging.getLogger("Unmanic.Plugin.vm_audio_transcode_create_stereo")

METADATA_TAG_KEY = 'UNMANIC_STEREO'
METADATA_TAG_VALUE = 'processed'


def check_file_has_stereo_tag(path):
    """
    Check if file has UNMANIC_STEREO=processed format-level metadata tag.

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
                    logger.debug("Found {}={} in file: '{}'".format(METADATA_TAG_KEY, METADATA_TAG_VALUE, path))
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


class Settings(PluginSettings):
    settings = {
        "encoder":               "aac",
        "advanced":              False,
        "max_muxing_queue_size": 2048,
        "main_options":          "",
        "advanced_options":      "",
        "custom_options":        "",
    }

    def __init__(self, *args, **kwargs):
        super(Settings, self).__init__(*args, **kwargs)
        self.form_settings = {
            "encoder":               {
                "label":          "Encoder",
                "input_type":     "select",
                "select_options": [
                    {
                        'value': "aac",
                        'label': "AAC (Advanced Audio Coding)",
                    },
                    {
                        'value': "ac3",
                        'label': "ATSC A/52A (AC-3)",
                    },
                ],
            },
            "advanced":              {
                "label": "Write your own FFmpeg params",
            },
            "max_muxing_queue_size": self.__set_max_muxing_queue_size_form_settings(),
            "main_options":          self.__set_main_options_form_settings(),
            "advanced_options":      self.__set_advanced_options_form_settings(),
            "custom_options":        self.__set_custom_options_form_settings(),
        }

    def __set_max_muxing_queue_size_form_settings(self):
        values = {
            "label":          "Max input stream packet buffer",
            "input_type":     "slider",
            "slider_options": {
                "min": 1024,
                "max": 10240,
            },
        }
        if self.get_setting('advanced'):
            values["display"] = 'hidden'
        return values

    def __set_main_options_form_settings(self):
        values = {
            "label":      "Write your own custom main options",
            "input_type": "textarea",
        }
        if not self.get_setting('advanced'):
            values["display"] = 'hidden'
        return values

    def __set_advanced_options_form_settings(self):
        values = {
            "label":      "Write your own custom advanced options",
            "input_type": "textarea",
        }
        if not self.get_setting('advanced'):
            values["display"] = 'hidden'
        return values

    def __set_custom_options_form_settings(self):
        values = {
            "label":      "Write your own custom audio options",
            "input_type": "textarea",
        }
        if not self.get_setting('advanced'):
            values["display"] = 'hidden'
        return values


class PluginStreamMapper(StreamMapper):
    def __init__(self):
        super(PluginStreamMapper, self).__init__(logger, ['audio'])

        self.audio_stream_tags = []
        self.audio_streams_info = []
        self.stream_count = 0
        self.stereo_mapping = []
        self.stereo_encoding = []

        self.settings = None

    def set_settings(self, settings):
        self.settings = settings

    def fetch_all_audio_stream_tags(self):
        # Require a list of probe streams to continue
        file_probe_streams = self.probe.get('streams')
        if not file_probe_streams:
            return False
        # Loop over all streams found in the file probe
        for stream_info in file_probe_streams:
            # If this is an audio stream
            if stream_info.get('codec_type', '').lower() == "audio":
                # Append to stereo stream tags... This allows us to ignore streams that are already downmixed
                self.audio_stream_tags.append(stream_info.get('tags', {}).get('title', ''))
                # Track full stream info for channel/codec checks
                self.audio_streams_info.append(stream_info)
                self.stream_count += 1

    def generate_legacy_audio_stream_tags(self, stream_info):
        legacy_tags = []
        try:
            audio_tag = ''.join([i for i in stream_info['tags']['title'] if not i.isdigit()]).rstrip('.') + 'Stereo'
        except:
            audio_tag = 'Stereo'

        legacy_tags.append(audio_tag)
        return legacy_tags

    def generate_audio_stream_tag(self, stream_info):
        """
        Generate a descriptive title for the stereo clone stream.
        Format: "{Language} {CODEC} stereo (Padrão)"
        Example: "Korean AAC stereo (Padrão)"
        """
        encoder = self.settings.get_setting('encoder') if self.settings else 'aac'
        codec_label = encoder.upper()

        language = stream_info.get('tags', {}).get('language', '')
        # Try to get a human-readable language name from the title
        original_title = stream_info.get('tags', {}).get('title', '')

        # Extract language name from original title (first word if it looks like a language name)
        lang_name = ''
        if original_title:
            # Take the first word of the original title as language name
            first_word = original_title.split()[0] if original_title.split() else ''
            # Check if first word is a language name (starts with uppercase, no digits)
            if first_word and first_word[0].isupper() and not any(c.isdigit() for c in first_word):
                lang_name = first_word

        if lang_name:
            audio_tag = '{} {} stereo (Padrão)'.format(lang_name, codec_label)
        elif language:
            audio_tag = '{} {} stereo (Padrão)'.format(language.upper(), codec_label)
        else:
            audio_tag = '{} stereo (Padrão)'.format(codec_label)

        return audio_tag

    def generate_legacy_audio_stream_tag(self, stream_info):
        """Generate the old-style tag for backwards compatibility checks."""
        title = stream_info.get('tags', {}).get('title', '')
        audio_tag = re.sub(r'\d+\.*\d*', ' ', title)
        audio_tag = re.sub(r'\s\s+', ' ', audio_tag)
        audio_tag = audio_tag.rstrip()
        audio_tag += ' [Stereo]'
        audio_tag = audio_tag.lstrip()
        return audio_tag

    def audio_tag_already_exists(self, audio_tag, legacy_audio_tags):
        if audio_tag in self.audio_stream_tags:
            return audio_tag
        # Also check for legacy tags
        # Some of the legacy unmanic converted streams had quotes around the title.
        # Because of this we need to do more than just check for an exact match
        for legacy_audio_tag in legacy_audio_tags:
            if legacy_audio_tag in self.audio_stream_tags or "'{}'".format(legacy_audio_tag) in self.audio_stream_tags:
                return legacy_audio_tag
        return False

    def stereo_already_exists_by_streams(self, stream_info):
        """
        Check if a stereo version already exists by inspecting audio stream properties.

        Layer 2: Check if a ≤2ch stream with the same language already exists.
        Layer 3: Check if a stream with the configured encoder codec + 2 channels exists.

        :param stream_info: The surround stream to check against
        :return: True if stereo clone already exists, False otherwise
        """
        language = stream_info.get('tags', {}).get('language', '')
        encoder = self.settings.get_setting('encoder') if self.settings else 'aac'

        for other_stream in self.audio_streams_info:
            other_channels = int(other_stream.get('channels', 0))
            if other_channels > 2:
                continue  # Skip surround streams

            other_language = other_stream.get('tags', {}).get('language', '')
            other_codec = other_stream.get('codec_name', '').lower()

            # Layer 2: Same language + ≤2 channels
            if language and other_language and language == other_language:
                logger.debug(
                    "Stream #{} already has a stereo stream in language '{}' (stream #{}).".format(
                        stream_info.get('index'), language, other_stream.get('index')))
                return True

            # Layer 3: Configured codec + 2 channels
            if other_codec == encoder and other_channels == 2:
                logger.debug(
                    "Stream #{} already has a {} stereo stream (stream #{}).".format(
                        stream_info.get('index'), encoder.upper(), other_stream.get('index')))
                return True

        return False

    def test_stream_needs_processing(self, stream_info: dict):
        channels = stream_info.get('channels')
        if not channels:
            logger.debug("Unable to determine number of channels in stream.")
            return False

        # Check if this stream has more than 2 channels
        if int(channels) > 2:
            # Check layer 2+3: stereo already exists by stream analysis
            if self.stereo_already_exists_by_streams(stream_info):
                return False

            # Check by title tag (original logic + legacy compatibility)
            legacy_audio_tags = self.generate_legacy_audio_stream_tags(stream_info)
            audio_tag = self.generate_audio_stream_tag(stream_info)
            legacy_tag = self.generate_legacy_audio_stream_tag(stream_info)
            existing_tag = self.audio_tag_already_exists(audio_tag, [legacy_tag] + legacy_audio_tags)
            if not existing_tag:
                # No stereo clone found — needs processing
                return True
            else:
                logger.debug(
                    "Stream #{} already has a stereo clone with the tag: '{}'.".format(
                        stream_info.get('index'), existing_tag))

        return False

    def custom_stream_mapping(self, stream_info: dict, stream_id: int):
        encoder = self.settings.get_setting('encoder')

        audio_tag = self.generate_audio_stream_tag(stream_info)

        stream_mapping = [
            '-map', '0:a:{}'.format(stream_id),
        ]
        stream_encoding = [
            '-c:a:{}'.format(self.stream_count), encoder,
        ]
        if self.settings.get_setting('advanced'):
            stream_encoding += self.settings.get_setting('custom_options').split()

        # Set channels and map title metadata
        stream_encoding += [
            "-ac", "2",
            "-metadata:s:a:{}".format(self.stream_count), "title={}".format(audio_tag),
        ]
        # Set language metadata
        if stream_info.get('tags', {}).get('language'):
            stream_encoding += [
                "-metadata:s:a:{}".format(self.stream_count),
                "language={}".format(stream_info.get('tags', {}).get('language')),
            ]

        # Store the stereo mapping to be appended later on
        self.stereo_mapping += stream_mapping
        self.stereo_encoding += stream_encoding

        # Increment the audio stream counter
        self.stream_count += 1

        # Copy the original stream
        return {
            'stream_mapping':  ['-map', '0:a:{}'.format(stream_id)],
            'stream_encoding': ['-c:a:{}'.format(stream_id), 'copy'],
        }

    def append_stereo_mapping(self):
        self.stream_mapping += self.stereo_mapping
        self.stream_encoding += self.stereo_encoding


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

    # Layer 1: Check format-level metadata tag (fastest exit)
    if check_file_has_stereo_tag(abspath):
        logger.debug("File '{}' already has {}={} tag. Skipping.".format(
            abspath, METADATA_TAG_KEY, METADATA_TAG_VALUE))
        return data

    # Get file probe
    probe = Probe(logger, allowed_mimetypes=['video'])
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
    mapper.fetch_all_audio_stream_tags()

    # Layers 2+3 are checked inside streams_need_processing → test_stream_needs_processing
    if mapper.streams_need_processing():
        # Mark this file to be added to the pending tasks
        data['add_file_to_pending_tasks'] = True
        logger.debug("File '{}' should be added to task list. Probe found streams require processing.".format(abspath))
    else:
        logger.debug("File '{}' does not contain streams requiring processing.".format(abspath))

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

    # Get the path to the file
    abspath = data.get('file_in')

    # Get file probe
    probe = Probe(logger, allowed_mimetypes=['video'])
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
    mapper.fetch_all_audio_stream_tags()

    if mapper.streams_need_processing():
        # Set the input file
        mapper.set_input_file(abspath)

        # Set the output file
        mapper.set_output_file(data.get('file_out'))

        # Append final clone mapping
        mapper.append_stereo_mapping()

        # Write format-level metadata tag to prevent reprocessing
        mapper.set_ffmpeg_advanced_options(**{'-metadata': '{}={}'.format(
            METADATA_TAG_KEY.lower(), METADATA_TAG_VALUE)})
        logger.info("Adding metadata tag '{}={}' to output file".format(
            METADATA_TAG_KEY.lower(), METADATA_TAG_VALUE))

        # Get generated ffmpeg args
        ffmpeg_args = mapper.get_ffmpeg_args()

        # Apply ffmpeg args to command
        data['exec_command'] = ['ffmpeg']
        data['exec_command'] += ffmpeg_args

        # Set the parser
        parser = Parser(logger)
        parser.set_probe(probe)
        data['command_progress_parser'] = parser.parse_progress

    return data
