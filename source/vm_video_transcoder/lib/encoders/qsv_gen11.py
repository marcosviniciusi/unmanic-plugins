#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    plugins.qsv_gen11.py

    Written by:               marcosviniciusi
    Date:                     07 Apr 2026

    Copyright:
        Copyright (C) 2026 marcosviniciusi

        This program is free software: you can redistribute it and/or modify it under the terms of the GNU General
        Public License as published by the Free Software Foundation, version 3.

        This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
        implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
        for more details.

        You should have received a copy of the GNU General Public License along with this program.
        If not, see <https://www.gnu.org/licenses/>.

"""
"""
Notes:
    - Intel Gen 11+ (Tiger Lake / i5-1135G7 and newer) QSV encoder with 8-bit/10-bit profile selection.
    - Provides the same functionality as the standard QSV encoder, plus a profile dropdown
      that allows the user to force 8-bit (Main) or 10-bit (Main10) encoding, or auto-detect from source.
    - Uses virtual encoder names (e.g. hevc_qsv_gen11) internally, mapped to real FFmpeg names (hevc_qsv).
"""

from vm_video_transcoder.lib.encoders.qsv import QsvEncoder
import logging

logger = logging.getLogger("Unmanic.Plugin.vm_video_transcoder")


# Map virtual encoder names to real FFmpeg encoder names
_REAL_ENCODER_MAP = {
    "h264_qsv_gen11": "h264_qsv",
    "hevc_qsv_gen11": "hevc_qsv",
    "av1_qsv_gen11":  "av1_qsv",
}


class QsvGen11Encoder(QsvEncoder):
    """
    QSV encoder for Intel Gen 11+ (Tiger Lake and newer) with 8-bit/10-bit profile selection.

    Inherits all QSV functionality (preset, rate control, quality, HW decoding) and adds
    a profile selector (Auto/Main/Main10) similar to VideoToolbox.

    Gen 11+ Intel GPUs have proper hardware 10-bit encoding support, making manual
    profile selection useful for:
    - Encoding SDR 8-bit content in 10-bit for improved gradient quality
    - Forcing 8-bit output from 10-bit/HDR sources for smaller files
    """

    def __init__(self, settings=None, probe=None):
        super().__init__(settings=settings, probe=probe)

    @staticmethod
    def _real_name(encoder_name):
        """Map virtual encoder name to real FFmpeg encoder name."""
        return _REAL_ENCODER_MAP.get(encoder_name, encoder_name)

    def provides(self):
        return {
            "h264_qsv_gen11": {
                "codec":          "h264",
                "label":          "QSV Gen11+ - h264_qsv (Tiger Lake+)",
                "ffmpeg_encoder": "h264_qsv",
            },
            "hevc_qsv_gen11": {
                "codec":          "hevc",
                "label":          "QSV Gen11+ - hevc_qsv (Tiger Lake+)",
                "ffmpeg_encoder": "hevc_qsv",
            },
            "av1_qsv_gen11": {
                "codec":          "av1",
                "label":          "QSV Gen11+ - av1_qsv (Tiger Lake+)",
                "ffmpeg_encoder": "av1_qsv",
            },
        }

    def options(self):
        # Explicit ordering: profile appears between HW decode and preset in the UI
        return {
            "qsv_decoding_method":            "cpu",
            "qsv_gen11_profile":              "auto",
            "qsv_preset":                     "slow",
            "qsv_encoder_ratecontrol_method": "LA_ICQ",
            "qsv_constant_quantizer_scale":   "25",
            "qsv_constant_quality_scale":     "23",
            "qsv_average_bitrate":            "5",
        }

    def encoder_details(self, encoder):
        provides = self.provides()
        return provides.get(encoder, {})

    # ── Pixel format and color config overrides ──────────────────────────

    def _target_pix_fmt_for_encoder(self, encoder_name: str) -> str:
        """
        Override to check the user-selected profile first.
        If user explicitly chose a profile, force the corresponding pixel format.
        Otherwise fall back to the base class auto-detection logic.
        """
        real_name = self._real_name(encoder_name)
        profile = self.settings.get_setting('qsv_gen11_profile') if self.settings else None
        is_h264 = "h264" in real_name.lower()

        if profile and profile != 'auto' and not is_h264:
            if profile == 'main10':
                return "p010le"
            elif profile == 'main':
                return "nv12"

        # Fall back to base class auto-detection with real name
        return super()._target_pix_fmt_for_encoder(real_name)

    def _target_color_config_for_encoder(self, encoder_name: str):
        """
        Override: when forcing 8-bit (Main profile), strip all HDR metadata.
        Otherwise delegate to parent.
        """
        real_name = self._real_name(encoder_name)
        profile = self.settings.get_setting('qsv_gen11_profile') if self.settings else None
        is_h264 = "h264" in real_name.lower()

        if profile == 'main' and not is_h264:
            # Forcing 8-bit: no HDR metadata
            return {
                "apply_color_params":  False,
                "setparams_filter":    None,
                "color_tags":          {},
                "stream_color_params": {},
            }

        return super()._target_color_config_for_encoder(real_name)

    # ── Filtergraph and encoding args ────────────────────────────────────

    def generate_default_args(self):
        """Same HW setup as standard QSV."""
        return super().generate_default_args()

    def generate_filtergraphs(self, current_filter_args, smart_filters, encoder_name):
        """Delegate to parent QSV with real encoder name.
        The overrides of _target_pix_fmt_for_encoder and _target_color_config_for_encoder
        handle profile-based pixel format and HDR stripping automatically."""
        real_name = self._real_name(encoder_name)
        return super().generate_filtergraphs(current_filter_args, smart_filters, real_name)

    def stream_args(self, stream_info, stream_id, encoder_name, filter_state=None):
        """
        Get base QSV stream args using the real encoder name, then post-process
        to apply the user-selected profile (if not auto).
        """
        real_name = self._real_name(encoder_name)

        # Get base args from parent QSV encoder
        result = super().stream_args(stream_info, stream_id, real_name, filter_state=filter_state)

        profile = self.settings.get_setting('qsv_gen11_profile') if self.settings else None
        if not profile or profile == 'auto':
            return result

        is_h264 = "h264" in real_name.lower()
        if is_h264:
            return result

        stream_args = result.get("stream_args", [])

        # Remove any existing -profile:v:N args set by parent (e.g. from HDR auto-detection)
        new_stream_args = []
        i = 0
        while i < len(stream_args):
            if i + 1 < len(stream_args) and stream_args[i].startswith('-profile:v:'):
                i += 2  # skip key and value
                continue
            new_stream_args.append(stream_args[i])
            i += 1

        # Add the user-selected profile for HEVC
        if real_name == "hevc_qsv":
            new_stream_args = [f'-profile:v:{stream_id}', profile] + new_stream_args

        # If forcing 8-bit, remove HDR color tags from stream args
        if profile == 'main':
            hdr_keys = {'-color_primaries', '-color_trc', '-colorspace', '-color_range'}
            filtered_args = []
            i = 0
            while i < len(new_stream_args):
                if i + 1 < len(new_stream_args) and new_stream_args[i] in hdr_keys:
                    i += 2  # skip key and value
                    continue
                filtered_args.append(new_stream_args[i])
                i += 1
            new_stream_args = filtered_args

        result["stream_args"] = new_stream_args
        return result

    # ── Form settings ────────────────────────────────────────────────────

    def get_qsv_gen11_profile_form_settings(self):
        """
        Profile selector for 8-bit/10-bit encoding.
        Visible for hevc_qsv_gen11 and av1_qsv_gen11 in standard mode only.
        """
        values = {
            "label":          "Profile",
            "description":    "Controls the bit depth of the encoded output.\n"
                              "Auto detects from source. Main forces 8-bit. Main10 forces 10-bit.\n"
                              "Use Main10 to encode SDR content in 10-bit for improved gradient quality,\n"
                              "or Main to force 8-bit output from 10-bit/HDR sources.",
            "sub_setting":    True,
            "input_type":     "select",
            "select_options": [],
        }

        encoder = self.settings.get_setting('video_encoder')

        if encoder == 'hevc_qsv_gen11':
            values["select_options"] = [
                {
                    "value": "auto",
                    "label": "Auto – Detect from source (recommended)",
                },
                {
                    "value": "main",
                    "label": "Main – Force 8-bit encoding",
                },
                {
                    "value": "main10",
                    "label": "Main10 – Force 10-bit encoding",
                },
            ]
        elif encoder == 'av1_qsv_gen11':
            values["select_options"] = [
                {
                    "value": "auto",
                    "label": "Auto – Detect from source (recommended)",
                },
                {
                    "value": "main",
                    "label": "8-bit encoding",
                },
                {
                    "value": "main10",
                    "label": "10-bit encoding",
                },
            ]
        else:
            values["display"] = "hidden"
            return values

        self._QsvGen11Encoder__set_default_option(values['select_options'], 'qsv_gen11_profile', 'auto')
        if self.settings.get_setting('mode') not in ['standard']:
            values["display"] = "hidden"
        return values

    def get_qsv_encoder_ratecontrol_method_form_settings(self):
        """
        Override parent to accept Gen11+ encoder names for LA_ICQ and LA options.
        Without this, those options would be hidden because the parent checks for
        'h264_qsv' and 'hevc_qsv' specifically.
        """
        values = {
            "label":          "Encoder ratecontrol method",
            "sub_setting":    True,
            "input_type":     "select",
            "select_options": []
        }
        values['select_options'] = [
            {
                "value": "CQP",
                "label": "CQP - Quality based mode using constant quantizer scale",
            },
            {
                "value": "ICQ",
                "label": "ICQ - Quality based mode using intelligent constant quality",
            }
        ]
        if self.settings.get_setting('video_encoder') in ['h264_qsv', 'h264_qsv_gen11']:
            values['select_options'] += [
                {
                    "value": "LA_ICQ",
                    "label": "LA_ICQ - Quality based mode using intelligent constant quality with lookahead",
                }
            ]
        values['select_options'] += [
            {
                "value": "VBR",
                "label": "VBR - Bitrate based mode using variable bitrate",
            },
        ]
        if self.settings.get_setting('video_encoder') in ['h264_qsv', 'hevc_qsv', 'h264_qsv_gen11', 'hevc_qsv_gen11']:
            values['select_options'] += [
                {
                    "value": "LA",
                    "label": "LA - Bitrate based mode using VBR with lookahead",
                }
            ]
        values['select_options'] += [
            {
                "value": "CBR",
                "label": "CBR - Bitrate based mode using constant bitrate",
            }
        ]
        self._QsvGen11Encoder__set_default_option(values['select_options'], 'qsv_encoder_ratecontrol_method', default_option='LA')
        if self.settings.get_setting('mode') not in ['standard']:
            values["display"] = "hidden"
        return values

    def __set_default_option(self, select_options, key, default_option=None):
        """
        Sets the default option if the currently set option is not available.
        Copied from parent to avoid name-mangling issues with private methods.
        """
        available_options = []
        for option in select_options:
            available_options.append(option.get('value'))
            if not default_option:
                default_option = option.get('value')
        current_value = self.settings.get_setting(key)
        if not getattr(self.settings, 'apply_default_fallbacks', True):
            return current_value
        if current_value not in available_options:
            self.settings.settings_configured[key] = default_option
            return default_option
        return current_value
