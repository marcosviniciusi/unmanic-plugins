#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unmanic Post-Processor Plugin: OpenTelemetry Task Trace

Sends completed task telemetry (traces/spans) to an OpenTelemetry-compatible
backend (SigNoz, Jaeger, Grafana Tempo, etc.) via OTLP HTTP.

Each completed task generates a span with:
  - Task duration (start_time → finish_time)
  - Source file path and metadata
  - Destination file paths
  - Processing success/failure status
"""

import logging
import os
import time

from unmanic.libs.unplugins.settings import PluginSettings

logger = logging.getLogger("Unmanic.Plugin.postprocessor_otel_trace")


class Settings(PluginSettings):
    settings = {
        'otel_endpoint':     'http://localhost:4318',
        'otel_service_name': 'unmanic',
        'otel_headers':      '',
        'otel_insecure':     True,
        'send_on_failure':   True,
    }

    form_settings = {
        'otel_endpoint': {
            'label':       'OTLP HTTP Endpoint',
            'description': 'The OTLP HTTP collector endpoint (e.g. http://signoz:4318 or https://ingest.us.signoz.cloud:443).',
            'input_type':  'text',
        },
        'otel_service_name': {
            'label':       'Service Name',
            'description': 'The service.name resource attribute sent with every span.',
            'input_type':  'text',
        },
        'otel_headers': {
            'label':       'OTLP Headers (optional)',
            'description': 'Comma-separated key=value pairs (e.g. signoz-ingestion-key=abc123). Leave empty for self-hosted.',
            'input_type':  'text',
        },
        'otel_insecure': {
            'label':       'Allow insecure (HTTP) connections',
            'description': 'Disable TLS verification for self-hosted backends on HTTP.',
            'input_type':  'checkbox',
        },
        'send_on_failure': {
            'label':       'Send trace on failed tasks too',
            'description': 'If unchecked, only successful tasks generate a trace span.',
            'input_type':  'checkbox',
        },
    }


def _parse_headers(header_string):
    """Parse comma-separated key=value header pairs into a dict."""
    headers = {}
    if not header_string or not header_string.strip():
        return headers
    for pair in header_string.split(','):
        pair = pair.strip()
        if '=' in pair:
            key, value = pair.split('=', 1)
            headers[key.strip()] = value.strip()
    return headers


def _get_file_size(path):
    """Get file size in bytes, return 0 if not accessible."""
    try:
        return os.path.getsize(path)
    except (OSError, TypeError):
        return 0


def _send_trace(settings, data):
    """Build and export an OTEL span for the completed task."""
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.trace import StatusCode
    except ImportError as e:
        logger.error(
            "OpenTelemetry packages not installed. "
            "Ensure requirements.txt includes the needed packages. Error: %s", e
        )
        return

    endpoint = settings.get_setting('otel_endpoint')
    service_name = settings.get_setting('otel_service_name')
    header_string = settings.get_setting('otel_headers')

    headers = _parse_headers(header_string)

    traces_endpoint = endpoint.rstrip('/') + '/v1/traces'

    resource = Resource.create({
        'service.name': service_name,
        'service.version': '0.1.0',
        'deployment.environment': 'production',
    })

    exporter = OTLPSpanExporter(
        endpoint=traces_endpoint,
        headers=headers,
    )

    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(exporter))

    tracer = provider.get_tracer('unmanic.postprocessor_otel_trace', '0.1.0')

    task_success = data.get('task_processing_success', False)
    file_move_success = data.get('file_move_processes_success', False)
    destination_files = data.get('destination_files', [])
    source_data = data.get('source_data', {})
    start_time = data.get('start_time', 0)
    finish_time = data.get('finish_time', 0)

    source_path = source_data.get('abspath', source_data.get('basename', 'unknown'))
    source_basename = os.path.basename(source_path) if source_path else 'unknown'

    span_name = "unmanic.task.{}".format('success' if task_success else 'failure')

    start_ns = int(start_time * 1e9) if start_time else None
    end_ns = int(finish_time * 1e9) if finish_time else None

    with tracer.start_span(
        name=span_name,
        start_time=start_ns,
        kind=trace.SpanKind.INTERNAL,
    ) as span:
        span.set_attribute('unmanic.task.success', task_success)
        span.set_attribute('unmanic.task.file_move_success', file_move_success)
        span.set_attribute('unmanic.source.path', str(source_path))
        span.set_attribute('unmanic.source.basename', source_basename)
        span.set_attribute('unmanic.source.size_bytes', _get_file_size(source_path))

        if destination_files:
            span.set_attribute('unmanic.destination.files', destination_files)
            span.set_attribute('unmanic.destination.count', len(destination_files))
            total_dest_size = sum(_get_file_size(f) for f in destination_files)
            span.set_attribute('unmanic.destination.total_size_bytes', total_dest_size)

        if start_time and finish_time:
            duration_s = finish_time - start_time
            span.set_attribute('unmanic.task.duration_seconds', round(duration_s, 2))

        for key, value in source_data.items():
            if isinstance(value, (str, int, float, bool)):
                span.set_attribute('unmanic.source.{}'.format(key), value)

        if task_success:
            span.set_status(trace.Status(StatusCode.OK))
        else:
            span.set_status(trace.Status(StatusCode.ERROR, 'Task processing failed'))

        if end_ns:
            span.end(end_time=end_ns)

    provider.force_flush()
    provider.shutdown()

    logger.info(
        "OTEL trace sent for '%s' (success=%s, duration=%.1fs) to %s",
        source_basename,
        task_success,
        (finish_time - start_time) if (start_time and finish_time) else 0,
        endpoint,
    )


def on_postprocessor_task_results(data):
    """
    Runner function - provides the results of the postprocessor task processing.

    The 'data' object argument includes:
        task_processing_success     - Boolean, did all task processes complete successfully.
        file_move_processes_success - Boolean, did all postprocessor movement tasks complete successfully.
        destination_files           - List containing all file paths created by postprocessor file movements.
        source_data                 - Dictionary containing data pertaining to the original source file.
        start_time                  - Float, UNIX timestamp when the task began.
        finish_time                 - Float, UNIX timestamp when the task completed.

    :param data:
    :return:
    """
    settings = Settings(library_id=data.get('library_id'))

    task_success = data.get('task_processing_success', False)
    send_on_failure = settings.get_setting('send_on_failure')

    if not task_success and not send_on_failure:
        logger.debug("Task failed and send_on_failure is disabled. Skipping trace.")
        return

    try:
        _send_trace(settings, data)
    except Exception as e:
        logger.error("Failed to send OTEL trace: %s", e)
