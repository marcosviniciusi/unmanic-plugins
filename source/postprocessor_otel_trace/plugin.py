#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unmanic Post-Processor Plugin: OpenTelemetry Task Log

Sends a structured log record for each completed Unmanic task to an
OpenTelemetry-compatible backend (SigNoz, Jaeger, Grafana Tempo, etc.)
via OTLP HTTP.

The log contains:
  - Task result (unmanic_processed: success / failed)
  - Full file path
  - Task duration
  - Environment metadata (host, service name)

NOTE: The Unmanic post-processor hook (on_postprocessor_task_results) provides
overall task success/failure status. Individual per-plugin statuses
(success/skipped/failed) are NOT available from this hook — the Unmanic API
does not expose them at the post-processor stage.
"""

import json
import logging
import os

from unmanic.libs.unplugins.settings import PluginSettings

logger = logging.getLogger("Unmanic.Plugin.postprocessor_otel_trace")


class Settings(PluginSettings):
    settings = {
        'otel_host':         'localhost',
        'otel_port':         '4318',
        'otel_protocol':     'http',
        'otel_path':         '/v1/logs',
        'otel_headers':      '',
        'otel_service_name': 'unmanic',
        'otel_environment':  'production',
        'otel_hostname':     '',
        'send_on_failure':   True,
    }

    form_settings = {
        'otel_host': {
            'label':       'OTEL Collector Host',
            'description': 'Hostname or IP of the OTEL collector (e.g. signoz-otel-collector, 192.168.1.100, ingest.us.signoz.cloud).',
            'input_type':  'text',
        },
        'otel_port': {
            'label':       'OTEL Collector Port',
            'description': 'Port of the OTEL HTTP receiver (default: 4318).',
            'input_type':  'text',
        },
        'otel_protocol': {
            'label':       'Protocol',
            'description': 'Use https for SigNoz Cloud or TLS-enabled collectors.',
            'input_type':  'select',
            'select_options': [
                {'value': 'http',  'label': 'HTTP'},
                {'value': 'https', 'label': 'HTTPS'},
            ],
        },
        'otel_path': {
            'label':       'OTLP Log Endpoint Path',
            'description': 'Path for the OTLP log endpoint (default: /v1/logs).',
            'input_type':  'text',
        },
        'otel_headers': {
            'label':       'OTLP Headers (optional)',
            'description': 'Comma-separated key=value pairs for authentication (e.g. signoz-ingestion-key=abc123). Leave empty for self-hosted.',
            'input_type':  'text',
        },
        'otel_service_name': {
            'label':       'Service Name',
            'description': 'The service.name sent with every log record (identifies this Unmanic instance).',
            'input_type':  'text',
        },
        'otel_environment': {
            'label':       'Environment',
            'description': 'Deployment environment label (e.g. production, staging, homelab).',
            'input_type':  'text',
        },
        'otel_hostname': {
            'label':       'Hostname',
            'description': 'Override the hostname sent with logs. Leave empty to auto-detect from system.',
            'input_type':  'text',
        },
        'send_on_failure': {
            'label':       'Send log on failed tasks too',
            'description': 'If unchecked, only successful tasks generate a log entry.',
            'input_type':  'checkbox',
        },
    }


def _build_endpoint(settings):
    """Build the full OTLP endpoint URL from host/port/protocol settings."""
    protocol = settings.get_setting('otel_protocol')
    host = settings.get_setting('otel_host').strip().rstrip('/')
    port = settings.get_setting('otel_port').strip()
    path = settings.get_setting('otel_path').strip()

    if not path.startswith('/'):
        path = '/' + path

    if port:
        return "{}://{}:{}{}".format(protocol, host, port, path)
    return "{}://{}{}".format(protocol, host, path)


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


def _format_duration(seconds):
    """Format seconds to human-readable duration."""
    if seconds <= 0:
        return "0s"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    parts = []
    if hours:
        parts.append("{}h".format(hours))
    if minutes:
        parts.append("{}m".format(minutes))
    parts.append("{}s".format(secs))
    return " ".join(parts)


def _build_task_log(settings, data):
    """
    Build a structured log record dict from the task data.

    This is the JSON body that will be sent to the OTEL collector.
    """
    task_success = data.get('task_processing_success', False)
    source_data = data.get('source_data', {})
    start_time = data.get('start_time', 0)
    finish_time = data.get('finish_time', 0)

    source_path = source_data.get('abspath', source_data.get('basename', 'unknown'))
    source_basename = os.path.basename(source_path) if source_path else 'unknown'
    duration_s = (finish_time - start_time) if (start_time and finish_time) else 0
    unmanic_status = "success" if task_success else "failed"

    service_name = settings.get_setting('otel_service_name')
    environment = settings.get_setting('otel_environment')
    configured_hostname = settings.get_setting('otel_hostname')
    hostname = configured_hostname.strip() if configured_hostname and configured_hostname.strip() else \
        os.environ.get('HOSTNAME', os.environ.get('COMPUTERNAME', 'unknown'))

    log_record = {
        "unmanic_processed":  unmanic_status,
        "task": {
            "file":             str(source_path),
            "basename":         source_basename,
            "duration":         _format_duration(duration_s),
            "duration_seconds": round(duration_s, 2),
        },
        "hostname":           hostname,
        "service":            service_name,
        "environment":        environment,
    }

    return log_record


def _send_log(settings, data):
    """Send the task log to the OTEL collector via OTLP HTTP."""
    try:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk._logs import LoggerProvider
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
        from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
        from opentelemetry._logs import set_logger_provider, SeverityNumber
    except ImportError as e:
        logger.error(
            "OpenTelemetry packages not installed. "
            "Ensure requirements.txt includes the needed packages. Error: %s", e
        )
        return

    endpoint = _build_endpoint(settings)
    header_string = settings.get_setting('otel_headers')
    service_name = settings.get_setting('otel_service_name')
    environment = settings.get_setting('otel_environment')
    headers = _parse_headers(header_string)

    configured_hostname = settings.get_setting('otel_hostname')
    hostname = configured_hostname.strip() if configured_hostname and configured_hostname.strip() else \
        os.environ.get('HOSTNAME', os.environ.get('COMPUTERNAME', 'unknown'))

    resource = Resource.create({
        'service.name':             service_name,
        'service.version':          '0.3.0',
        'deployment.environment':   environment,
        'host.name':                hostname,
    })

    exporter = OTLPLogExporter(
        endpoint=endpoint,
        headers=headers,
    )

    log_provider = LoggerProvider(resource=resource)
    log_provider.add_log_record_processor(BatchLogRecordProcessor(exporter))
    set_logger_provider(log_provider)

    otel_logger = log_provider.get_logger('unmanic.postprocessor_otel_trace', '0.3.0')

    task_log = _build_task_log(settings, data)
    task_success = data.get('task_processing_success', False)

    severity = SeverityNumber.INFO if task_success else SeverityNumber.ERROR
    severity_text = "INFO" if task_success else "ERROR"

    log_body = json.dumps(task_log, ensure_ascii=False, default=str)

    otel_logger.emit(
        otel_logger._logger.create_log_record(
            body=log_body,
            severity_number=severity,
            severity_text=severity_text,
            attributes={
                'unmanic.processed':        task_log['unmanic_processed'],
                'unmanic.task.file':        task_log['task']['file'],
                'unmanic.task.basename':    task_log['task']['basename'],
                'unmanic.task.duration_s':  task_log['task']['duration_seconds'],
                'log.type':                 'unmanic_task_result',
            },
        )
    )

    log_provider.force_flush()
    log_provider.shutdown()

    logger.info(
        "OTEL log sent for '%s' (unmanic_processed=%s, duration=%s) to %s",
        task_log['task']['basename'],
        task_log['unmanic_processed'],
        task_log['task']['duration'],
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
        logger.debug("Task failed and send_on_failure is disabled. Skipping log.")
        return

    try:
        _send_log(settings, data)
    except Exception as e:
        logger.error("Failed to send OTEL log: %s", e)
