# postprocessor_otel_trace

Post-Processor plugin for Unmanic that sends structured task logs to an OpenTelemetry-compatible backend via OTLP HTTP.

## How It Works

### Hook Used

This plugin implements the `on_postprocessor_task_results` hook, which is called by Unmanic **after every task completes** (both success and failure).

### Data Flow

```
Unmanic Task Completes
        │
        ▼
on_postprocessor_task_results(data)
        │
        ├── Build structured JSON log from task data
        │
        ├── Create OTEL LogRecord with:
        │     - body: JSON string with full task details
        │     - severity: INFO (success) or ERROR (failure)
        │     - attributes: key fields for filtering in SigNoz
        │     - resource: service.name, environment, hostname
        │
        └── Export via OTLP HTTP to {protocol}://{host}:{port}{path}
```

### What Unmanic Provides (data dict)

| Field | Type | Description |
|---|---|---|
| `task_processing_success` | bool | Overall task success |
| `file_move_processes_success` | bool | File movements success |
| `destination_files` | list | Output file paths |
| `source_data` | dict | Source file info (`abspath`, `basename`, etc.) |
| `start_time` | float | UNIX timestamp task start |
| `finish_time` | float | UNIX timestamp task end |
| `task_id` | int | Unique task ID |
| `library_id` | int | Library ID |

### JSON Log Structure

```json
{
    "unmanic_processed": "success",

    "task": {
        "id": 42,
        "basename": "Movie.mkv",
        "library_id": 1,
        "processing_success": true,
        "file_move_success": true,
        "duration_seconds": 127.45,
        "duration_human": "2m 7s",
        "start_time": "2026-03-12T14:30:00",
        "finish_time": "2026-03-12T14:32:07"
    },

    "source": {
        "path": "/media/movies/Movie.mkv",
        "size_bytes": 4294967296,
        "size_human": "4.00 GB"
    },

    "destination": {
        "files": ["/media/movies/Movie.mkv"],
        "count": 1,
        "total_size_bytes": 2147483648,
        "total_size_human": "2.00 GB"
    },

    "environment": {
        "service_name": "unmanic",
        "environment": "production",
        "hostname": "media-server"
    }
}
```

### Failed Task Example

```json
{
    "unmanic_processed": "failed",

    "task": {
        "id": 43,
        "basename": "Corrupted.avi",
        "library_id": 1,
        "processing_success": false,
        "file_move_success": false,
        "duration_seconds": 5.32,
        "duration_human": "5s",
        "start_time": "2026-03-12T14:35:00",
        "finish_time": "2026-03-12T14:35:05"
    },

    "source": {
        "path": "/media/movies/Corrupted.avi",
        "size_bytes": 734003200,
        "size_human": "700.00 MB"
    },

    "destination": {
        "files": [],
        "count": 0,
        "total_size_bytes": 0,
        "total_size_human": "0 B"
    },

    "environment": {
        "service_name": "unmanic",
        "environment": "production",
        "hostname": "media-server"
    }
}
```

### OTEL LogRecord Attributes (for filtering in SigNoz)

| Attribute | Example | Description |
|---|---|---|
| `unmanic.processed` | `success` / `failed` | Final task status |
| `unmanic.task.id` | `42` | Task ID |
| `unmanic.task.basename` | `Movie.mkv` | File name |
| `unmanic.source.path` | `/media/movies/Movie.mkv` | Full source path |
| `unmanic.task.duration_s` | `127.45` | Duration in seconds |
| `unmanic.destination.count` | `1` | Number of output files |
| `log.type` | `unmanic_task_result` | Fixed identifier |

## Configuration

### SigNoz Self-Hosted (docker-compose)

| Setting | Value |
|---|---|
| Host | `signoz-otel-collector` (or container IP) |
| Port | `4318` |
| Protocol | HTTP |
| Headers | *(empty)* |

### SigNoz Cloud

| Setting | Value |
|---|---|
| Host | `ingest.us.signoz.cloud` |
| Port | `443` |
| Protocol | HTTPS |
| Headers | `signoz-ingestion-key=YOUR_KEY` |

## Dependencies

Installed automatically by Unmanic from `requirements.txt`:

- `opentelemetry-api >= 1.20.0`
- `opentelemetry-sdk >= 1.20.0`
- `opentelemetry-exporter-otlp-proto-http >= 1.20.0`

## Limitations

The Unmanic `on_postprocessor_task_results` hook provides only the **overall task result** (success/failed). It does **NOT** provide:

- Individual plugin execution statuses (success/skipped/failed per plugin)
- Which plugins were executed in the pipeline
- Per-plugin logs or error messages

This is a limitation of the Unmanic plugin API, not this plugin. The Unmanic core does not expose per-plugin results to the post-processor stage.

## File Structure

```
postprocessor_otel_trace/
├── info.json           # Plugin metadata (id, version, priorities)
├── plugin.py           # Main plugin code
├── requirements.txt    # Python dependencies (OTEL SDK)
├── description.md      # Plugin description (shown in Unmanic UI)
├── changelog.md        # Version history
├── README.md           # This file
├── LICENSE             # GPL v3
└── .gitignore
```
