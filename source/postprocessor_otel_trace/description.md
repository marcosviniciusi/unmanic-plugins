Send completed task telemetry to an **OpenTelemetry-compatible backend** (SigNoz, Jaeger, Grafana Tempo, etc.) via **OTLP HTTP**.

Each completed Unmanic task generates a trace span containing:

- **Task duration** (start → finish timestamps)
- **Source file** path, basename, and size
- **Destination files** paths, count, and total size
- **Processing status** (success / failure)
- **Source metadata** (all available key-value pairs)

### Configuration

| Setting | Description | Default |
|---|---|---|
| OTLP HTTP Endpoint | Collector endpoint | `http://localhost:4318` |
| Service Name | OTEL `service.name` attribute | `unmanic` |
| OTLP Headers | Comma-separated `key=value` pairs for auth | *(empty)* |
| Allow insecure | Disable TLS verification | `true` |
| Send on failure | Also trace failed tasks | `true` |

### SigNoz Self-Hosted Example

Set the endpoint to `http://<signoz-host>:4318` and leave headers empty.

### SigNoz Cloud Example

Set the endpoint to `https://ingest.us.signoz.cloud:443` and headers to `signoz-ingestion-key=<your-key>`.
