Send completed task **logs** to an **OpenTelemetry-compatible backend** (SigNoz, Jaeger, Grafana Tempo, etc.) via **OTLP HTTP**.

Each completed Unmanic task generates a structured JSON log containing:

- **`unmanic_processed`**: `success` or `failed`
- **Task info**: basename, id, library_id, duration, start/finish timestamps
- **Source file**: path, size (bytes + human readable)
- **Destination files**: paths, count, total size
- **Environment**: service_name, environment, hostname

### Configuration

| Setting | Description | Default |
|---|---|---|
| OTEL Collector Host | Hostname or IP | `localhost` |
| OTEL Collector Port | Port number | `4318` |
| Protocol | HTTP or HTTPS | `http` |
| OTLP Log Endpoint Path | Path for log endpoint | `/v1/logs` |
| OTLP Headers | Auth headers (key=value) | *(empty)* |
| Service Name | service.name attribute | `unmanic` |
| Environment | deployment.environment | `production` |
| Hostname | Override auto-detected hostname | *(auto)* |
| Send on failure | Log failed tasks too | `true` |

### SigNoz Self-Hosted

- Host: `signoz-otel-collector` (or your host IP)
- Port: `4318`
- Protocol: `HTTP`
- Headers: *(empty)*

### SigNoz Cloud

- Host: `ingest.us.signoz.cloud`
- Port: `443`
- Protocol: `HTTPS`
- Headers: `signoz-ingestion-key=<your-key>`

### Limitations

The Unmanic post-processor hook provides only the **overall task status** (success/failed). Individual per-plugin statuses (which plugins ran, which were skipped) are **not available** from the Unmanic API at the post-processor stage.
