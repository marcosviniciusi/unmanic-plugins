**<span style="color:#56adda">0.2.0</span>** *(marcosviniciusi)*
- CHANGED: Send structured OTEL logs instead of traces (better for SigNoz log pipeline)
- NEW: Separate host, port, and protocol (HTTP/HTTPS) settings
- NEW: Structured JSON log body with unmanic_processed status (success/failed)
- NEW: Task details: id, library_id, duration, start/finish timestamps
- NEW: Source file details: path, basename, size (bytes + human readable)
- NEW: Destination file details: paths, count, total size
- NEW: Environment metadata: service_name, environment, hostname
- NEW: Configurable OTLP log endpoint path

**<span style="color:#56adda">0.1.0</span>** *(marcosviniciusi)*
- Initial release
- Send task completion traces to OTLP HTTP backend (SigNoz, Jaeger, Grafana Tempo)
- Configurable endpoint, service name, headers, and TLS settings
- Captures task duration, source/destination file info, and processing status
- Option to include or exclude failed task traces
