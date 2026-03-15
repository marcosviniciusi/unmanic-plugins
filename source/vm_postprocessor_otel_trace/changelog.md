**<span style="color:#56adda">0.4.0</span>** *(marcosviniciusi)*
- FIX: Source file size now correctly reports the original size before processing (was showing same size as destination)
- NEW: Added on_worker_process hook to capture original source file size before transcoding replaces the file
- Uses temp cache file (MD5-keyed) to pass source size from worker to post-processor stage

**<span style="color:#56adda">0.3.4</span>** *(marcosviniciusi)*
- FIX: Updated service.version and scope_version from hardcoded 0.3.0 to 0.3.4

**<span style="color:#56adda">0.3.3</span>** *(marcosviniciusi)*
- FIX: Replaced BatchLogRecordProcessor with SimpleLogRecordProcessor for synchronous export (logs were not reaching the backend)

**<span style="color:#56adda">0.3.2</span>** *(marcosviniciusi)*
- FIX: Removed unsupported 'resource' parameter from LogRecord constructor (resource is set at LoggerProvider level)

**<span style="color:#56adda">0.3.1</span>** *(marcosviniciusi)*
- FIX: Fixed "'Logger' object has no attribute '_logger'" error by using LogRecord directly instead of internal API

**<span style="color:#56adda">0.3.0</span>** *(marcosviniciusi)*
- CHANGED: Moved basename from source to task object for better search/filtering
- NEW: Configurable hostname setting (override auto-detected hostname)
- UPDATED: Full structured JSON with task, source, destination, environment objects

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
