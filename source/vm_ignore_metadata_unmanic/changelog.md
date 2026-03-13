**<span style="color:#56adda">0.0.3</span>** *(marcosviniciusi)*
- Changed to check `UNMANIC_FULL_PIPELINE=processed` tag (full pipeline completion)
- Changed priority to 0 (runs LAST — lower number = runs later in Unmanic) so it overrides all other plugins' decisions
- If tag exists, file is completely ignored — no other plugin can re-add it
- Renamed to "Ignore - Full Pipeline Completed"

**<span style="color:#56adda">0.0.2</span>** *(marcosviniciusi)*
- Standardized plugin name to "Ignore - Metadata Processed" for consistent naming
- Updated description with metadata tag details

**<span style="color:#56adda">0.0.1</span>**
- Initial version (marcosviniciusi)
