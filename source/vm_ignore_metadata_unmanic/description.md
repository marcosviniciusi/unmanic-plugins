
---

### What it does

**Completely ignores** files that have already completed the full processing pipeline by checking for the `UNMANIC_FULL_PIPELINE=processed` metadata tag.

This plugin runs with **priority 0** (runs last among file test plugins — in Unmanic, lower priority = runs later), meaning it has the **final word** — if the tag exists, no other plugin can re-add the file to the processing queue.

### How it works

1. During library scan, every file goes through all file test plugins in priority order (higher priority first)
2. Other plugins (priority 10-100) may set `add_file_to_pending_tasks = True` if they find work to do
3. This plugin runs **last** (priority 0) and checks for the `UNMANIC_FULL_PIPELINE=processed` tag
4. If the tag exists, it sets `add_file_to_pending_tasks = False` — **overriding all previous plugins**

### Works with

This plugin works together with **Tag Pipeline Complete** (`vm_tag_pipeline_complete`), which writes the `UNMANIC_FULL_PIPELINE=processed` tag as the last processing step. Once a file passes through the entire pipeline, it gets tagged and will never re-enter the queue.

---
