"""Standalone download worker — called via trickle for bandwidth limiting."""

import json
import os
import sqlite3
import sys

from huggingface_hub import snapshot_download

model_id = sys.argv[1]
db_path = sys.argv[2]
token = os.environ.get("HF_TOKEN") or None


def update_status(status: str, progress: float = 0.0):
    conn = sqlite3.connect(db_path)
    conn.execute(
        "UPDATE models SET status = ?, download_progress = ? WHERE model_id = ?",
        (status, progress, model_id),
    )
    conn.commit()
    conn.close()
    # Write progress to stdout so parent can read it
    print(json.dumps({"status": status, "progress": progress}), flush=True)


try:
    update_status("downloading", 0.0)
    snapshot_download(model_id, token=token)
    update_status("downloaded", 100.0)
except Exception as e:
    print(json.dumps({"status": "error", "error": str(e)}), flush=True)
    update_status("not_downloaded", 0.0)
    sys.exit(1)
