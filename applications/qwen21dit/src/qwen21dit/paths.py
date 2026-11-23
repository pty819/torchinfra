"""Local paths shared by the worker and offline tools."""
import os
from pathlib import Path

MODEL_DIR = Path(os.environ.get("QDIT_MODEL_DIR", "~/models/qwen21")).expanduser()
OUTDIR = Path(os.environ.get("QDIT_OUTDIR", "~/Pictures/qwen")).expanduser()
