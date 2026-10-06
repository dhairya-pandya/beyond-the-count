"""Cell Painting Shortcut Audit (cpsa)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
EXTERNAL = DATA / "external"
PROCESSED = DATA / "processed"
RESULTS = ROOT / "results"

__version__ = "0.1.0"
