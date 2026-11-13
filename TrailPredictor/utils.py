"""Project-wide data directory paths."""

from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
COMPUTED_DIR = DATA_DIR / "computed"
