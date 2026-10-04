"""Output locations. Override with the environment variables ETB_FIGDIR and ETB_RESDIR
(e.g. to reproduce the results without overwriting the archived ones)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIGDIR = os.environ.get("ETB_FIGDIR", str(ROOT / "paper" / "figs")).rstrip("/") + "/"
RESDIR = os.environ.get("ETB_RESDIR", str(ROOT / "results")).rstrip("/") + "/"
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(RESDIR, exist_ok=True)
