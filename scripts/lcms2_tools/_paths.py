"""Repository paths used by the project commands."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LCMS = ROOT / "vendor" / "Little-CMS"
LCMS_HEADER = LCMS / "include" / "lcms2.h"
EXT = ROOT / "lcms2" / "_ext"
