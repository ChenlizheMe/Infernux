"""Make every Hub test runnable alone, without collection-order side effects."""

from pathlib import Path
import sys


PACKAGING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGING_ROOT))
sys.path.insert(0, str(PACKAGING_ROOT.parent / "python"))
