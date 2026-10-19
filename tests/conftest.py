"""Put the package and the gold-train step's src/ on sys.path, as run-gold.sh does via PYTHONPATH.

These tests are deliberately torch-free: they import only modules that need PyYAML, so they run
without the GPU stack (see schedules.py and on_policy_draw.py for why that split exists).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT / "src", ROOT / "steps" / "distill-gold-train" / "src"):
    sys.path.insert(0, str(p))
