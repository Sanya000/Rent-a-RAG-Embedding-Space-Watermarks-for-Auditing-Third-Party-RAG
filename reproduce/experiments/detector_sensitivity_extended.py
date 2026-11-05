#!/usr/bin/env python3
"""Extended (tau_sim x tau_margin) sensitivity grid: 324 combos at W=28.

Kept for command compatibility with the paper run; equivalent to
``detector_sensitivity_sweep.py --grid extended`` and writes
``<out_dir>/sensitivity_extended.json``. The precompute and sweep code is the
one in ``detector_sensitivity_sweep.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detector_sensitivity_sweep import main  # noqa: E402

if __name__ == "__main__":
    argv = sys.argv[1:]
    if not any(a == "--grid" or a.startswith("--grid=") for a in argv):
        argv = ["--grid", "extended", *argv]
    main(argv)
