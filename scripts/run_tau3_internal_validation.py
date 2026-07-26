#!/usr/bin/env python3
"""Run complete internal-validation loss replay for one Tau-3 MLX adapter."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flightrecorder.mlx_internal_validation import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
