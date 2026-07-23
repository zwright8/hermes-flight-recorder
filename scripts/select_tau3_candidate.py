#!/usr/bin/env python3
"""Select and lock one governed Tau-3 development candidate."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flightrecorder.tau3_candidate_selection import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
