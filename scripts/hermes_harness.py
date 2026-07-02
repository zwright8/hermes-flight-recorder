#!/usr/bin/env python3
"""Run the offline Flight Recorder harness facade."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flightrecorder.harness import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
