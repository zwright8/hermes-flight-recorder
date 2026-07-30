from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from flightrecorder.schema_registry import (
    check_schema_file,
    list_schema_records,
)
from flightrecorder.tau3_development_screening import (
    Tau3DevelopmentScreeningError,
    build_tau3_development_screening,
    selected_tasks_by_domain,
    validate_tau3_development_screening,
)

ROOT = Path(__file__).resolve().parents[1]


class Tau3DevelopmentScreeningTests(unittest.TestCase):
    def test_builds_and_replays_one_task_per_domain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "development.json"
            out = root / "screening.json"
            _write_source(source)

            payload = build_tau3_development_screening(
                development_source=source,
                out_path=out,
                created_at="2026-07-30T00:00:00Z",
            )
            validation = validate_tau3_development_screening(
                screening=out,
                development_source=source,
            )

            self.assertTrue(validation["passed"], validation)
            self.assertFalse(payload["candidate_eligible"])
            self.assertTrue(payload["qualification_requires_full_development"])
            self.assertEqual(payload["task_count"], 3)
            self.assertEqual(payload["expected_run_count"], 12)
            self.assertEqual(
                [task["domain"] for task in payload["selected_tasks"]],
                ["airline", "retail", "telecom"],
            )
            self.assertEqual(
                set(
                    selected_tasks_by_domain(
                        screening=out,
                        development_source=source,
                    )
                ),
                {"airline", "retail", "telecom"},
            )
            self.assertTrue(
                check_schema_file(
                    out,
                    "tau3_development_screening",
                )["passed"]
            )

    def test_rejects_substituted_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "development.json"
            out = root / "screening.json"
            _write_source(source)
            build_tau3_development_screening(
                development_source=source,
                out_path=out,
                created_at="2026-07-30T00:00:00Z",
            )
            payload = _read(out)
            payload["selected_tasks"][0]["raw_id"] = "substituted"
            _write(out, payload)

            validation = validate_tau3_development_screening(
                screening=out,
                development_source=source,
            )

            self.assertFalse(validation["passed"])
            with self.assertRaises(Tau3DevelopmentScreeningError):
                selected_tasks_by_domain(
                    screening=out,
                    development_source=source,
                )

    def test_create_once_and_cli_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "development.json"
            out = root / "screening.json"
            _write_source(source)
            command = [
                sys.executable,
                str(ROOT / "scripts" / "build_tau3_development_screening.py"),
                "build",
                "--development-source",
                str(source),
                "--out",
                str(out),
                "--created-at",
                "2026-07-30T00:00:00Z",
            ]
            completed = subprocess.run(
                command,
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(
                completed.returncode,
                0,
                completed.stderr + completed.stdout,
            )
            with self.assertRaises(Tau3DevelopmentScreeningError):
                build_tau3_development_screening(
                    development_source=source,
                    out_path=out,
                )
            validate = subprocess.run(
                [
                    sys.executable,
                    str(
                        ROOT
                        / "scripts"
                        / "build_tau3_development_screening.py"
                    ),
                    "validate",
                    "--screening",
                    str(out),
                    "--development-source",
                    str(source),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(
                validate.returncode,
                0,
                validate.stderr + validate.stdout,
            )

    def test_schema_is_registered(self) -> None:
        names = {record["name"] for record in list_schema_records()}
        self.assertIn("tau3_development_screening", names)


def _write_source(path: Path) -> None:
    tasks = []
    for domain in ("airline", "retail", "telecom"):
        for ordinal in range(2):
            raw_id = f"{domain}-{ordinal}"
            tasks.append(
                {
                    "domain": domain,
                    "raw_id": raw_id,
                    "raw_id_sha256": _hash(f"{domain}:{raw_id}"),
                    "task_sha256": _hash(f"task:{raw_id}"),
                    "prompt_sha256": _hash(f"prompt:{raw_id}"),
                    "family_id": _hash(f"family:{raw_id}"),
                }
            )
    payload = {
        "schema_version": "hfr.tau3_source_split.v1",
        "source_revision": "1" * 40,
        "split": "development",
        "task_schema_version": "tau2.tasks.v1",
        "algorithm": "test-family-split",
        "salt_sha256": "2" * 64,
        "task_count": len(tasks),
        "family_count": len(tasks),
        "family_ids": [task["family_id"] for task in tasks],
        "tasks": tasks,
    }
    _write(path, payload)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
