from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from flightrecorder.tau3_competitive_v3 import (
    DATASET_SCHEMA_VERSION,
    PLAN_FILENAME,
    PLAN_SCHEMA_VERSION,
    Tau3CompetitiveV3BindingError,
    bind_tau3_competitive_v3_evidence,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "bind_tau3_competitive_v3_evidence.py"


class Tau3CompetitiveV3EvidenceBindingTests(unittest.TestCase):
    def test_binds_content_addressed_stage_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "bundle"
            plan_path, evidence_path = write_fixture(bundle)

            first = bind_tau3_competitive_v3_evidence(
                bundle,
                stage="dataset",
                evidence_path=evidence_path,
            )
            plan_mode = plan_path.stat().st_mode & 0o777
            second = bind_tau3_competitive_v3_evidence(
                bundle,
                stage="dataset",
                evidence_path=evidence_path,
            )
            plan = read_json(plan_path)

            self.assertTrue(first["passed"])
            self.assertTrue(first["changed"])
            self.assertFalse(second["changed"])
            self.assertEqual(
                plan["evidence_refs"]["dataset"],
                {
                    "path": "dataset-evidence.json",
                    "sha256": sha256_file(evidence_path),
                },
            )
            self.assertEqual(first["plan_sha256"], second["plan_sha256"])
            self.assertEqual(plan_mode, 0o644)

    def test_refuses_to_replace_an_existing_stage_binding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "bundle"
            _, evidence_path = write_fixture(bundle)
            bind_tau3_competitive_v3_evidence(
                bundle,
                stage="dataset",
                evidence_path=evidence_path,
            )
            evidence_path.write_text(
                json.dumps(
                    {
                        "schema_version": DATASET_SCHEMA_VERSION,
                        "changed": True,
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                Tau3CompetitiveV3BindingError,
                "refusing to replace immutable evidence_refs.dataset",
            ):
                bind_tau3_competitive_v3_evidence(
                    bundle,
                    stage="dataset",
                    evidence_path=evidence_path,
                )

    def test_refuses_artifact_outside_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = root / "bundle"
            plan_path, _ = write_fixture(bundle)
            outside = root / "outside.json"
            write_json(
                outside,
                {"schema_version": DATASET_SCHEMA_VERSION},
            )

            with self.assertRaisesRegex(
                Tau3CompetitiveV3BindingError,
                "inside the bundle",
            ):
                bind_tau3_competitive_v3_evidence(
                    plan_path.parent,
                    stage="dataset",
                    evidence_path=outside,
                )

    def test_refuses_wrong_stage_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "bundle"
            _, evidence_path = write_fixture(bundle)
            write_json(evidence_path, {"schema_version": "wrong"})

            with self.assertRaisesRegex(
                Tau3CompetitiveV3BindingError,
                f"schema_version must be {DATASET_SCHEMA_VERSION}",
            ):
                bind_tau3_competitive_v3_evidence(
                    bundle,
                    stage="dataset",
                    evidence_path=evidence_path,
                )

    def test_cli_reports_binding_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "bundle"
            plan_path, evidence_path = write_fixture(bundle)

            proc = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--bundle",
                    str(bundle),
                    "--stage",
                    "dataset",
                    "--evidence",
                    str(evidence_path),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            result = json.loads(proc.stdout)

            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue(result["passed"])
            self.assertTrue(result["changed"])
            self.assertEqual(
                read_json(plan_path)["evidence_refs"]["dataset"],
                result["evidence_ref"],
            )


def write_fixture(bundle: Path) -> tuple[Path, Path]:
    bundle.mkdir(parents=True)
    plan_path = bundle / PLAN_FILENAME
    evidence_path = bundle / "dataset-evidence.json"
    write_json(plan_path, {"schema_version": PLAN_SCHEMA_VERSION})
    write_json(
        evidence_path,
        {"schema_version": DATASET_SCHEMA_VERSION},
    )
    return plan_path, evidence_path


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
