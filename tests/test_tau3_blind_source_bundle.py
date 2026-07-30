from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from flightrecorder.tau3_blind_source_bundle import (
    Tau3BlindSourceBundleError,
    validate_tau3_blind_source_bundle,
)
from flightrecorder.schema_registry import check_schema_contract


class Tau3BlindSourceBundleTests(unittest.TestCase):
    def test_validates_hash_only_cross_artifact_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._fixture(Path(tmp))
            result = validate_tau3_blind_source_bundle(**fixture)
            self.assertTrue(result["passed"])
            self.assertEqual(result["task_count"], 100)
            self.assertEqual(
                result["domain_counts"],
                {"airline": 34, "retail": 33, "telecom": 33},
            )
            self.assertTrue(result["hashes_only"])
            self.assertTrue(
                check_schema_contract(
                    result,
                    name_or_id="tau3_blind_source_bundle_validation",
                )["passed"]
            )

    def test_rejects_generator_script_substitution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._fixture(Path(tmp))
            script = fixture["generator_script"]
            script.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
            script.chmod(0o700)
            with self.assertRaisesRegex(
                Tau3BlindSourceBundleError,
                "exact executable|generator script hash mismatch",
            ):
                validate_tau3_blind_source_bundle(**fixture)

    def test_rejects_nonzero_contamination_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._fixture(Path(tmp))
            path = fixture["fresh_contamination_replay"]
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["overlaps"]["training_task"] = 1
            self._write_private(path, payload)
            with self.assertRaisesRegex(
                Tau3BlindSourceBundleError,
                "schema validation failed|nonzero overlap",
            ):
                validate_tau3_blind_source_bundle(**fixture)

    def test_rejects_group_readable_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._fixture(Path(tmp))
            fixture["sealed_source_manifest"].chmod(0o640)
            with self.assertRaisesRegex(
                Tau3BlindSourceBundleError,
                "group/world accessible",
            ):
                validate_tau3_blind_source_bundle(**fixture)

    def _fixture(self, root: Path) -> dict[str, object]:
        tau_repo = root / "tau"
        tau_repo.mkdir()
        (tau_repo / "README").write_text("pinned\n", encoding="utf-8")
        subprocess.run(
            ["git", "-C", str(tau_repo), "init", "-q"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(tau_repo), "add", "README"],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(tau_repo),
                "-c",
                "user.name=HFR Tests",
                "-c",
                "user.email=hfr-tests@example.invalid",
                "commit",
                "-q",
                "-m",
                "fixture",
            ],
            check=True,
        )
        revision = subprocess.run(
            ["git", "-C", str(tau_repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        generator_repo = root / "generator"
        generator_repo.mkdir()
        subprocess.run(
            ["git", "-C", str(generator_repo), "init", "-q"],
            check=True,
        )
        script = generator_repo / "custodian"
        script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        script.chmod(0o700)
        subprocess.run(
            ["git", "-C", str(generator_repo), "add", "custodian"],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(generator_repo),
                "-c",
                "user.name=HFR Tests",
                "-c",
                "user.email=hfr-tests@example.invalid",
                "commit",
                "-q",
                "-m",
                "generator fixture",
            ],
            check=True,
        )
        generator_commit = subprocess.run(
            ["git", "-C", str(generator_repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        training = root / "train.jsonl"
        training.write_text('{"row":1}\n', encoding="utf-8")
        development = root / "development.json"
        development.write_text('{"hashes_only":true}\n', encoding="utf-8")
        retired = root / "retired.json"
        retired.write_text('{"hashes_only":true}\n', encoding="utf-8")

        entries = []
        counts = {"airline": 34, "retail": 33, "telecom": 33}
        index = 0
        for domain, count in counts.items():
            for _ in range(count):
                entries.append(
                    {
                        "domain": domain,
                        "task_id_sha256": self._digest(f"id-{index}"),
                        "prompt_sha256": self._digest(f"prompt-{index}"),
                        "task_sha256": self._digest(f"task-{index}"),
                    }
                )
                index += 1
        sealed = root / "sealed.json"
        self._write_private(
            sealed,
            {
                "schema_version": "hfr.tau3_sealed_source_manifest.v1",
                "source_revision": revision,
                "hashes_only": True,
                "task_count": 100,
                "domain_counts": counts,
                "entries": entries,
            },
        )
        validation = root / "generator.json"
        self._write_private(
            validation,
            {
                "schema_version": "hfr.tau3_blind_generator_validation.v1",
                "created_at": "2026-07-30T00:00:00Z",
                "passed": True,
                "source_revision": revision,
                "sealed_source_manifest_sha256": self._sha256(sealed),
                "task_count": 100,
                "domain_counts": counts,
                "generator_source": {
                    "commit_sha": generator_commit,
                    "script_sha256": self._sha256(script),
                },
                "golden_replay": {
                    "passed": True,
                    "replayed_task_count": 100,
                    "passed_task_count": 100,
                    "failed_task_count": 0,
                    "state_check_failure_count": 0,
                },
                "schema_validation_passed": True,
                "task_hashes_unique": True,
                "prompt_hashes_unique": True,
                "hashes_only": True,
                "local_paths_included": False,
                "raw_payload_included": False,
            },
        )
        overlaps = {
            f"{source}_{kind}": 0
            for source in ("training", "development", "retired_sealed")
            for kind in ("task", "task_id", "prompt", "family")
        }
        contamination = root / "contamination.json"
        self._write_private(
            contamination,
            {
                "schema_version": "hfr.tau3_fresh_contamination_replay.v1",
                "created_at": "2026-07-30T00:00:00Z",
                "passed": True,
                "training_dataset_sha256": self._sha256(training),
                "development_source_sha256": self._sha256(development),
                "retired_sealed_source_manifest_sha256": self._sha256(retired),
                "fresh_sealed_source_manifest_sha256": self._sha256(sealed),
                "overlaps": overlaps,
                "hashes_only": True,
                "local_paths_included": False,
                "raw_payload_included": False,
            },
        )
        return {
            "sealed_source_manifest": sealed,
            "generator_validation": validation,
            "fresh_contamination_replay": contamination,
            "generator_script": script,
            "tau_repo": tau_repo,
            "training_dataset": training,
            "development_source": development,
            "retired_sealed_source": retired,
            "expected_source_revision": revision,
            "expected_generator_commit": generator_commit,
        }

    def _write_private(self, path: Path, payload: dict[str, object]) -> None:
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        path.chmod(0o600)

    def _sha256(self, path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _digest(self, value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    unittest.main()
