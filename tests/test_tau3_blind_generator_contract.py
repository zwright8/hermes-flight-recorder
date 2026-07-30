from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def _load_generator():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / ("generate_tau3_blind_sealed_source.py")
    )
    spec = importlib.util.spec_from_file_location("tau3_blind_generator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load blind generator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Tau3BlindGeneratorContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.generator = _load_generator()

    def test_creates_only_exact_private_result_hierarchy(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            output = Path(tmp) / "results" / "airline" / "seed-101" / "results.json"
            self.generator._prepare_result_output(
                output,
                domain="airline",
                seed=101,
            )
            self.assertTrue(output.parent.is_dir())
            self.assertEqual(output.parent.stat().st_mode & 0o777, 0o700)
            self.assertFalse(output.exists())

    def test_rejects_coordinate_drift(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            output = Path(tmp) / "results" / "retail" / "seed-202" / "different.json"
            with self.assertRaisesRegex(
                self.generator.CustodianError,
                "new absolute path",
            ):
                self.generator._prepare_result_output(
                    output,
                    domain="retail",
                    seed=202,
                )

    def test_rejects_symlinked_result_parent(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            actual = root / "actual"
            actual.mkdir()
            linked = root / "results"
            linked.symlink_to(actual, target_is_directory=True)
            output = linked / "telecom" / "seed-303" / "results.json"
            with self.assertRaisesRegex(
                self.generator.CustodianError,
                "symlink component",
            ):
                self.generator._prepare_result_output(
                    output,
                    domain="telecom",
                    seed=303,
                )

    def test_request_contract_accepts_only_frozen_loopback_harness(self) -> None:
        request = self._request()
        self.generator._require_request(request)
        request["agent"]["llm_args"]["api_base"] = "https://example.com/v1"
        with self.assertRaisesRegex(
            self.generator.CustodianError,
            "endpoint is not loopback",
        ):
            self.generator._require_request(request)

    def test_request_contract_rejects_timeout_drift(self) -> None:
        request = self._request()
        request["harness"]["timeout_seconds"] = 0
        with self.assertRaisesRegex(
            self.generator.CustodianError,
            "timeout is invalid",
        ):
            self.generator._require_request(request)

    def test_prepare_never_overwrites_or_deletes_existing_output(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            existing = root / "sealed.json"
            existing.write_text("preserve\n", encoding="utf-8")
            args = SimpleNamespace(
                sealed_manifest_out=existing,
                generator_validation_out=root / "generator.json",
                contamination_out=root / "contamination.json",
            )
            with (
                patch.object(self.generator, "_prepare_new") as prepare_new,
                self.assertRaisesRegex(
                    self.generator.CustodianError,
                    "refusing to overwrite",
                ),
            ):
                self.generator.prepare(args)
            prepare_new.assert_not_called()
            self.assertEqual(existing.read_text(encoding="utf-8"), "preserve\n")

    def _request(self) -> dict:
        decoding = {
            "api_base": "http://127.0.0.1:18080/v1",
            "api_key": "local",
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 1024,
            "num_retries": 0,
        }
        return {
            "schema_version": self.generator.REQUEST_SCHEMA,
            "result_schema_version": self.generator.RESULT_SCHEMA,
            "domain": "airline",
            "seed": 101,
            "agent": {
                "implementation": "llm_agent",
                "model": "local/agent",
                "llm_args": dict(decoding),
            },
            "user": {
                "implementation": "user_simulator",
                "model": "local/user",
                "llm_args": dict(decoding),
            },
            "reviewer": {
                "model": "local/reviewer",
                "api_base": "http://127.0.0.1:18082/v1",
            },
            "harness": {
                "num_trials": 1,
                "max_steps": 30,
                "max_errors": 10,
                "timeout_seconds": 600,
                "max_concurrency": 1,
                "max_retries": 0,
                "hallucination_retries": 0,
                "auto_resume": False,
                "auto_review": True,
                "review_mode": "full",
                "communication_protocol_enforced": True,
                "context_window": self.generator.CONTEXT_WINDOW,
                "test_time_search": False,
            },
        }


if __name__ == "__main__":
    unittest.main()
