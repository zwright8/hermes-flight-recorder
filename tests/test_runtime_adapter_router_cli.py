from __future__ import annotations

import copy
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from flightrecorder import cli
from flightrecorder.runtime_adapter_router import canonical_sha256


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
BASE_ENV = {
    "base_model_id": "Qwen/Qwen3-0.6B",
    "base_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
    "tokenizer_revision": "tok-rev-1",
    "chat_template_sha256": SHA_D,
}
ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_ref(path: Path, base: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": str(path.relative_to(base)),
        "sha256": __import__("hashlib").sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def run_cli(argv: list[str]) -> tuple[int, str, str]:
    stdout = StringIO()
    stderr = StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            code = cli.main(argv)
        except SystemExit as exc:
            code = int(exc.code) if isinstance(exc.code, int) else 1
    return code, stdout.getvalue(), stderr.getvalue()


def task_contract() -> dict[str, object]:
    return {
        "task_id": "task-1",
        "contract_fingerprint": SHA_A,
        "capabilities": {"required": ["browser.search"], "optional": ["browser.read"]},
        "domains": ["browser"],
        "requires_tools": True,
        "allow_no_tools": False,
    }


def tool_catalog() -> dict[str, object]:
    return {
        "tools": [
            {
                "name": "browser.search",
                "version": "1",
                "definition_sha256": SHA_A,
                "capabilities": ["browser.search", "browser.read"],
                "risk_class": "read",
                "write_capable": False,
                "parameters_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "minLength": 1}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            }
        ]
    }


def promotion_decision() -> dict[str, object]:
    return copy.deepcopy(
        json.loads((ROOT / "examples/agentic_training/promotion_governance/promotion_decision.json").read_text(encoding="utf-8"))
    )


class RuntimeAdapterRouterCliTests(unittest.TestCase):
    def test_cli_builds_validates_and_rejects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_path = root / "task.json"
            tool_catalog_path = root / "tool_catalog.json"
            tool_policy_path = root / "tool_policy.json"
            env_path = root / "environment.json"
            selection_path = root / "selection.json"
            route_policy_path = root / "route_policy.json"
            candidate_catalog_path = root / "candidate_catalog.json"
            route_path = root / "route.json"
            promotion_path = root / "promotion.json"
            training_path = root / "training_result.json"
            evaluation_path = root / "evaluation_result.json"

            write_json(task_path, task_contract())
            write_json(tool_catalog_path, tool_catalog())
            write_json(
                tool_policy_path,
                {
                    "policy_id": "tool-policy",
                    "known_capabilities": ["browser.search", "browser.read"],
                    "allowed_risk_classes": ["read"],
                    "allow_write_tools": False,
                },
            )
            write_json(env_path, {"runtime": "local", **BASE_ENV})
            write_json(route_policy_path, {"policy_id": "route-policy", "allow_generalist_fallback": True, **BASE_ENV})
            write_json(promotion_path, promotion_decision())
            write_json(training_path, {"schema_version": "test.training", "ok": True})
            write_json(evaluation_path, {"schema_version": "test.evaluation", "ok": True})

            candidate = {
                "candidate_id": "local/mock-candidate",
                "candidate_kind": "specialist",
                "adapter_id": "adapter-browser",
                "active_adapter_ids": ["adapter-browser"],
                "adapter_revision": "rev-browser",
                "adapter_sha256": SHA_B,
                **BASE_ENV,
                "domains": ["browser"],
                "capabilities": ["browser.search"],
                "registry_entry_id": "registry-browser",
                "promotion_decision": read_json(promotion_path),
                "promotion_evidence_ref": file_ref(promotion_path, root),
                "promotion_binding": {
                    "independent_evidence": True,
                    "candidate_id": "local/mock-candidate",
                    "registry_entry_id": "registry-browser",
                    "adapter_id": "adapter-browser",
                    "adapter_revision": "rev-browser",
                    "adapter_sha256": SHA_B,
                    **BASE_ENV,
                    "training_result_ref": file_ref(training_path, root),
                    "evaluation_result_ref": file_ref(evaluation_path, root),
                },
            }
            write_json(candidate_catalog_path, {"candidates": [candidate]})

            self.assertEqual(
                run_cli(
                    [
                        "runtime-router",
                        "tool-capabilities",
                        "--task-contract",
                        str(task_path),
                        "--tool-catalog",
                        str(tool_catalog_path),
                        "--policy",
                        str(tool_policy_path),
                        "--environment",
                        str(env_path),
                        "--out",
                        str(selection_path),
                    ]
                )[0],
                0,
            )
            self.assertEqual(
                run_cli(
                    [
                        "runtime-router",
                        "adapter",
                        "--task-contract",
                        str(task_path),
                        "--capability-selection",
                        str(selection_path),
                        "--candidate-catalog",
                        str(candidate_catalog_path),
                        "--routing-policy",
                        str(route_policy_path),
                        "--runtime-environment",
                        str(env_path),
                        "--out",
                        str(route_path),
                    ]
                )[0],
                0,
            )

            code, _, _ = run_cli(
                [
                    "validate",
                    "--tool-capability-selection",
                    str(selection_path),
                    "--adapter-route-decision",
                    str(route_path),
                    "--strict",
                ]
            )
            self.assertEqual(code, 0)

            self.assertNotEqual(
                run_cli(
                    [
                        "runtime-router",
                        "tool-capabilities",
                        "--task-contract",
                        str(task_path),
                        "--tool-catalog",
                        str(tool_catalog_path),
                        "--policy",
                        str(tool_policy_path),
                        "--out",
                        str(selection_path),
                    ]
                )[0],
                0,
            )

            tampered_selection = copy.deepcopy(read_json(selection_path))
            tampered_selection["selected_tools"][0]["definition_sha256"] = SHA_C  # type: ignore[index]
            tampered_selection_path = root / "selection_tampered.json"
            write_json(tampered_selection_path, tampered_selection)
            self.assertNotEqual(
                run_cli(["validate", "--tool-capability-selection", str(tampered_selection_path), "--strict"])[0],
                0,
            )

            tampered_route = copy.deepcopy(read_json(route_path))
            tampered_route["capability_selection_ref"]["path"] = "../selection.json"  # type: ignore[index]
            tampered_route_path = root / "route_tampered.json"
            write_json(tampered_route_path, tampered_route)
            self.assertNotEqual(
                run_cli(["validate", "--adapter-route-decision", str(tampered_route_path), "--strict"])[0],
                0,
            )

            tampered_promotion_route = copy.deepcopy(read_json(route_path))
            tampered_promotion_route["evaluated_candidates"][0]["promotion_evidence"]["artifact_ref"]["sha256"] = SHA_C  # type: ignore[index]
            tampered_promotion_path = root / "route_promotion_tampered.json"
            write_json(tampered_promotion_path, tampered_promotion_route)
            self.assertNotEqual(
                run_cli(["validate", "--adapter-route-decision", str(tampered_promotion_path), "--strict"])[0],
                0,
            )

            promotion_contents_tampered = copy.deepcopy(read_json(route_path))
            old_promotion = read_json(promotion_path)
            old_promotion["recommendation"] = "block_promotion"
            write_json(promotion_path, old_promotion)
            self.assertNotEqual(
                run_cli(["validate", "--adapter-route-decision", str(route_path), "--strict"])[0],
                0,
            )

            self.assertNotEqual(canonical_sha256(read_json(promotion_path)), promotion_contents_tampered["evaluated_candidates"][0]["promotion_evidence"]["decision_sha256"])  # type: ignore[index]


if __name__ == "__main__":
    unittest.main()
