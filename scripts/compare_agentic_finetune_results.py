#!/usr/bin/env python3
"""Compare baseline, trace-only, and Flight Recorder fine-tune suite results."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


FORBIDDEN_RULE_IDS = ("forbidden_actions", "secret_exposure")
UNSUPPORTED_CLAIM_RULE_IDS = ("required_evidence", "final_answer", "unsupported_claim")
SCENARIO_COMPARABILITY_CHECK = "same_heldout_scenarios"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
    }


def counts(items: list[dict[str, Any]]) -> dict[str, int]:
    return {str(item["id"]): int(item["count"]) for item in items if "id" in item and "count" in item}


def total_count(counter: dict[str, int], keys: tuple[str, ...] | None = None) -> int:
    if keys is None:
        return sum(counter.values())
    return sum(counter.get(key, 0) for key in keys)


def resolve_run_dir(summary_path: Path, run_dir: str) -> Path:
    raw = Path(run_dir)
    if raw.exists():
        return raw
    candidate = summary_path.parent / raw.name
    if candidate.exists():
        return candidate
    return raw


def task_completion_metrics(summary_path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    configured = 0
    complete = 0
    incomplete = 0
    passed_checks = 0
    required_checks = 0
    missing_files = []
    for run in summary.get("runs", []):
        run_dir = resolve_run_dir(summary_path, str(run.get("run_dir") or ""))
        path = run_dir / "task_completion.json"
        if not path.exists():
            missing_files.append(str(path))
            continue
        data = load_json(path)
        if data.get("task_evidence_configured"):
            configured += 1
        if data.get("status") == "complete":
            complete += 1
        elif data.get("status") == "incomplete":
            incomplete += 1
        passed_checks += int(data.get("passed_check_count") or 0)
        required_checks += int(data.get("required_check_count") or 0)
    return {
        "configured": configured,
        "complete": complete,
        "incomplete": incomplete,
        "passed_checks": passed_checks,
        "required_checks": required_checks,
        "check_pass_rate": round(passed_checks / required_checks, 4) if required_checks else None,
        "missing_files": missing_files,
    }


def summarize(summary_path: Path) -> dict[str, Any]:
    summary = load_json(summary_path)
    metrics = summary.get("metrics") or {}
    critical = counts(metrics.get("critical_failure_counts", []))
    failed_rules = counts(metrics.get("failed_rule_counts", []))
    scenario_ids = sorted(str(run.get("scenario_id")) for run in summary.get("runs", []))
    return {
        "path": str(summary_path),
        "total": int(summary.get("total") or len(summary.get("runs", []))),
        "passed": int(summary.get("passed") or metrics.get("passed") or 0),
        "failed": int(summary.get("failed") or metrics.get("failed") or 0),
        "pass_rate": float(metrics.get("pass_rate") or 0.0),
        "average_score": float(metrics.get("average_score") or 0.0),
        "critical_failure_counts": critical,
        "critical_failure_total": total_count(critical),
        "failed_rule_counts": failed_rules,
        "forbidden_action_failures": total_count(critical, FORBIDDEN_RULE_IDS),
        "unsupported_claim_failures": total_count(critical, UNSUPPORTED_CLAIM_RULE_IDS),
        "scenario_ids": scenario_ids,
        "task_completion": task_completion_metrics(summary_path, summary),
        "artifact_hashes": {
            "suite_summary": artifact_record(summary_path),
        },
    }


def better_than(candidate: float | int | None, reference: float | int | None) -> bool:
    if candidate is None or reference is None:
        return False
    return candidate > reference


def lower_than(candidate: float | int | None, reference: float | int | None) -> bool:
    if candidate is None or reference is None:
        return False
    return candidate < reference


def no_more_than(candidate: float | int | None, reference: float | int | None) -> bool:
    if candidate is None or reference is None:
        return False
    return candidate <= reference


def check(name: str, passed: bool, actual: Any, expected: Any, **extra: Any) -> dict[str, Any]:
    result = {"id": name, "passed": bool(passed), "actual": actual, "expected": expected}
    result.update(extra)
    return result


def comparative_check(name: str, scenario_comparable: bool, passed: bool, actual: Any, expected: Any) -> dict[str, Any]:
    if scenario_comparable:
        return check(name, passed, actual, expected, requires_identical_scenarios=True)
    return check(
        name,
        False,
        actual,
        expected,
        requires_identical_scenarios=True,
        blocked_by=SCENARIO_COMPARABILITY_CHECK,
        reason="Comparative claim suppressed because held-out scenario id lists are not identical across arms.",
    )


def scenario_set_summary(
    baseline: dict[str, Any],
    trace_only: dict[str, Any],
    flightrecorder: dict[str, Any],
) -> dict[str, Any]:
    arms = {
        "baseline": baseline["scenario_ids"],
        "trace_only": trace_only["scenario_ids"],
        "flightrecorder": flightrecorder["scenario_ids"],
    }
    arm_sets = {name: set(ids) for name, ids in arms.items()}
    all_ids = sorted(set().union(*arm_sets.values()))
    common_ids = sorted(set.intersection(*arm_sets.values())) if arm_sets else []
    return {
        "identical": baseline["scenario_ids"] == trace_only["scenario_ids"] == flightrecorder["scenario_ids"],
        "requires_identical_scenarios_for_claims": True,
        "scenario_ids": baseline["scenario_ids"] if baseline["scenario_ids"] == trace_only["scenario_ids"] == flightrecorder["scenario_ids"] else [],
        "common_scenario_ids": common_ids,
        "all_scenario_ids": all_ids,
        "arms": arms,
        "missing_by_arm": {
            name: [scenario_id for scenario_id in all_ids if scenario_id not in ids]
            for name, ids in arm_sets.items()
        },
    }


def governance_handoff(checks: list[dict[str, Any]], scenario_set: dict[str, Any]) -> dict[str, Any]:
    failed_checks = [str(item["id"]) for item in checks if not item["passed"]]
    if not scenario_set["identical"]:
        return {
            "ready": False,
            "status": "not_comparable",
            "recommendation": "rerun_identical_scenario_eval",
            "failed_checks": failed_checks,
            "blocking_reasons": [SCENARIO_COMPARABILITY_CHECK],
            "next_actions": [
                "Rerun baseline, trace-only, champion, and candidate arms on the exact same held-out scenario id list.",
                "Do not use pass-rate, score, or regression deltas from this comparison for promotion decisions.",
            ],
        }
    if failed_checks:
        return {
            "ready": True,
            "status": "blocked_by_eval_checks",
            "recommendation": "block_promotion",
            "failed_checks": failed_checks,
            "blocking_reasons": failed_checks,
            "next_actions": [
                "Open repair work for the failed eval checks before asking Governance to move aliases.",
            ],
        }
    return {
        "ready": True,
        "status": "eval_checks_passed",
        "recommendation": "send_to_governance",
        "failed_checks": [],
        "blocking_reasons": [],
        "next_actions": [
            "Governance must still verify evidence, data, model, serving, safety, license, rollback, and card gates.",
        ],
    }


def repair_work_items(checks: list[dict[str, Any]], scenario_set: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for item in checks:
        if item["passed"]:
            continue
        check_id = str(item["id"])
        priority = "critical" if check_id == SCENARIO_COMPARABILITY_CHECK or item.get("blocked_by") else "high"
        items.append(
            {
                "schema_version": "hfr.eval.repair_work_item.v1",
                "id": f"eval_{check_id}",
                "source_check": check_id,
                "priority": priority,
                "scenario_ids": scenario_set["all_scenario_ids"],
                "reason": item.get("reason") or f"Eval comparison check {check_id!r} failed.",
                "suggested_action": _repair_action(check_id),
            }
        )
    return items


def repair_artifact(result: dict[str, Any]) -> dict[str, Any]:
    items = list(result.get("repair_work_items") or [])
    return {
        "schema_version": "hfr.eval.repair_work_items.v1",
        "source_comparison": {
            "schema_version": result.get("schema_version"),
            "comparison_status": result.get("comparison_status"),
            "scenario_set": result.get("scenario_set"),
        },
        "item_count": len(items),
        "items": items,
        "governance_handoff": {
            "ready": True,
            "status": "repair_items_available" if items else "no_repair_items",
            "recommendation": "send_to_repair_loop" if items else "no_action",
            "next_actions": _repair_artifact_next_actions(items),
        },
    }


def curriculum_artifact(result: dict[str, Any]) -> dict[str, Any]:
    suggestions = [_curriculum_suggestion(item) for item in result.get("repair_work_items") or []]
    return {
        "schema_version": "hfr.eval.curriculum_suggestions.v1",
        "source_comparison": {
            "schema_version": result.get("schema_version"),
            "comparison_status": result.get("comparison_status"),
            "scenario_set": result.get("scenario_set"),
        },
        "suggestion_count": len(suggestions),
        "suggestions": suggestions,
        "governance_handoff": {
            "ready": True,
            "status": "curriculum_suggestions_available" if suggestions else "no_curriculum_suggestions",
            "recommendation": "send_to_curriculum_loop" if suggestions else "no_action",
            "next_actions": _curriculum_artifact_next_actions(suggestions),
        },
    }


def write_repair_outputs(result: dict[str, Any], *, repair_out: Path, curriculum_out: Path) -> dict[str, Any]:
    repair = repair_artifact(result)
    curriculum = curriculum_artifact(result)
    write_json(repair_out, repair)
    write_json(curriculum_out, curriculum)
    return {
        "eval_repair_work_items": artifact_record(repair_out),
        "eval_curriculum_suggestions": artifact_record(curriculum_out),
    }


def _repair_artifact_next_actions(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["No eval repair work items were generated by this comparison."]
    return [
        "Review the failed eval checks and open repair tasks before promotion review.",
        "Use scenario ids from each item to rerun the same held-out scenario list after repairs.",
    ]


def _curriculum_artifact_next_actions(suggestions: list[dict[str, Any]]) -> list[str]:
    if not suggestions:
        return ["No eval curriculum suggestions were generated by this comparison."]
    return [
        "Convert high-priority suggestions into split-safe repair or curriculum examples.",
        "Keep held-out scenario ids excluded from training rows unless a new split manifest explicitly moves them.",
    ]


def _curriculum_suggestion(item: dict[str, Any]) -> dict[str, Any]:
    source_check = str(item.get("source_check") or "unknown")
    return {
        "schema_version": "hfr.eval.curriculum_suggestion.v1",
        "id": f"curriculum_{source_check}",
        "source_repair_item": item.get("id"),
        "source_check": source_check,
        "priority": item.get("priority") or "medium",
        "scenario_ids": item.get("scenario_ids") if isinstance(item.get("scenario_ids"), list) else [],
        "curriculum_focus": _curriculum_focus(source_check),
        "suggested_action": item.get("suggested_action") or _repair_action(source_check),
    }


def _curriculum_focus(check_id: str) -> str:
    if check_id == SCENARIO_COMPARABILITY_CHECK:
        return "heldout_scenario_set_discipline"
    if "forbidden_action" in check_id:
        return "forbidden_action_red_team"
    if "unsupported_claim" in check_id:
        return "evidence_backed_claims"
    if "task_completion" in check_id:
        return "task_completion_evidence"
    if "critical_failures" in check_id:
        return "critical_failure_regression"
    return "eval_regression"


def _repair_action(check_id: str) -> str:
    if check_id == SCENARIO_COMPARABILITY_CHECK:
        return "Rerun every compared arm against an identical held-out scenario list before making comparative claims."
    if "forbidden_action" in check_id:
        return "Add or repair red-team coverage for forbidden actions and rerun the held-out eval."
    if "unsupported_claim" in check_id:
        return "Add evidence-backed task-completion examples and rerun unsupported-claim regression scenarios."
    if "task_completion" in check_id:
        return "Add curriculum or repair examples for the regressed task-completion evidence path."
    return "Inspect failed scenarios, update repair/curriculum data, and rerun the held-out comparison."


def compare(baseline: dict[str, Any], trace_only: dict[str, Any], flightrecorder: dict[str, Any]) -> dict[str, Any]:
    scenario_set = scenario_set_summary(baseline, trace_only, flightrecorder)
    same_scenarios = scenario_set["identical"]
    checks = [
        check(
            SCENARIO_COMPARABILITY_CHECK,
            same_scenarios,
            {
                "baseline": baseline["scenario_ids"],
                "trace_only": trace_only["scenario_ids"],
                "flightrecorder": flightrecorder["scenario_ids"],
            },
            "identical scenario id lists",
        ),
        comparative_check(
            "higher_pass_rate_than_baseline",
            same_scenarios,
            better_than(flightrecorder["pass_rate"], baseline["pass_rate"]),
            flightrecorder["pass_rate"],
            {">": baseline["pass_rate"]},
        ),
        comparative_check(
            "higher_pass_rate_than_trace_only",
            same_scenarios,
            better_than(flightrecorder["pass_rate"], trace_only["pass_rate"]),
            flightrecorder["pass_rate"],
            {">": trace_only["pass_rate"]},
        ),
        comparative_check(
            "higher_average_score_than_baseline",
            same_scenarios,
            better_than(flightrecorder["average_score"], baseline["average_score"]),
            flightrecorder["average_score"],
            {">": baseline["average_score"]},
        ),
        comparative_check(
            "higher_average_score_than_trace_only",
            same_scenarios,
            better_than(flightrecorder["average_score"], trace_only["average_score"]),
            flightrecorder["average_score"],
            {">": trace_only["average_score"]},
        ),
        comparative_check(
            "fewer_critical_failures_than_baseline",
            same_scenarios,
            lower_than(flightrecorder["critical_failure_total"], baseline["critical_failure_total"]),
            flightrecorder["critical_failure_total"],
            {"<": baseline["critical_failure_total"]},
        ),
        comparative_check(
            "fewer_critical_failures_than_trace_only",
            same_scenarios,
            lower_than(flightrecorder["critical_failure_total"], trace_only["critical_failure_total"]),
            flightrecorder["critical_failure_total"],
            {"<": trace_only["critical_failure_total"]},
        ),
        comparative_check(
            "improved_task_completion_vs_baseline",
            same_scenarios,
            better_than(
                flightrecorder["task_completion"]["check_pass_rate"],
                baseline["task_completion"]["check_pass_rate"],
            ),
            flightrecorder["task_completion"],
            {"> check_pass_rate": baseline["task_completion"]["check_pass_rate"]},
        ),
        comparative_check(
            "improved_task_completion_vs_trace_only",
            same_scenarios,
            better_than(
                flightrecorder["task_completion"]["check_pass_rate"],
                trace_only["task_completion"]["check_pass_rate"],
            ),
            flightrecorder["task_completion"],
            {"> check_pass_rate": trace_only["task_completion"]["check_pass_rate"]},
        ),
        comparative_check(
            "no_new_forbidden_action_regressions_vs_baseline",
            same_scenarios,
            no_more_than(flightrecorder["forbidden_action_failures"], baseline["forbidden_action_failures"]),
            flightrecorder["forbidden_action_failures"],
            {"<=": baseline["forbidden_action_failures"]},
        ),
        comparative_check(
            "no_new_forbidden_action_regressions_vs_trace_only",
            same_scenarios,
            no_more_than(flightrecorder["forbidden_action_failures"], trace_only["forbidden_action_failures"]),
            flightrecorder["forbidden_action_failures"],
            {"<=": trace_only["forbidden_action_failures"]},
        ),
        comparative_check(
            "no_new_unsupported_claim_regressions_vs_baseline",
            same_scenarios,
            no_more_than(flightrecorder["unsupported_claim_failures"], baseline["unsupported_claim_failures"]),
            flightrecorder["unsupported_claim_failures"],
            {"<=": baseline["unsupported_claim_failures"]},
        ),
        comparative_check(
            "no_new_unsupported_claim_regressions_vs_trace_only",
            same_scenarios,
            no_more_than(flightrecorder["unsupported_claim_failures"], trace_only["unsupported_claim_failures"]),
            flightrecorder["unsupported_claim_failures"],
            {"<=": trace_only["unsupported_claim_failures"]},
        ),
    ]
    failed = [item for item in checks if not item["passed"]]
    handoff = governance_handoff(checks, scenario_set)
    return {
        "schema_version": "hfr.agentic_finetune_promotion_comparison.v1",
        "passed": not failed,
        "comparison_status": "comparable" if same_scenarios else "not_comparable",
        "scenario_set": scenario_set,
        "governance_handoff": handoff,
        "repair_work_items": repair_work_items(checks, scenario_set),
        "check_count": len(checks),
        "failed_check_count": len(failed),
        "checks": checks,
        "summaries": {
            "baseline": baseline,
            "trace_only": trace_only,
            "flightrecorder": flightrecorder,
        },
    }


def write_report(path: Path, result: dict[str, Any]) -> None:
    handoff = result["governance_handoff"]
    lines = [
        "# Agentic Fine-Tune Promotion Comparison",
        "",
        f"- Eval checks passed: {result['passed']}",
        f"- Comparison status: {result['comparison_status']}",
        f"- Governance status: {handoff['status']}",
        f"- Governance recommendation: {handoff['recommendation']}",
        f"- Checks: {result['check_count']}",
        f"- Failed checks: {result['failed_check_count']}",
        "",
        "## Scenario Comparability",
        "",
        f"- Identical held-out scenarios: {result['scenario_set']['identical']}",
        f"- Scenario ids: {result['scenario_set']['scenario_ids'] or 'not comparable'}",
        "",
        "## Summary Metrics",
        "",
        "| Arm | Pass Rate | Average Score | Critical Failures | Forbidden | Unsupported Claims | Task Check Pass Rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, summary in result["summaries"].items():
        task_rate = summary["task_completion"]["check_pass_rate"]
        lines.append(
            "| {name} | {pass_rate} | {score} | {critical} | {forbidden} | {unsupported} | {task_rate} |".format(
                name=name,
                pass_rate=summary["pass_rate"],
                score=summary["average_score"],
                critical=summary["critical_failure_total"],
                forbidden=summary["forbidden_action_failures"],
                unsupported=summary["unsupported_claim_failures"],
                task_rate=task_rate,
            )
        )
    lines.extend(["", "## Checks", ""])
    for item in result["checks"]:
        mark = "PASS" if item["passed"] else "FAIL"
        blocked = f" blocked_by={item['blocked_by']}" if item.get("blocked_by") else ""
        lines.append(f"- {mark}: `{item['id']}` actual={item['actual']} expected={item['expected']}{blocked}")
    lines.extend(["", "## Governance Handoff", ""])
    lines.append(f"- Ready for governance consumption: {handoff['ready']}")
    lines.append(f"- Blocking reasons: {handoff['blocking_reasons'] or 'none'}")
    lines.append("- Next actions:")
    for action in handoff["next_actions"]:
        lines.append(f"  - {action}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline suite_summary.json")
    parser.add_argument("--trace-only", type=Path, required=True, help="Trace-only fine-tuned suite_summary.json")
    parser.add_argument("--flightrecorder", type=Path, required=True, help="Flight Recorder fine-tuned suite_summary.json")
    parser.add_argument("--out", type=Path, default=Path("experiments/qwen3_4b_flightrecorder/promotion_comparison.json"))
    parser.add_argument("--report", type=Path, default=Path("experiments/qwen3_4b_flightrecorder/PROMOTION_REPORT.md"))
    parser.add_argument("--repair-out", type=Path, help="Standalone eval repair work items JSON; defaults beside --out")
    parser.add_argument("--curriculum-out", type=Path, help="Standalone eval curriculum suggestions JSON; defaults beside --out")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = compare(summarize(args.baseline), summarize(args.trace_only), summarize(args.flightrecorder))
    result["input_artifacts"] = {
        "baseline_suite_summary": artifact_record(args.baseline),
        "trace_only_suite_summary": artifact_record(args.trace_only),
        "flightrecorder_suite_summary": artifact_record(args.flightrecorder),
    }
    repair_out = args.repair_out or args.out.with_name("eval_repair_work_items.json")
    curriculum_out = args.curriculum_out or args.out.with_name("eval_curriculum_suggestions.json")
    result["output_artifacts"] = write_repair_outputs(result, repair_out=repair_out, curriculum_out=curriculum_out)
    write_json(args.out, result)
    write_report(args.report, result)
    print(json.dumps({"passed": result["passed"], "failed_check_count": result["failed_check_count"], "out": str(args.out)}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
