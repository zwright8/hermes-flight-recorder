#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ -z "${PYTHON:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON=python
  elif command -v python3.11 >/dev/null 2>&1; then
    PYTHON=python3.11
  else
    PYTHON=python3
  fi
fi
export PYTHON

if "$PYTHON" -c 'import sys; raise SystemExit(0 if sys.flags.optimize else 1)'; then
  echo "release check requires Python assertions; unset PYTHONOPTIMIZE and do not use -O" >&2
  exit 1
fi

cleanup_local_artifacts() {
  find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
  rm -rf hermes_flight_recorder.egg-info build dist
}
trap cleanup_local_artifacts EXIT

assert_help_contains() {
  local expected="$1"
  shift
  local output
  output="$("$@" 2>&1)"
  grep -F -q -- "$expected" <<<"$output"
}

"$PYTHON" -m unittest discover
"$PYTHON" -m compileall -q flightrecorder scripts tests
"$PYTHON" scripts/live_hermes_smoke.py --help >/dev/null
"$PYTHON" scripts/live_verifier_smoke.py --help >/dev/null
assert_help_contains "--evidence-handoff" "$PYTHON" -m flightrecorder run-suite --help
"$PYTHON" -m flightrecorder schemas --help >/dev/null
"$PYTHON" -m flightrecorder schemas --name trace >/dev/null
rm -rf schema_contracts_check
"$PYTHON" -m flightrecorder schemas --write-dir schema_contracts_check >/dev/null
test -f schema_contracts_check/manifest.json
"$PYTHON" - <<'PY'
import json
from pathlib import Path

bundle = Path("schema_contracts_check")
catalog = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
missing = [
    record["filename"]
    for record in catalog.get("schemas", [])
    if not (bundle / record["filename"]).is_file()
]
if missing:
    raise SystemExit("missing exported schema file(s): " + ", ".join(sorted(missing)))
PY
"$PYTHON" scripts/live_verifier_smoke.py \
  --out runs/live_verifier_smoke_release_check \
  --force \
  --provider slack >/dev/null
"$PYTHON" -m flightrecorder schemas \
  --check runs/live_verifier_smoke_release_check/live_verifier_smoke_summary.json >/dev/null
"$PYTHON" -m flightrecorder schemas \
  --check scenarios/prompt_injection_good.json \
  --name scenario >/dev/null
"$PYTHON" -m flightrecorder schemas \
  --check scenarios/email_reply_completion_good.json \
  --name scenario >/dev/null
rm -rf schema_contracts_check
"$PYTHON" -m flightrecorder repair-queue --help >/dev/null
"$PYTHON" -m flightrecorder improvement-plan --help >/dev/null
"$PYTHON" -m flightrecorder improvement-ledger --help >/dev/null
"$PYTHON" -m flightrecorder gate-improvement-ledger --help >/dev/null
./demo.sh
grep -F -q "Evidence Artifacts" runs/index.html
grep -F -q "Improvement Ledger Gate" runs/index.html
grep -F -q "improvement_ledger_gate.json" runs/index.html
"$PYTHON" -m flightrecorder schemas --check runs/prompt_injection_good/normalized_trace.json >/dev/null
"$PYTHON" -m flightrecorder schemas --check runs/prompt_injection_good/scorecard.json >/dev/null
"$PYTHON" -m flightrecorder schemas --check runs/prompt_injection_good/task_completion.json >/dev/null
"$PYTHON" -m flightrecorder schemas --check runs/prompt_injection_good/run_digest.json >/dev/null
"$PYTHON" -m flightrecorder schemas --check runs/email_reply_completion_good/task_completion.json >/dev/null
"$PYTHON" -m flightrecorder schemas --check runs/email_reply_completion_good/state_diff.json >/dev/null
"$PYTHON" -m flightrecorder schemas --check runs/email_reply_completion_good/run_digest.json >/dev/null
"$PYTHON" -m flightrecorder schemas --check runs/evidence_bundle.json >/dev/null
"$PYTHON" -m flightrecorder schemas --check runs/improvement_plan.json >/dev/null
"$PYTHON" -m flightrecorder schemas --check runs/improvement_ledger.json >/dev/null
"$PYTHON" -m flightrecorder schemas --check runs/improvement_ledger_gate.json >/dev/null
"$PYTHON" -m flightrecorder schemas --check runs/training_export/manifest.json >/dev/null
"$PYTHON" -m flightrecorder schemas --check runs/training_export/dataset_metrics.json >/dev/null
"$PYTHON" -m flightrecorder schemas --check runs/training_export/curriculum.json >/dev/null
"$PYTHON" -m flightrecorder schemas --check runs/training_export/dataset_splits.json >/dev/null
"$PYTHON" -m flightrecorder schemas --check-jsonl runs/training_export/episodes.jsonl >/dev/null
"$PYTHON" -m flightrecorder schemas --check-jsonl runs/training_export/rewards.jsonl --name rl_reward >/dev/null
"$PYTHON" -m flightrecorder schemas --check-jsonl runs/training_export/step_rewards.jsonl --name rl_step_reward >/dev/null
"$PYTHON" -m flightrecorder schemas --check-jsonl runs/training_export/preferences.jsonl --name rl_preference >/dev/null
"$PYTHON" -m flightrecorder schemas --check-jsonl runs/training_export/failure_modes.jsonl --name rl_failure_mode >/dev/null
"$PYTHON" -m flightrecorder schemas --check-jsonl runs/training_export/sft.jsonl --name rl_sft >/dev/null
"$PYTHON" -m flightrecorder schemas --check-jsonl runs/training_export/dpo.jsonl --name rl_dpo >/dev/null
"$PYTHON" -m flightrecorder schemas --check-jsonl runs/training_export/reward_model.jsonl --name rl_reward_model >/dev/null
rm -rf replay_runs
"$PYTHON" -m flightrecorder replay-bundle \
  --lineage runs/prompt_injection_good/artifact_lineage.json \
  --out replay_runs/prompt_injection_good_bundle >/dev/null
mv replay_runs/prompt_injection_good_bundle replay_runs/moved_prompt_injection_good_bundle
"$PYTHON" -m flightrecorder validate \
  --replay-bundle replay_runs/moved_prompt_injection_good_bundle \
  --strict >/dev/null
"$PYTHON" -m flightrecorder replay \
  --lineage replay_runs/moved_prompt_injection_good_bundle/artifact_lineage.json \
  --out replay_runs/prompt_injection_good_replay >/dev/null
"$PYTHON" -m flightrecorder validate \
  --run replay_runs/prompt_injection_good_replay \
  --strict >/dev/null
"$PYTHON" -m flightrecorder check-scenarios \
  --scenarios scenarios \
  --require-traces \
  --strict \
  --out runs/scenario_check.json >/dev/null
test -f runs/scenario_check.json
"$PYTHON" -m flightrecorder schemas --check runs/scenario_check.json >/dev/null
"$PYTHON" -m flightrecorder validate --scenario-check runs/scenario_check.json --strict >/dev/null
"$PYTHON" -m flightrecorder scenario-quality \
  --scenarios scenarios \
  --require-traces \
  --out runs/scenario_quality_check.json \
  --min-average-score 80 \
  --min-scenario-score 60 \
  --min-observable-rate 0.8 \
  --max-weak-scenarios 0 \
  --max-final-only-scenarios 0 \
  --max-missing-traces 0 >/dev/null
test -f runs/scenario_quality_check.json
"$PYTHON" -m flightrecorder trace-observability \
  --runs runs \
  --out runs/trace_observability_check.json \
  --min-average-events 2 \
  --min-event-type-count 2 \
  --min-tool-or-api-run-rate 0.5 \
  --max-empty-final-answers 0 \
  --require-event-type assistant_message >/dev/null
test -f runs/trace_observability_check.json
"$PYTHON" -m flightrecorder validate \
  --scenario-quality runs/scenario_quality.json \
  --scenario-quality runs/scenario_quality_check.json \
  --evidence-bundle runs/evidence_bundle.json \
  --repair-queue runs/repair_queue.json \
  --trace-observability runs/trace_observability.json \
  --trace-observability runs/trace_observability_check.json \
  --strict >/dev/null
"$PYTHON" -m flightrecorder validate \
  --improvement-plan runs/improvement_plan.json \
  --improvement-ledger runs/improvement_ledger.json \
  --improvement-ledger-gate runs/improvement_ledger_gate.json \
  --strict >/dev/null
if "$PYTHON" -m flightrecorder gate-improvement-ledger \
  --improvement-ledger runs/improvement_ledger.json \
  --max-recurring-work-items 0 >/dev/null; then
  echo "gate-improvement-ledger did not fail a too-strict recurring work-item threshold" >&2
  exit 1
fi
if "$PYTHON" -m flightrecorder scenario-quality \
  --scenarios scenarios \
  --require-traces \
  --min-scenario-score 90 >/dev/null; then
  echo "scenario-quality did not fail a too-high minimum scenario score" >&2
  exit 1
fi
if "$PYTHON" -m flightrecorder trace-observability \
  --runs runs \
  --min-average-events 999 >/dev/null; then
  echo "trace-observability did not fail a too-high average event threshold" >&2
  exit 1
fi
"$PYTHON" -m flightrecorder draft-scenario \
  --trace fixtures/email_reply_completion_good.observer.jsonl \
  --id draft_email_reply \
  --title "Draft Email Reply" \
  --prompt "Reply to email-123." \
  --out runs/draft_email_reply.scenario.json >/dev/null
"$PYTHON" -m flightrecorder run \
  --scenario runs/draft_email_reply.scenario.json \
  --out runs/draft_email_reply \
  --fail-on-score >/dev/null
"$PYTHON" -m flightrecorder capture-state \
  --file task_completion=runs/email_reply_completion_good/task_completion.json \
  --json task_completion=runs/email_reply_completion_good/task_completion.json \
  --set gmail.threads.email-123.sent_replies.0.status=sent \
  --set gmail.threads.email-123.sent_replies.0.message_id=msg-email-123-001 \
  --out runs/captured_state.json >/dev/null
"$PYTHON" -m flightrecorder validate \
  --state-snapshot runs/captured_state.json \
  --strict >/dev/null
"$PYTHON" -m flightrecorder validate \
  --state-diff runs/email_reply_completion_good/state_diff.json \
  --strict >/dev/null
"$PYTHON" -m flightrecorder validate \
  --run-digest runs/email_reply_completion_good/run_digest.json \
  --strict >/dev/null
"$PYTHON" -m flightrecorder digest \
  --run runs/email_reply_completion_good \
  --out runs/email_reply_completion_good/regenerated_run_digest.json \
  --markdown-out runs/email_reply_completion_good/run_digest.md >/dev/null
"$PYTHON" -m flightrecorder report \
  --scenario scenarios/email_reply_completion_good.json \
  --trace runs/email_reply_completion_good/normalized_trace.json \
  --score runs/email_reply_completion_good/scorecard.json \
  --state-diff runs/email_reply_completion_good/state_diff.json \
  --out runs/email_reply_completion_good/standalone_state_report.html >/dev/null
test -f runs/draft_email_reply.scenario.json
test -f runs/draft_email_reply/scorecard.json
test -f runs/draft_email_reply/task_completion.json
test -f runs/draft_email_reply/artifact_lineage.json
test -f runs/captured_state.json
"$PYTHON" -m flightrecorder schemas --check runs/captured_state.json >/dev/null
test -f runs/email_reply_completion_good/scorecard.junit.xml
test -f runs/email_reply_completion_good/scorecard.md
test -f runs/email_reply_completion_good/state_snapshot.json
test -f runs/email_reply_completion_good/before_state_snapshot.json
test -f runs/email_reply_completion_good/state_diff.json
test -f runs/email_reply_completion_good/task_completion.json
test -f runs/email_reply_completion_good/run_digest.json
test -f runs/email_reply_completion_good/run_digest.md
test -f runs/email_reply_completion_good/regenerated_run_digest.json
test -f runs/email_reply_completion_good/artifact_lineage.json
test -f runs/email_reply_completion_good/standalone_state_report.html
test -f replay_runs/moved_prompt_injection_good_bundle/replay_bundle.json
test -f replay_runs/prompt_injection_good_replay/artifact_lineage.json
test -f runs/prompt_injection_compare.json
"$PYTHON" -m flightrecorder schemas --check runs/prompt_injection_compare.json >/dev/null
test -f runs/prompt_injection_compare.html
test -f runs/suite_compare.json
"$PYTHON" -m flightrecorder schemas --check runs/suite_compare.json >/dev/null
test -f runs/suite_compare.html
test -f runs/suite_trend.json
"$PYTHON" -m flightrecorder schemas --check runs/suite_trend.json >/dev/null
test -f runs/suite_trend.html
test -f runs/scenario_quality.json
"$PYTHON" -m flightrecorder schemas --check runs/scenario_quality.json >/dev/null
test -f runs/evidence_coverage.json
"$PYTHON" -m flightrecorder schemas --check runs/evidence_coverage.json >/dev/null
test -f runs/trace_observability.json
"$PYTHON" -m flightrecorder schemas --check runs/trace_observability.json >/dev/null
test -f runs/repair_queue.json
"$PYTHON" -m flightrecorder schemas --check runs/repair_queue.json >/dev/null
test -f runs/evidence_bundle.json
test -f runs/improvement_plan.json
test -f runs/improvement_ledger.json
test -f runs/improvement_ledger_gate.json
test -f runs/suite_summary.json
"$PYTHON" -m flightrecorder schemas --check runs/suite_summary.json >/dev/null
"$PYTHON" - <<'PY'
import json
from pathlib import Path

summary = json.loads(Path("runs/suite_summary.json").read_text(encoding="utf-8"))
scenario_quality = json.loads(Path("runs/scenario_quality.json").read_text(encoding="utf-8"))
evidence_coverage = json.loads(Path("runs/evidence_coverage.json").read_text(encoding="utf-8"))
trace_observability = json.loads(Path("runs/trace_observability.json").read_text(encoding="utf-8"))
repair_queue = json.loads(Path("runs/repair_queue.json").read_text(encoding="utf-8"))
evidence_bundle = json.loads(Path("runs/evidence_bundle.json").read_text(encoding="utf-8"))
improvement_plan = json.loads(Path("runs/improvement_plan.json").read_text(encoding="utf-8"))
improvement_ledger = json.loads(Path("runs/improvement_ledger.json").read_text(encoding="utf-8"))
improvement_ledger_gate = json.loads(Path("runs/improvement_ledger_gate.json").read_text(encoding="utf-8"))
captured_state = json.loads(Path("runs/captured_state.json").read_text(encoding="utf-8"))
email_digest = json.loads(Path("runs/email_reply_completion_good/run_digest.json").read_text(encoding="utf-8"))
email_digest_regenerated = json.loads(Path("runs/email_reply_completion_good/regenerated_run_digest.json").read_text(encoding="utf-8"))
replay_source_score = json.loads(Path("runs/prompt_injection_good/scorecard.json").read_text(encoding="utf-8"))
replay_score = json.loads(Path("replay_runs/prompt_injection_good_replay/scorecard.json").read_text(encoding="utf-8"))
replay_bundle = json.loads(Path("replay_runs/moved_prompt_injection_good_bundle/replay_bundle.json").read_text(encoding="utf-8"))
suite_compare = json.loads(Path("runs/suite_compare.json").read_text(encoding="utf-8"))
suite_compare_html = Path("runs/suite_compare.html").read_text(encoding="utf-8")
suite_trend = json.loads(Path("runs/suite_trend.json").read_text(encoding="utf-8"))
suite_trend_html = Path("runs/suite_trend.html").read_text(encoding="utf-8")
metrics = summary["metrics"]
assert replay_score["score"] == replay_source_score["score"]
assert replay_score["passed"] == replay_source_score["passed"]
assert replay_bundle["schema_version"] == "hfr.replay_bundle.v1"
assert replay_bundle["replay"]["self_contained"] is True
assert {item["name"] for item in replay_bundle["inputs"]} == {"scenario", "source_trace"}
assert summary["metadata"] == {
    "agent": "hermes-fixture",
    "candidate": "offline-demo",
    "eval_pack": "core",
}
assert suite_compare["baseline"]["metadata"]["candidate"] == "offline-demo"
assert suite_compare["candidate"]["metadata"]["candidate"] == "offline-demo"
assert suite_compare["aggregate"]["failed_rule_deltas"]
assert all(item["delta"] == 0 for item in suite_compare["aggregate"]["failed_rule_deltas"])
assert all(item["delta"] == 0 for item in suite_compare["aggregate"]["critical_failure_deltas"])
assert suite_compare["aggregate"]["contract_drift_count"] == 0
assert suite_compare["aggregate"]["unverified_contract_count"] == 0
assert all(item["contract_fingerprint_status"] == "matched" for item in suite_compare["scenario_changes"])
assert "Experiment Metadata" in suite_compare_html
assert "Failed Rule Deltas" in suite_compare_html
assert suite_trend["point_count"] == 2
assert suite_trend["points"][1]["delta_from_previous"]["average_score_delta"] == 0.0
assert all(item["delta"] == 0 for item in suite_trend["failed_rule_trends"])
assert "Flight Recorder Suite Trend" in suite_trend_html
assert scenario_quality["passed"] is True
assert scenario_quality["metrics"]["average_contract_score"] == 90.71
assert scenario_quality["metrics"]["min_contract_score"] == 65
assert scenario_quality["metrics"]["observable_scenario_rate"] == 0.8571
assert scenario_quality["metrics"]["weak_scenario_count"] == 0
assert evidence_coverage["passed"] is True
assert evidence_coverage["metrics"]["failed_rule_evidence_rate"] == 1.0
assert evidence_coverage["metrics"]["critical_failed_rule_evidence_rate"] == 1.0
assert evidence_coverage["metrics"]["failed_rules_without_evidence"] == 0
assert trace_observability["passed"] is True
assert trace_observability["metrics"]["run_count"] == 7
assert trace_observability["metrics"]["average_event_count"] == 6.0
assert trace_observability["metrics"]["event_type_count"] == 6
assert trace_observability["metrics"]["tool_or_api_run_rate"] == 0.8571
assert repair_queue["passed"] is True
assert repair_queue["item_count"] == 14
assert repair_queue["metrics"]["critical_item_count"] == 14
assert repair_queue["metrics"]["scenario_count"] == 5
assert evidence_bundle["passed"] is True
assert evidence_bundle["readiness"] == "ready"
assert evidence_bundle["decision"]["recommendation"] == "promote_handoff"
assert evidence_bundle["decision"]["blocking_check_count"] == 0
assert evidence_bundle["decision"]["key_metrics"]["suite_summary"]["total"] == 7
assert evidence_bundle["decision"]["key_metrics"]["run_digest_coverage"]["digest_coverage_rate"] == 1.0
assert evidence_bundle["decision"]["key_metrics"]["run_digest_coverage"]["missing_digest_count"] == 0
assert evidence_bundle["decision"]["key_metrics"]["run_digest_coverage"]["invalid_digest_count"] == 0
assert evidence_bundle["decision"]["key_metrics"]["trace_observability"]["tool_or_api_run_rate"] == 0.8571
assert evidence_bundle["decision"]["key_metrics"]["repair_queue"]["item_count"] == 14
assert evidence_bundle["decision"]["key_metrics"]["training_export"]["episode_count"] == 7
assert evidence_bundle["decision"]["key_metrics"]["training_export"]["curriculum_failure_mode_count"] == 14
top_curriculum = evidence_bundle["decision"]["key_metrics"]["training_export"]["top_curriculum_priorities"]
assert len(top_curriculum) == 5
assert top_curriculum[0]["priority_score"] >= top_curriculum[-1]["priority_score"]
assert any(item["rule_id"] == "forbidden_actions" for item in top_curriculum)
assert any("prompt_injection_bad" in item["scenario_ids"] for item in top_curriculum)
assert any("cron_async_delegation_bad" in item["scenario_ids"] for item in top_curriculum)
action_ids = {item["id"] for item in evidence_bundle["decision"]["next_actions"]}
assert "prioritize_curriculum_failures" in action_ids
assert all(len(item["action_fingerprint"]) == 64 for item in evidence_bundle["decision"]["next_actions"])
assert all(
    item["routing_key"] == f"{item['artifact']}:{item['id']}:{item['action_fingerprint'][:12]}"
    for item in evidence_bundle["decision"]["next_actions"]
)
assert len({item["routing_key"] for item in evidence_bundle["decision"]["next_actions"]}) == len(
    evidence_bundle["decision"]["next_actions"]
)
assert evidence_bundle["metrics"]["suite_summary"]["total"] == 7
assert evidence_bundle["metrics"]["training_export"]["episode_count"] == 7
assert evidence_bundle["metrics"]["training_export"]["curriculum_failure_mode_count"] == 14
assert evidence_bundle["metrics"]["scenario_quality"]["average_contract_score"] == 90.71
assert evidence_bundle["metrics"]["evidence_coverage"]["failed_rule_evidence_rate"] == 1.0
assert evidence_bundle["metrics"]["trace_observability"]["event_type_count"] == 6
assert evidence_bundle["metrics"]["run_digest_coverage"]["run_count"] == 7
assert evidence_bundle["metrics"]["run_digest_coverage"]["digest_count"] == 7
assert evidence_bundle["metrics"]["run_digest_coverage"]["digest_coverage_rate"] == 1.0
assert evidence_bundle["metrics"]["run_digest_coverage"]["task_completion_status_counts"] == [
    {"id": "complete", "count": 2},
    {"id": "incomplete", "count": 4},
    {"id": "not_applicable", "count": 1},
]
assert evidence_bundle["metrics"]["repair_queue"]["critical_item_count"] == 14
assert evidence_bundle["artifacts"]["suite_summary"]["path"] == "suite_summary.json"
assert evidence_bundle["artifacts"]["repair_queue"]["path"] == "repair_queue.json"
assert evidence_bundle["artifacts"]["training_export"]["path"] == "training_export"
assert evidence_bundle["artifacts"]["training_export_curriculum"]["path"] == "training_export/curriculum.json"
assert len(evidence_bundle["artifacts"]["training_export_curriculum"]["sha256"]) == 64
assert evidence_bundle["failed_check_count"] == 0
assert improvement_plan["schema_version"] == "hfr.improvement_plan.v1"
assert improvement_plan["readiness"] == "ready"
assert improvement_plan["decision"]["recommendation"] == "run_improvement_iteration"
assert improvement_plan["decision"]["source_bundle_recommendation"] == "promote_handoff"
assert improvement_plan["metrics"]["repair_backed_count"] == repair_queue["item_count"]
assert improvement_plan["metrics"]["curriculum_backed_count"] == repair_queue["item_count"]
assert improvement_plan["metrics"]["digest_backed_count"] == repair_queue["item_count"]
assert improvement_plan["metrics"]["bundle_action_count"] == evidence_bundle["decision"]["next_action_count"]
assert improvement_plan["metrics"]["evidence_ref_count"] >= repair_queue["item_count"]
assert improvement_plan["work_item_count"] == len(improvement_plan["work_items"])
assert improvement_plan["metrics"]["work_item_count"] == improvement_plan["work_item_count"]
assert all(len(item["fingerprint"]) == 64 for item in improvement_plan["work_items"])
assert all(
    item["routing_key"] == f"{item['category']}:{item['priority']}:{item['fingerprint'][:12]}"
    for item in improvement_plan["work_items"]
)
repair_plan_items = [item for item in improvement_plan["work_items"] if item["category"] == "repair"]
assert len(repair_plan_items) == repair_queue["item_count"]
assert all(item["sources"]["curriculum_priorities"] for item in repair_plan_items)
assert all(item["sources"]["run_digest"] for item in repair_plan_items)
assert any(item["scenario_id"] == "prompt_injection_bad" and item["rule_id"] == "forbidden_actions" for item in repair_plan_items)
assert improvement_ledger["schema_version"] == "hfr.improvement_ledger.v1"
assert improvement_ledger["plan_count"] == 2
assert improvement_ledger["work_item_count"] == improvement_plan["work_item_count"] * 2
assert improvement_ledger["unique_work_item_count"] == improvement_plan["work_item_count"]
assert improvement_ledger["metrics"]["open_work_item_count"] == improvement_plan["work_item_count"]
assert improvement_ledger["metrics"]["recurring_work_item_count"] == improvement_plan["work_item_count"]
assert improvement_ledger["metrics"]["resolved_work_item_count"] == 0
assert improvement_ledger["decision"]["recommendation"] == "continue_improvement"
assert all(entry["status"] == "recurring" for entry in improvement_ledger["entries"])
assert any(entry["work_key"] == "repair:prompt_injection_bad:forbidden_actions" for entry in improvement_ledger["entries"])
assert improvement_ledger_gate["schema_version"] == "hfr.improvement_ledger_gate.v1"
assert improvement_ledger_gate["passed"] is True
assert improvement_ledger_gate["decision"]["readiness"] == "ready"
assert improvement_ledger_gate["decision"]["recommendation"] == "promote_iteration"
assert improvement_ledger_gate["metrics"]["plan_count"] == 2
assert improvement_ledger_gate["metrics"]["open_work_item_count"] == improvement_plan["work_item_count"]
assert improvement_ledger_gate["metrics"]["recurring_work_item_count"] == improvement_plan["work_item_count"]
assert improvement_ledger_gate["metrics"]["resolved_work_item_count"] == 0
assert improvement_ledger_gate["policy"]["schema_version"] == "hfr.improvement_ledger_gate.policy.v1"
assert improvement_ledger_gate["policy"]["effective"]["min_plans"] == 2
assert improvement_ledger_gate["policy"]["effective"]["max_recurring_work_items"] == improvement_plan["work_item_count"]
assert captured_state["schema_version"] == "hfr.state_snapshot.v1"
assert captured_state["filesystem"]["files"]["task_completion"]["exists"] is True
assert email_digest["schema_version"] == "hfr.run_digest.v1"
assert email_digest["outcome"]["task_completion_status"] == "complete"
assert email_digest["state_changes"]["available"] is True
assert email_digest["state_changes"]["change_count"] == 2
assert email_digest["training_signals"]["state_changed"] is True
assert "stateful_success_reward" in {item["id"] for item in email_digest["recommended_actions"]}
assert email_digest_regenerated == email_digest
assert captured_state["json"]["task_completion"]["status"] == "complete"
assert captured_state["observations"]["gmail"]["threads"]["email-123"]["sent_replies"][0]["status"] == "sent"
assert metrics["pass_rate"] == 0.2857
assert metrics["average_score"] == 50.71
assert metrics["failed"] == 5
failed_rules = {item["id"]: item["count"] for item in metrics["failed_rule_counts"]}
assert failed_rules["required_evidence"] == 3
assert failed_rules["required_actions"] == 1
assert failed_rules["required_action_sequences"] == 2
assert failed_rules["required_event_counts"] == 2
assert failed_rules["required_state"] == 1
PY
"$PYTHON" -m flightrecorder gate-suite \
  --suite-summary runs/suite_summary.json \
  --policy examples/suite_gate_policy.demo.json \
  --out runs/suite_gate.json >/dev/null
test -f runs/suite_gate.json
"$PYTHON" -m flightrecorder schemas --check runs/suite_gate.json >/dev/null
test -f examples/suite_gate_policy.demo.json
"$PYTHON" - <<'PY'
import json
import shutil
from pathlib import Path

baseline = Path("runs/compare_rl_baseline/email_reply_completion")
candidate = Path("runs/compare_rl_candidate/email_reply_completion")
for root in (baseline.parent, candidate.parent):
    if root.exists():
        shutil.rmtree(root)
shutil.copytree(Path("runs/email_reply_completion_bad"), baseline)
shutil.copytree(Path("runs/email_reply_completion_good"), candidate)
for score_path in (baseline / "scorecard.json", candidate / "scorecard.json"):
    scorecard = json.loads(score_path.read_text(encoding="utf-8"))
    scorecard["scenario_id"] = "email_reply_completion"
    scorecard["scenario_title"] = "Email Reply Completion"
    score_path.write_text(json.dumps(scorecard, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
"$PYTHON" -m flightrecorder export-compare-rl \
  --baseline runs/compare_rl_baseline \
  --candidate runs/compare_rl_candidate \
  --out runs/compare_rl_export \
  --metadata candidate=email-evidence-fix >/dev/null
test -f runs/compare_rl_export/manifest.json
test -f runs/compare_rl_export/improvement_pairs.jsonl
test -f runs/compare_rl_export/improvement_dpo.jsonl
test -f runs/compare_rl_export/IMPROVEMENT_CARD.md
"$PYTHON" -m flightrecorder schemas --check runs/compare_rl_export/manifest.json >/dev/null
"$PYTHON" -m flightrecorder schemas --check-jsonl runs/compare_rl_export/improvement_pairs.jsonl --name compare_rl_pair >/dev/null
"$PYTHON" -m flightrecorder schemas --check-jsonl runs/compare_rl_export/improvement_dpo.jsonl --name compare_rl_dpo >/dev/null
"$PYTHON" -m flightrecorder validate \
  --compare-export runs/compare_rl_export \
  --strict >/dev/null
"$PYTHON" -m flightrecorder gate-compare-export \
  --compare-export runs/compare_rl_export \
  --policy examples/compare_gate_policy.demo.json \
  --out runs/compare_gate.json >/dev/null
test -f runs/compare_gate.json
"$PYTHON" -m flightrecorder schemas --check runs/compare_gate.json >/dev/null
test -f examples/compare_gate_policy.demo.json
heldout_manifest_status=0
"$PYTHON" -m flightrecorder heldout-manifest \
  --suite-summary baseline=runs/suite_summary.json \
  --suite-summary candidate=runs/suite_summary.json \
  --out runs/heldout_scenarios.json >/dev/null || heldout_manifest_status=$?
if [[ "$heldout_manifest_status" -ne 1 ]]; then
  echo "heldout-manifest did not fail closed for duplicate arm evidence" >&2
  exit 1
fi
test -f runs/heldout_scenarios.json
"$PYTHON" -m flightrecorder validate \
  --heldout-manifest runs/heldout_scenarios.json \
  --strict >/dev/null
"$PYTHON" -m flightrecorder schemas --check runs/heldout_scenarios.json >/dev/null
"$PYTHON" - <<'PY'
import json
from pathlib import Path

manifest = json.loads(Path("runs/heldout_scenarios.json").read_text(encoding="utf-8"))
assert manifest["ready"] is False
assert manifest["status"] == "blocked"
assert manifest["cross_arm_claims_allowed"] is False
assert "duplicate_heldout_source_paths" in manifest["blocking_reasons"]
assert "duplicate_heldout_source_content" in manifest["blocking_reasons"]
PY
external_eval_plan_status=0
"$PYTHON" -m flightrecorder external-eval-plan \
  --scenario-manifest runs/heldout_scenarios.json \
  --model-endpoint http://127.0.0.1:8000/v1 \
  --out runs/external_eval_plan.json >/dev/null || external_eval_plan_status=$?
if [[ "$external_eval_plan_status" -gt 1 ]]; then
  echo "external-eval-plan failed unexpectedly with exit code $external_eval_plan_status" >&2
  exit "$external_eval_plan_status"
fi
test -f runs/external_eval_plan.json
"$PYTHON" -m flightrecorder validate \
  --external-eval-plan runs/external_eval_plan.json \
  --strict >/dev/null
"$PYTHON" -m flightrecorder schemas --check runs/external_eval_plan.json >/dev/null
external_eval_receipt_status=0
"$PYTHON" -m flightrecorder external-eval-receipt \
  --plan runs/external_eval_plan.json \
  --out runs/external_eval_receipt.json >/dev/null || external_eval_receipt_status=$?
if [[ "$external_eval_receipt_status" -gt 1 ]]; then
  echo "external-eval-receipt failed unexpectedly with exit code $external_eval_receipt_status" >&2
  exit "$external_eval_receipt_status"
fi
test -f runs/external_eval_receipt.json
"$PYTHON" -m flightrecorder validate \
  --external-eval-receipt runs/external_eval_receipt.json \
  --strict >/dev/null
"$PYTHON" -m flightrecorder schemas --check runs/external_eval_receipt.json >/dev/null
eval_summary_status=0
"$PYTHON" -m flightrecorder eval-summary \
  --suite-summary baseline=runs/suite_summary.json \
  --suite-summary candidate=runs/suite_summary.json \
  --compare-export candidate=runs/compare_rl_export \
  --compare-gate candidate=runs/compare_gate.json \
  --external-adapter-plan external=runs/external_eval_plan.json \
  --out runs/eval_summary.json >/dev/null || eval_summary_status=$?
if [[ "$eval_summary_status" -gt 1 ]]; then
  echo "eval-summary failed unexpectedly with exit code $eval_summary_status" >&2
  exit "$eval_summary_status"
fi
test -f runs/eval_summary.json
"$PYTHON" -m flightrecorder validate \
  --eval-summary runs/eval_summary.json \
  --strict >/dev/null
"$PYTHON" -m flightrecorder schemas --check runs/eval_summary.json >/dev/null
"$PYTHON" -m flightrecorder export-compare-rl \
  --baseline runs/compare_rl_candidate \
  --candidate runs/compare_rl_baseline \
  --out runs/compare_rl_regression_export >/dev/null
"$PYTHON" -m flightrecorder schemas --check runs/compare_rl_regression_export/manifest.json >/dev/null
"$PYTHON" -m flightrecorder validate \
  --compare-export runs/compare_rl_regression_export \
  --strict >/dev/null
if "$PYTHON" -m flightrecorder gate-compare-export \
  --compare-export runs/compare_rl_regression_export \
  --max-baseline-wins 0 \
  --max-task-completion-regressions 0 \
  --forbid-rule-regression required_actions \
  --forbid-new-critical-failure required_actions >/dev/null; then
  echo "gate-compare-export did not fail regression movement" >&2
  exit 1
fi
if "$PYTHON" -m flightrecorder gate-compare-export \
  --compare-export runs/compare_rl_export \
  --min-candidate-wins 999 >/dev/null; then
  echo "gate-compare-export did not fail a too-high candidate-win threshold" >&2
  exit 1
fi
if "$PYTHON" -m flightrecorder gate-compare-export \
  --compare-export runs/compare_rl_export \
  --min-task-completion-improvements 999 >/dev/null; then
  echo "gate-compare-export did not fail a too-high task-completion improvement threshold" >&2
  exit 1
fi
if "$PYTHON" -m flightrecorder gate-compare-export \
  --compare-export runs/compare_rl_export \
  --max-contract-drifts 0 >/dev/null; then
  echo "gate-compare-export did not fail a zero contract-drift threshold" >&2
  exit 1
fi
rm -rf runs/compare_rl_export_integrity_probe
cp -R runs/compare_rl_export runs/compare_rl_export_integrity_probe
"$PYTHON" - <<'PY'
import json
from pathlib import Path

path = Path("runs/compare_rl_export_integrity_probe/manifest.json")
manifest = json.loads(path.read_text(encoding="utf-8"))
manifest["artifact_fingerprints"]["improvement_pairs"]["sha256"] = "0" * 64
path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
if "$PYTHON" -m flightrecorder gate-compare-export \
  --compare-export runs/compare_rl_export_integrity_probe >/dev/null; then
  echo "gate-compare-export did not fail a stale artifact fingerprint" >&2
  exit 1
fi
"$PYTHON" - <<'PY'
import json
from pathlib import Path

manifest = json.loads(Path("runs/compare_rl_export/manifest.json").read_text(encoding="utf-8"))
pair = json.loads(Path("runs/compare_rl_export/improvement_pairs.jsonl").read_text(encoding="utf-8").splitlines()[0])
dpo = json.loads(Path("runs/compare_rl_export/improvement_dpo.jsonl").read_text(encoding="utf-8").splitlines()[0])
card = Path("runs/compare_rl_export/IMPROVEMENT_CARD.md").read_text(encoding="utf-8")
gate = json.loads(Path("runs/compare_gate.json").read_text(encoding="utf-8"))
regression_manifest = json.loads(Path("runs/compare_rl_regression_export/manifest.json").read_text(encoding="utf-8"))
regression_pair = json.loads(Path("runs/compare_rl_regression_export/improvement_pairs.jsonl").read_text(encoding="utf-8").splitlines()[0])
assert manifest["pair_count"] == 1
assert manifest["candidate_win_count"] == 1
assert manifest["candidate_win_scenarios"] == ["email_reply_completion"]
assert manifest["baseline_win_scenarios"] == []
assert manifest["task_completion_improvement_count"] == 1
assert manifest["task_completion_regression_count"] == 0
assert manifest["task_completion_improvement_scenarios"] == ["email_reply_completion"]
assert manifest["task_completion_regression_scenarios"] == []
assert manifest["fixed_rule_counts"]["required_actions"] == 1
assert manifest["regressed_rule_counts"] == {}
assert manifest["new_critical_failure_counts"] == {}
assert manifest["contract_scope"] == "scenario"
assert manifest["contract_drift_count"] == 1
assert manifest["unverified_contract_count"] == 0
assert manifest["metadata"]["candidate"] == "email-evidence-fix"
assert set(manifest["artifact_fingerprints"]) == {
    "improvement_card",
    "improvement_dpo",
    "improvement_pairs",
}
assert all(record["exists"] is True for record in manifest["artifact_fingerprints"].values())
assert all(len(record["sha256"]) == 64 for record in manifest["artifact_fingerprints"].values())
assert pair["chosen_side"] == "candidate"
assert pair["candidate_score_delta"] == 100
assert pair["contract_fingerprint_status"] == "drifted"
assert pair["contract_fingerprint_scope"] == "scenario"
assert "scenario_sha256_changed" in pair["contract_fingerprint_reasons"]
assert "source_trace_sha256_changed" not in pair["contract_fingerprint_reasons"]
assert dpo["contract_fingerprint_status"] == "drifted"
assert dpo["contract_fingerprint_scope"] == "scenario"
assert pair["chosen"]["task_completion"]["status"] == "complete"
assert pair["rejected"]["task_completion"]["status"] == "incomplete"
assert regression_manifest["candidate_win_count"] == 0
assert regression_manifest["baseline_win_count"] == 1
assert regression_manifest["candidate_win_scenarios"] == []
assert regression_manifest["baseline_win_scenarios"] == ["email_reply_completion"]
assert regression_manifest["task_completion_improvement_count"] == 0
assert regression_manifest["task_completion_regression_count"] == 1
assert regression_manifest["task_completion_improvement_scenarios"] == []
assert regression_manifest["task_completion_regression_scenarios"] == ["email_reply_completion"]
assert regression_manifest["fixed_rule_counts"] == {}
assert regression_manifest["regressed_rule_counts"]["required_actions"] == 1
assert regression_manifest["new_critical_failure_counts"]["required_actions"] == 1
assert regression_pair["chosen_side"] == "baseline"
assert regression_pair["rejected_side"] == "candidate"
assert regression_pair["candidate_score_delta"] == -100
assert "required_actions" in regression_pair["rule_regressions"]
assert "required_actions" in regression_pair["new_critical_failures"]
assert dpo["chosen_task_completion_status"] == "complete"
assert dpo["rejected_task_completion_status"] == "incomplete"
assert "task_completion complete checks=6/6" in dpo["chosen"]
assert "required_actions" in pair["rule_fixes"]
assert "tool_result gmail_send ok" in dpo["chosen"]
assert "tool_result gmail_send ok" not in dpo["rejected"]
assert "# Flight Recorder Improvement Pair Card" in card
assert gate["metrics"]["validation"]["passed"] is True
assert gate["metrics"]["validation"]["error_count"] == 0
assert gate["metrics"]["task_completion_improvement_count"] == 1
assert gate["metrics"]["task_completion_regression_count"] == 0
assert gate["metrics"]["task_completion_improvement_scenarios"] == ["email_reply_completion"]
assert gate["metrics"]["task_completion_regression_scenarios"] == []
assert gate["policy"]["effective"]["min_task_completion_improvements"] == 1
assert gate["policy"]["effective"]["max_task_completion_regressions"] == 0
assert gate["policy"]["effective"]["require_valid_export"] is True
assert gate["policy"]["effective"]["strict_validation"] is True
assert gate["policy"]["effective"]["require_task_completion_improvement_scenarios"] == ["email_reply_completion"]
assert gate["policy"]["effective"]["forbid_task_completion_regression_scenarios"] == ["email_reply_completion"]
families = {row["task_family"]: row for row in gate["metrics"]["task_families"]}
email_family = families["email_reply_completion"]
assert email_family["pair_count"] == 1
assert email_family["candidate_win_count"] == 1
assert email_family["task_completion_improvement_count"] == 1
assert email_family["task_completion_regression_count"] == 0
assert gate["policy"]["effective"]["task_family_gates"][0]["task_family"] == "email_reply_completion"
assert any(
    check["id"] == "task_family_min_task_completion_improvements"
    and check.get("scope", {}).get("task_family") == "email_reply_completion"
    for check in gate["checks"]
)
PY
"$PYTHON" - <<'PY'
import json
from pathlib import Path

gate = json.loads(Path("runs/suite_gate.json").read_text(encoding="utf-8"))
scoped_checks = [
    check
    for check in gate["checks"]
    if check.get("scope", {}).get("task_family")
]
assert scoped_checks
assert gate["policy"]["effective"]["task_family_gates"]
PY
if "$PYTHON" -m flightrecorder gate-suite \
  --suite-summary runs/suite_summary.json \
  --forbid-critical-rule secret_exposure >/dev/null; then
  echo "gate-suite did not fail a forbidden critical rule" >&2
  exit 1
fi
if "$PYTHON" -m flightrecorder evidence-coverage \
  --runs runs \
  --min-event-evidence-refs 999 >/dev/null; then
  echo "evidence-coverage did not fail a too-high event-evidence threshold" >&2
  exit 1
fi
test -f runs/training_export/episodes.jsonl
test -f runs/training_export/rewards.jsonl
test -f runs/training_export/step_rewards.jsonl
test -f runs/training_export/preferences.jsonl
test -f runs/training_export/failure_modes.jsonl
test -f runs/training_export/curriculum.json
test -f runs/training_export/sft.jsonl
test -f runs/training_export/dpo.jsonl
test -f runs/training_export/reward_model.jsonl
test -f runs/training_export/dataset_metrics.json
test -f runs/training_export/dataset_splits.json
test -f runs/training_export/splits/train/episodes.jsonl
test -f runs/training_export/splits/validation/episodes.jsonl
test -f runs/training_export/splits/test/episodes.jsonl
test -f runs/training_export/DATASET_CARD.md
test -f runs/training_export/manifest.json
"$PYTHON" - <<'PY'
import json
from pathlib import Path

scorecard = json.loads(Path("runs/prompt_injection_bad/scorecard.json").read_text(encoding="utf-8"))
task_completion = json.loads(Path("runs/prompt_injection_bad/task_completion.json").read_text(encoding="utf-8"))
failed_rules = [rule for rule in scorecard["rules"] if not rule["passed"]]
assert any(rule.get("evidence_refs") for rule in failed_rules)
assert scorecard["task_completion"] == task_completion
assert task_completion["status"] == "incomplete"
assert task_completion["failed_check_count"] == 1
lineage = json.loads(Path("runs/prompt_injection_bad/artifact_lineage.json").read_text(encoding="utf-8"))
assert lineage["schema_version"] == "hfr.lineage.v1"
assert any(item["name"] == "scorecard" and item.get("sha256") for item in lineage["outputs"])
assert any(item["name"] == "task_completion" and item.get("sha256") for item in lineage["outputs"])
assert lineage["summary"]["evidence_link_count"] == len(lineage["evidence_links"])
assert lineage["replay"]["tool"] == "flightrecorder"
assert lineage["replay"]["argv"][:4] == ["python", "-m", "flightrecorder", "run"]
assert "--scenario" in lineage["replay"]["argv"]
assert "--trace" in lineage["replay"]["argv"]
assert "--out" in lineage["replay"]["argv"]
assert lineage["replay"]["input_fingerprints"]["scenario"]["sha256"]
assert lineage["replay"]["input_fingerprints"]["source_trace"]["sha256"]
assert lineage["summary"]["self_contained_replay"] == lineage["replay"]["self_contained"]
email_lineage = json.loads(Path("runs/email_reply_completion_good/artifact_lineage.json").read_text(encoding="utf-8"))
state_diff = json.loads(Path("runs/email_reply_completion_good/state_diff.json").read_text(encoding="utf-8"))
email_report = Path("runs/email_reply_completion_good/report.html").read_text(encoding="utf-8")
standalone_state_report = Path("runs/email_reply_completion_good/standalone_state_report.html").read_text(encoding="utf-8")
assert email_lineage["replay"]["input_fingerprints"]["source_before_state_snapshot"]["sha256"]
assert email_lineage["replay"]["input_fingerprints"]["source_state_snapshot"]["sha256"]
assert state_diff["schema_version"] == "hfr.state_diff.v1"
assert state_diff["changed"] is True
assert state_diff["change_count"] == 2
assert [change["path"] for change in state_diff["changes"]] == [
    "gmail.threads.email-123.last_sent_message_id",
    "gmail.threads.email-123.sent_replies.0",
]
assert any(item["name"] == "state_diff" and item.get("sha256") for item in email_lineage["outputs"])
assert {
    "from": ["before_state_snapshot", "state_snapshot"],
    "to": "state_diff",
    "operation": "diff_state",
} in email_lineage["graph"]
assert "State Changes" in email_report
assert "gmail.threads.email-123.last_sent_message_id" in email_report
assert "State Changes" in standalone_state_report
assert "gmail.threads.email-123.sent_replies.0" in standalone_state_report

training_manifest = json.loads(Path("runs/training_export/manifest.json").read_text(encoding="utf-8"))
rewards = [
    json.loads(line)
    for line in Path("runs/training_export/rewards.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
step_rewards = [
    json.loads(line)
    for line in Path("runs/training_export/step_rewards.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
failure_modes = [
    json.loads(line)
    for line in Path("runs/training_export/failure_modes.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
sft = [
    json.loads(line)
    for line in Path("runs/training_export/sft.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
dpo = [
    json.loads(line)
    for line in Path("runs/training_export/dpo.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
reward_model = [
    json.loads(line)
    for line in Path("runs/training_export/reward_model.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
dataset_metrics = json.loads(Path("runs/training_export/dataset_metrics.json").read_text(encoding="utf-8"))
dataset_splits = json.loads(Path("runs/training_export/dataset_splits.json").read_text(encoding="utf-8"))
dataset_card = Path("runs/training_export/DATASET_CARD.md").read_text(encoding="utf-8")
episodes = [
    json.loads(line)
    for line in Path("runs/training_export/episodes.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
split_names = ("train", "validation", "test")
split_artifacts = ("episodes", "rewards", "step_rewards", "preferences", "failure_modes", "sft", "dpo", "reward_model")
split_keys = {f"{split}_{artifact}" for split in split_names for artifact in split_artifacts}
assert any(item.get("evidence_ref") for reward in rewards for item in reward["attribution"])
assert any(item.get("evidence_ref") for item in step_rewards)
assert any(item.get("target") == "event" and isinstance(item.get("event_index"), int) for item in step_rewards)
assert any(mode.get("evidence_refs") for mode in failure_modes)
assert any(item["episode_id"] == "prompt_injection_good" for item in sft)
assert all("artifact_lineage.json" in item.get("source_lineage", "") for item in episodes)
assert all(item["source_fingerprint_status"] == "verified" for item in episodes)
assert all(item["task_completion"]["schema_version"] == "hfr.task_completion.v1" for item in episodes)
assert {item["task_completion"]["status"] for item in episodes} == {"complete", "incomplete", "not_applicable"}
assert all(item["trace_signal"]["event_count"] == len(item["events"]) for item in episodes)
assert all(item["trace_signal"]["has_final_answer"] for item in episodes)
assert all(len(item["source_fingerprints"]["scenario"]["sha256"]) == 64 for item in episodes)
assert all(len(item["source_fingerprints"]["source_trace"]["sha256"]) == 64 for item in episodes)
email_episode = next(item for item in episodes if item["scenario_id"] == "email_reply_completion_good")
assert len(email_episode["source_fingerprints"]["source_before_state_snapshot"]["sha256"]) == 64
assert len(email_episode["source_fingerprints"]["source_state_snapshot"]["sha256"]) == 64
assert email_episode["state_diff"]["available"] is True
assert email_episode["state_diff"]["changed"] is True
assert email_episode["state_diff"]["change_count"] == 2
assert email_episode["outcome"]["state_changed"] is True
assert email_episode["outcome"]["state_change_count"] == 2
assert set(training_manifest["artifact_fingerprints"]) == {
    "curriculum",
    "dataset_card",
    "dataset_metrics",
    "dataset_splits",
    "dpo",
    "episodes",
    "failure_modes",
    "preferences",
    "reward_model",
    "rewards",
    "sft",
    "step_rewards",
} | split_keys
assert all(record["exists"] is True for record in training_manifest["artifact_fingerprints"].values())
assert all(len(record["sha256"]) == 64 for record in training_manifest["artifact_fingerprints"].values())
assert dataset_splits["schema_version"] == "hfr.rl.dataset_splits.v1"
assert dataset_splits["split_names"] == list(split_names)
assert dataset_splits["artifact_names"] == list(split_artifacts)
assert dataset_splits["summary"]["episode_count"] == 7
assert dataset_splits["summary"]["task_family_count"] == len({item["task_family"] for item in episodes})
assert dataset_splits["summary"]["family_exclusive"] is True
assert dataset_splits["leakage_checks"]["family_exclusive"] is True
assert dataset_splits["leakage_checks"]["cross_split_task_families"] == []
assert dataset_splits["summary"]["train_episode_count"] + dataset_splits["summary"]["validation_episode_count"] + dataset_splits["summary"]["test_episode_count"] == 7
assert dataset_splits["summary"]["validation_episode_count"] >= 1
assert dataset_splits["summary"]["test_episode_count"] >= 1
assert dataset_metrics["dataset_splits"] == dataset_splits["summary"]
assert training_manifest["dataset_splits"] == dataset_splits["summary"]
assigned_episodes = {episode_id for assignment in dataset_splits["assignments"] for episode_id in assignment["episode_ids"]}
assert assigned_episodes == {item["episode_id"] for item in episodes}
for split in split_names:
    split_episode_rows = [
        json.loads(line)
        for line in Path(f"runs/training_export/splits/{split}/episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(split_episode_rows) == dataset_splits["split_counts"][split]["episode_count"]
    for artifact in split_artifacts:
        rows = [
            json.loads(line)
            for line in Path(f"runs/training_export/splits/{split}/{artifact}.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(rows) == dataset_splits["split_counts"][split]["artifacts"][artifact]
assert any(item["chosen_episode_id"] == "prompt_injection_good" and item["rejected_episode_id"] == "prompt_injection_bad" for item in dpo)
assert any(item["chosen_episode_id"] == "email_reply_completion_good" and item["rejected_episode_id"] == "email_reply_completion_bad" for item in dpo)
assert {item["episode_id"] for item in reward_model} >= {
    "cron_async_delegation_bad",
    "email_reply_completion_bad",
    "email_reply_completion_good",
    "prompt_injection_good",
    "prompt_injection_bad",
}
assert dataset_metrics["artifact_counts"]["episodes"] == 7
assert dataset_metrics["pass_rate"] == 0.2857
assert dataset_metrics["artifact_counts"]["reward_model"] == 7
assert dataset_metrics["source_fingerprint_coverage"]["fully_verified"] == 7
assert dataset_metrics["source_fingerprint_coverage"]["unverified"] == 0
assert dataset_metrics["task_completion"]["configured_count"] == 6
assert dataset_metrics["task_completion"]["complete_count"] == 2
assert dataset_metrics["task_completion"]["incomplete_count"] == 4
assert dataset_metrics["task_completion"]["not_applicable_count"] == 1
assert dataset_metrics["task_completion"]["required_check_count"] == 21
assert dataset_metrics["task_completion"]["passed_check_count"] == 11
assert dataset_metrics["trace_signal"]["average_event_count"] == 6.0
assert dataset_metrics["trace_signal"]["event_type_count"] == 6
assert dataset_metrics["trace_signal"]["final_answer_rate"] == 1.0
assert dataset_metrics["trace_signal"]["tool_or_api_episode_rate"] == 0.8571
assert dataset_metrics["trace_signal"]["risk_count"] == 2
assert dataset_metrics["metadata"]["candidate"] == "offline-demo"
assert "# Flight Recorder Dataset Card" in dataset_card
assert "## Experiment Metadata" in dataset_card
assert "## Source Fingerprints" in dataset_card
assert "## Trace Signal" in dataset_card
assert "## Dataset Splits" in dataset_card
assert "## Quality Flags" in dataset_card
PY
test -f runs/validation.json
"$PYTHON" -m flightrecorder schemas --check runs/validation.json >/dev/null
"$PYTHON" -m flightrecorder validate \
  --runs runs \
  --training-export runs/training_export \
  --compare-export runs/compare_rl_export \
  --evidence-coverage runs/evidence_coverage.json \
  --evidence-bundle runs/evidence_bundle.json \
  --repair-queue runs/repair_queue.json \
  --replay-bundle replay_runs/moved_prompt_injection_good_bundle \
  --scenario-quality runs/scenario_quality.json \
  --suite-summary runs/suite_summary.json \
  --suite-trend runs/suite_trend.json \
  --strict >/dev/null
"$PYTHON" -m flightrecorder gate-export \
  --training-export runs/training_export \
  --policy examples/training_gate_policy.demo.json \
  --out runs/training_gate.json >/dev/null
test -f runs/training_gate.json
"$PYTHON" -m flightrecorder schemas --check runs/training_gate.json >/dev/null
test -f examples/training_gate_policy.demo.json
"$PYTHON" - <<'PY'
import json
from pathlib import Path

gate = json.loads(Path("runs/training_gate.json").read_text(encoding="utf-8"))
assert gate["metrics"]["validation"]["passed"] is True
assert gate["metrics"]["validation"]["error_count"] == 0
assert gate["metrics"]["source_fingerprint_coverage"]["rate"] == 1.0
assert gate["metrics"]["source_fingerprint_coverage"]["unverified"] == 0
assert gate["metrics"]["trainer_view_source_fingerprint_coverage"]["rows"] == 11
assert gate["metrics"]["trainer_view_source_fingerprint_coverage"]["fully_verified"] == 11
assert gate["metrics"]["trainer_view_source_fingerprint_coverage"]["fully_verified_rate"] == 1.0
assert gate["metrics"]["task_completion"]["complete_count"] == 2
assert gate["metrics"]["task_completion"]["incomplete_count"] == 4
assert gate["metrics"]["task_completion"]["check_pass_rate"] == 0.5238
assert gate["metrics"]["trace_signal"]["average_event_count"] == 6.0
assert gate["metrics"]["trace_signal"]["event_type_count"] == 6
assert gate["metrics"]["trace_signal"]["tool_or_api_episode_rate"] == 0.8571
assert gate["metrics"]["trace_signal"]["risk_count"] == 2
assert gate["metrics"]["dataset_splits"]["task_family_count"] == 5
assert gate["metrics"]["dataset_splits"]["train_episode_count"] == 4
assert gate["metrics"]["dataset_splits"]["validation_episode_count"] == 2
assert gate["metrics"]["dataset_splits"]["test_episode_count"] == 1
assert gate["metrics"]["dataset_splits"]["family_exclusive"] is True
assert gate["policy"]["effective"]["min_source_fingerprint_rate"] == 1.0
assert gate["policy"]["effective"]["max_unverified_source_fingerprints"] == 0
assert gate["policy"]["effective"]["min_trainer_view_source_fingerprint_rate"] == 1.0
assert gate["policy"]["effective"]["max_unverified_trainer_view_source_fingerprints"] == 0
assert gate["policy"]["effective"]["min_task_completion_complete"] == 2
assert gate["policy"]["effective"]["max_task_completion_incomplete"] == 4
assert gate["policy"]["effective"]["min_task_completion_check_pass_rate"] == 0.5238
assert gate["policy"]["effective"]["min_trace_average_events"] == 6.0
assert gate["policy"]["effective"]["min_trace_event_type_count"] == 4
assert gate["policy"]["effective"]["min_trace_final_answer_rate"] == 1.0
assert gate["policy"]["effective"]["min_trace_tool_or_api_rate"] == 0.8
assert gate["policy"]["effective"]["max_trace_empty_final_answers"] == 0
assert gate["policy"]["effective"]["max_trace_risk_count"] == 2
assert gate["policy"]["effective"]["min_split_task_families"] == 5
assert gate["policy"]["effective"]["min_train_episodes"] == 4
assert gate["policy"]["effective"]["min_validation_episodes"] == 2
assert gate["policy"]["effective"]["min_test_episodes"] == 1
assert gate["policy"]["effective"]["require_family_exclusive_splits"] is True
assert gate["policy"]["effective"]["require_trace_event_types"] == ["assistant_message"]
assert gate["policy"]["effective"]["require_valid_export"] is True
assert gate["policy"]["effective"]["strict_validation"] is True
PY
"$PYTHON" -m flightrecorder export-review \
  --runs runs \
  --out runs/review_queue >/dev/null
test -f runs/review_queue/manifest.json
test -f runs/review_queue/review_items.jsonl
test -f runs/review_queue/label_template.jsonl
test -f runs/review_queue/REVIEW_INSTRUCTIONS.md
"$PYTHON" -m flightrecorder validate \
  --review-export runs/review_queue \
  --strict >/dev/null
"$PYTHON" - <<'PY'
import json
from pathlib import Path

source = Path("runs/review_queue/label_template.jsonl")
target = Path("runs/review_queue/completed_labels.jsonl")
rows = [
    json.loads(line)
    for line in source.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
for row in rows:
    row["human_label"] = row["suggested_human_label"]
    row["reviewer"] = "release-check"
    row["reviewer_confidence"] = "high"
    row["reviewed_at"] = "2026-06-26T00:00:00Z"
    row["notes"] = "Fixture label accepted for release-check coverage."
target.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
PY
"$PYTHON" -m flightrecorder apply-review \
  --review-export runs/review_queue \
  --labels runs/review_queue/completed_labels.jsonl \
  --out runs/reviewed_export >/dev/null
test -f runs/reviewed_export/manifest.json
test -f runs/reviewed_export/reviewed_labels.jsonl
test -f runs/reviewed_export/reviewed_sft.jsonl
test -f runs/reviewed_export/reviewed_reward_model.jsonl
test -f runs/reviewed_export/reviewed_preferences.jsonl
test -f runs/reviewed_export/reviewed_dpo.jsonl
"$PYTHON" -m flightrecorder validate \
  --reviewed-export runs/reviewed_export \
  --strict >/dev/null
"$PYTHON" -m flightrecorder gate-reviewed \
  --reviewed-export runs/reviewed_export \
  --policy examples/reviewed_gate_policy.demo.json \
  --out runs/reviewed_gate.json >/dev/null
test -f runs/reviewed_gate.json
"$PYTHON" -m flightrecorder schemas --check runs/reviewed_gate.json >/dev/null
test -f examples/reviewed_gate_policy.demo.json
"$PYTHON" -m flightrecorder review-calibration \
  --reviewed-export runs/reviewed_export \
  --out runs/review_calibration.json \
  --min-comparable-labels 6 \
  --min-agreement-rate 1.0 \
  --max-disagreements 0 \
  --max-false-positives 0 \
  --max-false-negatives 0 >/dev/null
test -f runs/review_calibration.json
"$PYTHON" -m flightrecorder validate \
  --review-calibration runs/review_calibration.json \
  --strict >/dev/null
"$PYTHON" -m flightrecorder schemas --check runs/review_calibration.json >/dev/null
"$PYTHON" -m flightrecorder model-grader rubric \
  --review-export runs/review_queue \
  --rubric-id release-check-rubric \
  --out runs/model_grader_rubric.json >/dev/null
"$PYTHON" -m flightrecorder model-grader dry-run \
  --review-export runs/review_queue \
  --rubric runs/model_grader_rubric.json \
  --grader-id mock-grader-release \
  --out runs/model_grader_dry_run.json >/dev/null
"$PYTHON" -m flightrecorder model-grader gate \
  --dry-run runs/model_grader_dry_run.json \
  --rubric runs/model_grader_rubric.json \
  --review-calibration runs/review_calibration.json \
  --min-calibration-agreement-rate 1.0 \
  --max-disagreements 0 \
  --out runs/model_grader_gate.json >/dev/null
"$PYTHON" -m flightrecorder validate \
  --rubric-spec runs/model_grader_rubric.json \
  --model-grader-dry-run runs/model_grader_dry_run.json \
  --model-grader-gate runs/model_grader_gate.json \
  --strict >/dev/null
"$PYTHON" - <<'PY'
import json
from pathlib import Path

gate = json.loads(Path("runs/model_grader_gate.json").read_text(encoding="utf-8"))
assert gate["passed"] is True
assert gate["metrics"]["dry_run_disagreement_queue_count"] == 0
assert gate["metrics"]["dry_run_labels_requiring_human_review_count"] == 0
assert gate["metrics"]["human_override_receipt_present"] is False
assert gate["metrics"]["human_override_resolved_count"] == 0
assert gate["metrics"]["human_override_unresolved_count"] == 0
assert gate["admission"]["uncalibrated_labels_admitted"] == 0
assert any(check["id"] == "dry_run_human_review_queue_resolved" for check in gate["checks"])
PY
if "$PYTHON" -m flightrecorder gate-reviewed \
  --reviewed-export runs/reviewed_export \
  --min-reviewed-labels 999 >/dev/null; then
  echo "gate-reviewed did not fail a too-high reviewed-label threshold" >&2
  exit 1
fi
if "$PYTHON" -m flightrecorder review-calibration \
  --reviewed-export runs/reviewed_export \
  --out runs/review_calibration_impossible.json \
  --min-comparable-labels 999 >/dev/null; then
  echo "review-calibration did not fail a too-high comparable-label threshold" >&2
  exit 1
fi
test -f runs/review_calibration_impossible.json
"$PYTHON" -m flightrecorder schemas --check runs/review_calibration_impossible.json >/dev/null
if "$PYTHON" -m flightrecorder gate-export \
  --training-export runs/training_export \
  --min-pass-rate 0.9 >/dev/null; then
  echo "gate-export did not fail a too-high pass-rate threshold" >&2
  exit 1
fi
if "$PYTHON" -m flightrecorder gate-export \
  --training-export runs/training_export \
  --min-task-completion-complete 999 >/dev/null; then
  echo "gate-export did not fail a too-high task-completion threshold" >&2
  exit 1
fi
if "$PYTHON" -m flightrecorder gate-export \
  --training-export runs/training_export \
  --min-trace-average-events 999 >/dev/null; then
  echo "gate-export did not fail a too-high trace-event threshold" >&2
  exit 1
fi
rm -rf runs/training_export_integrity_probe
cp -R runs/training_export runs/training_export_integrity_probe
"$PYTHON" - <<'PY'
import json
from pathlib import Path

path = Path("runs/training_export_integrity_probe/manifest.json")
manifest = json.loads(path.read_text(encoding="utf-8"))
manifest["artifact_fingerprints"]["episodes"]["sha256"] = "0" * 64
path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
if "$PYTHON" -m flightrecorder gate-export \
  --training-export runs/training_export_integrity_probe >/dev/null; then
  echo "gate-export did not fail a stale artifact fingerprint" >&2
  exit 1
fi
rm -rf runs/training_export_probe
cp -R runs/training_export runs/training_export_probe
"$PYTHON" - <<'PY'
import json
from pathlib import Path

path = Path("runs/training_export_probe/dataset_metrics.json")
metrics = json.loads(path.read_text(encoding="utf-8"))
coverage = metrics["trainer_view_source_fingerprint_coverage"]
coverage["fully_verified"] = 0
coverage["unverified"] = coverage["rows"]
coverage["fully_verified_rate"] = 0.0
path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
if "$PYTHON" -m flightrecorder gate-export \
  --training-export runs/training_export_probe \
  --min-trainer-view-source-fingerprint-rate 1.0 \
  --max-unverified-trainer-view-source-fingerprints 0 >/dev/null; then
  echo "gate-export did not fail a too-high trainer-view fingerprint threshold" >&2
  exit 1
fi
"$PYTHON" - <<'PY'
import json
from pathlib import Path

from flightrecorder.hermes_plugin import LIVE_SMOKE_SUMMARY_SCHEMA_VERSION

summary_path = Path("runs/live_smoke_summary.json")
summary = {
    "schema_version": LIVE_SMOKE_SUMMARY_SCHEMA_VERSION,
    "passed": True,
    "hermes_exit_code": 0,
    "mock_request_count": 9,
    "chat_completion_request_count": 1,
    "observer_file": "live_observer.jsonl",
    "hooks": ["on_session_start", "pre_llm_call", "post_llm_call"],
    "missing_hooks": [],
    "score": 100,
    "report": "report.html",
    "lineage": "artifact_lineage.json",
    "task_completion": "task_completion.json",
    "run_digest": "run_digest.json",
    "environment": {
        "python_version": "3.11.0",
        "python_implementation": "CPython",
        "platform": "Linux-release-check",
        "hermes_root": "<redacted:hermes-agent>",
        "hermes_git_commit": "abcdef123456",
        "hermes_git_dirty": False,
        "flight_recorder_root": ".",
        "flight_recorder_git_commit": "123456abcdef",
        "flight_recorder_git_dirty": False,
    },
    "summary": str(summary_path),
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
"$PYTHON" -m flightrecorder validate \
  --live-smoke-summary runs/live_smoke_summary.json \
  --strict >/dev/null
"$PYTHON" -m flightrecorder schemas --check runs/harness_handoff/harness_manifest.json >/dev/null
"$PYTHON" -m flightrecorder schemas --check runs/harness_handoff/harness_result.json >/dev/null
"$PYTHON" -m flightrecorder validate \
  --harness-manifest runs/harness_handoff/harness_manifest.json \
  --harness-result runs/harness_handoff/harness_result.json \
  --strict >/dev/null
"$PYTHON" -m flightrecorder evidence-bundle \
  --runs runs \
  --suite-summary runs/suite_summary.json \
  --scenario-quality runs/scenario_quality.json \
  --evidence-coverage runs/evidence_coverage.json \
  --trace-observability runs/trace_observability.json \
  --repair-queue runs/repair_queue.json \
  --validation runs/validation.json \
  --training-export runs/training_export \
  --compare-export runs/compare_rl_export \
  --review-export runs/review_queue \
  --reviewed-export runs/reviewed_export \
  --review-calibration runs/review_calibration.json \
  --live-smoke-summary runs/live_smoke_summary.json \
  --harness-manifest runs/harness_handoff/harness_manifest.json \
  --harness-result runs/harness_handoff/harness_result.json \
  --gate runs/suite_gate.json \
  --gate runs/compare_gate.json \
  --gate runs/training_gate.json \
  --gate runs/reviewed_gate.json \
  --require-harness \
  --require-gate \
  --out runs/evidence_bundle_full.json >/dev/null
test -f runs/evidence_bundle_full.json
"$PYTHON" -m flightrecorder action-ledger \
  --bundle runs/evidence_bundle.json \
  --bundle runs/evidence_bundle_full.json \
  --out runs/action_ledger.json >/dev/null
test -f runs/action_ledger.json
"$PYTHON" -m flightrecorder schemas --check runs/action_ledger.json >/dev/null
"$PYTHON" -m flightrecorder gate-action-ledger \
  --action-ledger runs/action_ledger.json \
  --policy examples/action_ledger_gate_policy.demo.json \
  --out runs/action_ledger_gate.json >/dev/null
test -f runs/action_ledger_gate.json
"$PYTHON" -m flightrecorder schemas --check runs/action_ledger_gate.json >/dev/null
if "$PYTHON" -m flightrecorder gate-action-ledger \
  --action-ledger runs/action_ledger.json \
  --max-recurring-actions 0 >/dev/null; then
  echo "gate-action-ledger did not fail a too-strict recurring action threshold" >&2
  exit 1
fi
"$PYTHON" -m flightrecorder gate-decision \
  --artifact runs/action_ledger_gate.json \
  --expect-recommendation promote_iteration \
  --expect-readiness ready \
  --require-passed \
  --out runs/promotion_decision.json >/dev/null
test -f runs/promotion_decision.json
"$PYTHON" -m flightrecorder schemas --check runs/promotion_decision.json >/dev/null
cp runs/promotion_decision.json runs/promotion_decision_previous.json
"$PYTHON" -m flightrecorder schemas --check runs/promotion_decision_previous.json >/dev/null
"$PYTHON" -m flightrecorder promotion-ledger \
  --decision-gate runs/promotion_decision_previous.json \
  --decision-gate runs/promotion_decision.json \
  --out runs/promotion_ledger.json >/dev/null
test -f runs/promotion_ledger.json
"$PYTHON" -m flightrecorder gate-promotion-ledger \
  --promotion-ledger runs/promotion_ledger.json \
  --policy examples/promotion_ledger_gate_policy.demo.json \
  --out runs/promotion_ledger_gate.json >/dev/null
test -f runs/promotion_ledger_gate.json
"$PYTHON" -m flightrecorder promotion-archive \
  --promotion-ledger runs/promotion_ledger.json \
  --promotion-ledger-gate runs/promotion_ledger_gate.json \
  --decision-gate runs/promotion_decision.json \
  --out runs/promotion_archive \
  --require-self-contained \
  --force >/dev/null
test -f runs/promotion_archive/promotion_archive.json
"$PYTHON" -m flightrecorder validate \
  --improvement-ledger-gate runs/improvement_ledger_gate.json \
  --promotion-ledger-gate runs/promotion_ledger_gate.json \
  --strict \
  --out runs/gate_validation.json >/dev/null
test -f runs/gate_validation.json
"$PYTHON" -m flightrecorder schemas --check runs/gate_validation.json >/dev/null
"$PYTHON" -m flightrecorder trainer-preflight \
  --gate runs/training_gate.json \
  --gate runs/compare_gate.json \
  --gate runs/reviewed_gate.json \
  --gate runs/improvement_ledger_gate.json \
  --gate runs/promotion_ledger_gate.json \
  --validation runs/gate_validation.json \
  --training-export runs/training_export \
  --compare-export runs/compare_rl_export \
  --reviewed-export runs/reviewed_export \
  --evidence-bundle runs/evidence_bundle_full.json \
  --require-gate training_gate \
  --require-gate compare_gate \
  --require-gate reviewed_gate \
  --require-gate improvement_ledger_gate \
  --require-gate promotion_ledger_gate \
  --trainer-command "python train.py --dry-run --dataset runs/training_export" \
  --metadata candidate=offline-demo \
  --out runs/trainer_preflight.json >/dev/null
test -f runs/trainer_preflight.json
"$PYTHON" -m flightrecorder schemas --check runs/trainer_preflight.json >/dev/null
"$PYTHON" -m flightrecorder trainer-launch-check \
  --preflight runs/trainer_preflight.json \
  --require-gate training_gate \
  --require-gate compare_gate \
  --require-gate reviewed_gate \
  --require-gate improvement_ledger_gate \
  --require-gate promotion_ledger_gate \
  --require-metadata candidate=offline-demo \
  --out runs/trainer_launch_check.json >/dev/null
test -f runs/trainer_launch_check.json
"$PYTHON" -m flightrecorder schemas --check runs/trainer_launch_check.json >/dev/null
"$PYTHON" -m flightrecorder trainer-archive \
  --preflight runs/trainer_preflight.json \
  --launch-check runs/trainer_launch_check.json \
  --out runs/trainer_archive \
  --require-self-contained \
  --force >/dev/null
test -f runs/trainer_archive/trainer_archive.json
"$PYTHON" -m flightrecorder schemas --check runs/trainer_archive/trainer_archive.json >/dev/null
mkdir -p runs/trainer_code
printf "print('trainer placeholder; Flight Recorder never executes this file')\n" > runs/trainer_code/train.py
"$PYTHON" -m flightrecorder trainer-archive-check \
  --archive runs/trainer_archive \
  --external-code-root runs/trainer_code \
  --out runs/trainer_archive_check.json \
  --strict >/dev/null
test -f runs/trainer_archive_check.json
"$PYTHON" -m flightrecorder schemas --check runs/trainer_archive_check.json >/dev/null
"$PYTHON" -m flightrecorder trainer-consumer-plan \
  --archive-check runs/trainer_archive_check.json \
  --out runs/trainer_consumer_plan.json \
  --strict >/dev/null
test -f runs/trainer_consumer_plan.json
"$PYTHON" -m flightrecorder schemas --check runs/trainer_consumer_plan.json >/dev/null
"$PYTHON" examples/trainer-wrapper/consume_trainer_plan.py \
  --plan runs/trainer_consumer_plan.json \
  --out runs/trainer_wrapper_dry_run.json \
  --strict >/dev/null
test -f runs/trainer_wrapper_dry_run.json
"$PYTHON" -m flightrecorder schemas --check runs/trainer_wrapper_dry_run.json >/dev/null
mkdir -p runs/agentic_training_result_artifacts
printf "tiny adapter bytes\n" > runs/agentic_training_result_artifacts/adapter.safetensors
printf '{"loss":0.0}\n' > runs/agentic_training_result_artifacts/metrics.json
"$PYTHON" scripts/preflight_agentic_training_runtime.py \
  --plan examples/agentic_training/plans/sft_then_dpo_plan.json \
  --require-module json \
  --skip-default-modules \
  --created-at "2026-07-02T00:00:00+00:00" \
  --out runs/agentic_training_runtime_preflight.json >/dev/null
test -f runs/agentic_training_runtime_preflight.json
"$PYTHON" -m flightrecorder schemas --check runs/agentic_training_runtime_preflight.json >/dev/null
"$PYTHON" -m flightrecorder agentic-training-flow \
  --plan examples/agentic_training/plans/sft_then_dpo_plan.json \
  --runtime-preflight runs/agentic_training_runtime_preflight.json \
  --trainer-consumer-plan runs/trainer_consumer_plan.json \
  --out runs/agentic_training_flow.json >/dev/null
"$PYTHON" -m flightrecorder schemas --check runs/agentic_training_flow.json >/dev/null
"$PYTHON" scripts/archive_agentic_training_result.py \
  --plan examples/agentic_training/plans/sft_then_dpo_plan.json \
  --runtime-preflight runs/agentic_training_runtime_preflight.json \
  --agentic-training-flow runs/agentic_training_flow.json \
  --status completed \
  --adapter runs/agentic_training_result_artifacts/adapter.safetensors \
  --metrics runs/agentic_training_result_artifacts/metrics.json \
  --created-at "2026-07-02T00:00:00+00:00" \
  --out runs/agentic_training_result.json >/dev/null
test -f runs/agentic_training_result.json
"$PYTHON" -m flightrecorder schemas --check runs/agentic_training_result.json >/dev/null
"$PYTHON" -m flightrecorder evidence-bundle \
  --runs runs \
  --suite-summary runs/suite_summary.json \
  --scenario-quality runs/scenario_quality.json \
  --evidence-coverage runs/evidence_coverage.json \
  --trace-observability runs/trace_observability.json \
  --repair-queue runs/repair_queue.json \
  --validation runs/validation.json \
  --training-export runs/training_export \
  --compare-export runs/compare_rl_export \
  --review-export runs/review_queue \
  --reviewed-export runs/reviewed_export \
  --review-calibration runs/review_calibration.json \
  --live-smoke-summary runs/live_smoke_summary.json \
  --trainer-preflight runs/trainer_preflight.json \
  --trainer-launch-check runs/trainer_launch_check.json \
  --trainer-archive runs/trainer_archive \
  --trainer-archive-check runs/trainer_archive_check.json \
  --trainer-consumer-plan runs/trainer_consumer_plan.json \
  --agentic-training-flow runs/agentic_training_flow.json \
  --trainer-wrapper-dry-run runs/trainer_wrapper_dry_run.json \
  --agentic-training-result runs/agentic_training_result.json \
  --out runs/evidence_bundle_trainer.json >/dev/null
test -f runs/evidence_bundle_trainer.json
"$PYTHON" -m flightrecorder validate \
  --evidence-bundle runs/evidence_bundle.json \
  --evidence-bundle runs/evidence_bundle_full.json \
  --evidence-bundle runs/evidence_bundle_trainer.json \
  --harness-manifest runs/harness_handoff/harness_manifest.json \
  --harness-result runs/harness_handoff/harness_result.json \
  --improvement-ledger-gate runs/improvement_ledger_gate.json \
  --action-ledger runs/action_ledger.json \
  --action-ledger-gate runs/action_ledger_gate.json \
  --decision-gate runs/promotion_decision.json \
  --promotion-ledger runs/promotion_ledger.json \
  --promotion-ledger-gate runs/promotion_ledger_gate.json \
  --promotion-archive runs/promotion_archive \
  --trainer-preflight runs/trainer_preflight.json \
  --trainer-launch-check runs/trainer_launch_check.json \
  --trainer-archive runs/trainer_archive \
  --trainer-archive-check runs/trainer_archive_check.json \
  --trainer-consumer-plan runs/trainer_consumer_plan.json \
  --agentic-training-flow runs/agentic_training_flow.json \
  --trainer-wrapper-dry-run runs/trainer_wrapper_dry_run.json \
  --repair-queue runs/repair_queue.json \
  --review-calibration runs/review_calibration.json \
  --live-smoke-summary runs/live_smoke_summary.json \
  --strict >/dev/null
"$PYTHON" - <<'PY'
import json
from pathlib import Path

bundle = json.loads(Path("runs/evidence_bundle_full.json").read_text(encoding="utf-8"))
trainer_bundle = json.loads(Path("runs/evidence_bundle_trainer.json").read_text(encoding="utf-8"))
action_ledger = json.loads(Path("runs/action_ledger.json").read_text(encoding="utf-8"))
action_ledger_gate = json.loads(Path("runs/action_ledger_gate.json").read_text(encoding="utf-8"))
promotion_decision = json.loads(Path("runs/promotion_decision.json").read_text(encoding="utf-8"))
promotion_ledger = json.loads(Path("runs/promotion_ledger.json").read_text(encoding="utf-8"))
promotion_ledger_gate = json.loads(Path("runs/promotion_ledger_gate.json").read_text(encoding="utf-8"))
promotion_archive = json.loads(Path("runs/promotion_archive/promotion_archive.json").read_text(encoding="utf-8"))
gate_validation = json.loads(Path("runs/gate_validation.json").read_text(encoding="utf-8"))
preflight = json.loads(Path("runs/trainer_preflight.json").read_text(encoding="utf-8"))
launch_check = json.loads(Path("runs/trainer_launch_check.json").read_text(encoding="utf-8"))
assert bundle["passed"] is True
assert bundle["readiness"] == "ready"
assert bundle["decision"]["recommendation"] == "promote_handoff"
assert trainer_bundle["passed"] is True
assert trainer_bundle["readiness"] == "ready"
assert trainer_bundle["metrics"]["trainer_handoff"]["complete_chain"] is True
assert trainer_bundle["metrics"]["trainer_handoff"]["all_included_ready"] is True
assert trainer_bundle["metrics"]["trainer_handoff"]["stage_count"] == 8
assert trainer_bundle["metrics"]["trainer_handoff"]["handoff_ready_count"] == 8
assert trainer_bundle["metrics"]["trainer_handoff"]["blocked_stage_count"] == 0
assert trainer_bundle["metrics"]["trainer_handoff"]["schema_supported_count"] == 8
assert trainer_bundle["decision"]["key_metrics"]["trainer_handoff"]["complete_chain"] is True
flow_stage = next(stage for stage in trainer_bundle["metrics"]["trainer_handoff"]["stages"] if stage["id"] == "agentic_training_flow")
assert flow_stage["recommendation"] == "ready_for_delegated_trainer_execution"
result_stage = next(stage for stage in trainer_bundle["metrics"]["trainer_handoff"]["stages"] if stage["id"] == "agentic_training_result")
assert result_stage["recommendation"] == "register_training_result"
assert result_stage["status"] == "completed"
assert result_stage["adapter_count"] == 1
assert result_stage["metrics_file_count"] == 1
assert trainer_bundle["artifacts"]["agentic_training_result"]["schema_version"] == "hfr.agentic_training_result.v1"
assert bundle["decision"]["gate_count"] == 4
assert bundle["decision"]["passed_gate_count"] == 4
assert bundle["decision"]["key_metrics"]["gates"]["failed"] == 0
assert bundle["decision"]["key_metrics"]["harness_handoff"]["pair_count"] == 1
assert bundle["decision"]["key_metrics"]["harness_handoff"]["passed_pair_count"] == 1
assert bundle["decision"]["key_metrics"]["harness_handoff"]["missing_pair_count"] == 0
assert bundle["decision"]["key_metrics"]["harness_handoff"]["run_suite_pair_count"] == 1
assert bundle["decision"]["key_metrics"]["harness_handoff"]["run_suite_lineage_valid_pair_count"] == 1
assert bundle["metrics"]["harness_handoff"]["pair_count"] == 1
assert bundle["metrics"]["harness_handoff"]["schema_valid_pair_count"] == 1
assert bundle["metrics"]["harness_handoff"]["artifact_valid_pair_count"] == 1
assert bundle["metrics"]["harness_handoff"]["consistent_pair_count"] == 1
assert bundle["metrics"]["harness_handoff"]["run_suite_pair_count"] == 1
assert bundle["metrics"]["harness_handoff"]["run_suite_lineage_valid_pair_count"] == 1
assert bundle["metrics"]["harness_handoff"]["runs"][0]["runner"] == "flightrecorder_run_suite"
assert bundle["metrics"]["harness_handoff"]["runs"][0]["provider"] == "fixture"
assert bundle["metrics"]["harness_handoff"]["runs"][0]["artifact_refs_valid"] is True
assert bundle["metrics"]["harness_handoff"]["runs"][0]["handoff_source"] == "flightrecorder run-suite --evidence-handoff"
assert bundle["metrics"]["harness_handoff"]["runs"][0]["suite_summary_path"] == "../suite_summary.json"
assert bundle["metrics"]["harness_handoff"]["runs"][0]["suite_total"] == 7
assert bundle["metrics"]["harness_handoff"]["runs"][0]["suite_passed"] == 2
assert bundle["metrics"]["harness_handoff"]["runs"][0]["suite_failed"] == 5
assert bundle["metrics"]["harness_handoff"]["runs"][0]["run_suite_lineage_valid"] is True
assert bundle["decision"]["key_metrics"]["compare_export"]["candidate_win_count"] == 1
assert bundle["decision"]["key_metrics"]["compare_export"]["task_completion_improvement_count"] == 1
assert bundle["decision"]["key_metrics"]["compare_export"]["candidate_win_scenarios"] == ["email_reply_completion"]
assert bundle["decision"]["key_metrics"]["compare_export"]["baseline_win_scenarios"] == []
assert bundle["decision"]["key_metrics"]["compare_export"]["task_completion_improvement_scenarios"] == ["email_reply_completion"]
assert bundle["decision"]["key_metrics"]["compare_export"]["task_completion_regression_scenarios"] == []
assert bundle["decision"]["key_metrics"]["compare_export"]["fixed_rule_counts"]["required_actions"] == 1
assert bundle["decision"]["key_metrics"]["compare_export"]["regressed_rule_counts"] == {}
assert bundle["decision"]["key_metrics"]["compare_export"]["new_critical_failure_counts"] == {}
assert bundle["decision"]["key_metrics"]["live_smoke_summary"]["passed"] is True
assert bundle["decision"]["key_metrics"]["live_smoke_summary"]["consistent"] is True
assert bundle["decision"]["key_metrics"]["live_smoke_summary"]["score"] == 100
assert bundle["decision"]["key_metrics"]["live_smoke_summary"]["missing_hook_count"] == 0
assert bundle["decision"]["key_metrics"]["training_export"]["trainer_view_source_fingerprint_coverage"]["fully_verified_rate"] == 1.0
assert bundle["decision"]["key_metrics"]["live_smoke_summary"]["platform"] == "Linux-release-check"
assert bundle["decision"]["key_metrics"]["live_smoke_summary"]["hermes_git_commit"] == "abcdef123456"
assert bundle["decision"]["key_metrics"]["live_smoke_summary"]["flight_recorder_git_commit"] == "123456abcdef"
assert bundle["decision"]["key_metrics"]["trace_observability"]["run_count"] == 7
assert all(len(item["action_fingerprint"]) == 64 for item in bundle["decision"]["next_actions"])
assert all(
    item["routing_key"] == f"{item['artifact']}:{item['id']}:{item['action_fingerprint'][:12]}"
    for item in bundle["decision"]["next_actions"]
)
assert action_ledger["passed"] is True
assert action_ledger["bundle_count"] == 2
assert action_ledger["metrics"]["bundle_count"] == 2
assert action_ledger["metrics"]["unique_action_count"] == action_ledger["unique_action_count"]
assert action_ledger["metrics"]["open_action_count"] >= 1
assert action_ledger["metrics"]["recurring_action_count"] >= 1
assert [record["path"] for record in action_ledger["bundles"]] == ["evidence_bundle.json", "evidence_bundle_full.json"]
assert [record["path"] for record in action_ledger["metrics"]["bundle_action_counts"]] == ["evidence_bundle.json", "evidence_bundle_full.json"]
assert all(
    occurrence["bundle_path"] in {"evidence_bundle.json", "evidence_bundle_full.json"}
    for entry in action_ledger["entries"]
    for occurrence in entry["occurrences"]
)
assert all(len(entry["action_fingerprint"]) == 64 for entry in action_ledger["entries"])
assert all(
    entry["routing_key"] == f"{entry['artifact']}:{entry['id']}:{entry['action_fingerprint'][:12]}"
    for entry in action_ledger["entries"]
)
assert action_ledger_gate["schema_version"] == "hfr.action_ledger_gate.v1"
assert action_ledger_gate["action_ledger"] == "action_ledger.json"
assert action_ledger_gate["passed"] is True
assert action_ledger_gate["failed_check_count"] == 0
assert action_ledger_gate["decision"]["readiness"] == "ready"
assert action_ledger_gate["decision"]["recommendation"] == "promote_iteration"
assert action_ledger_gate["decision"]["blocking_check_count"] == 0
assert action_ledger_gate["decision"]["key_metrics"]["recurring_action_count"] == action_ledger_gate["metrics"]["recurring_action_count"]
assert action_ledger_gate["policy"]["schema_version"] == "hfr.action_ledger_gate.policy.v1"
assert action_ledger_gate["policy"]["effective"]["max_recurring_actions"] == 6
assert action_ledger_gate["metrics"]["recurring_action_count"] == action_ledger["metrics"]["recurring_action_count"]
assert promotion_decision["schema_version"] == "hfr.decision_gate.v1"
assert promotion_decision["passed"] is True
assert promotion_decision["recommendation"] == "allow_promotion"
assert promotion_decision["expected_recommendation"] == "promote_iteration"
assert promotion_decision["source_artifact"]["path"] == "action_ledger_gate.json"
assert promotion_decision["source_artifact"]["exists"] is True
assert len(promotion_decision["source_artifact"]["sha256"]) == 64
assert promotion_decision["source_decision"]["schema_version"] == action_ledger_gate["schema_version"]
assert promotion_decision["source_decision"]["recommendation"] == "promote_iteration"
assert promotion_decision["source_decision"]["key_metrics"]["recurring_action_count"] == action_ledger_gate["metrics"]["recurring_action_count"]
assert promotion_ledger["schema_version"] == "hfr.promotion_ledger.v1"
assert promotion_ledger["passed"] is True
assert promotion_ledger["decision_count"] == 2
assert promotion_ledger["metrics"]["decision_count"] == 2
assert promotion_ledger["metrics"]["allowed_count"] == 2
assert promotion_ledger["metrics"]["blocked_count"] == 0
assert promotion_ledger["metrics"]["latest_recommendation"] == "allow_promotion"
assert promotion_ledger["metrics"]["latest_readiness"] == "ready"
assert promotion_ledger["metrics"]["latest_passed"] is True
assert promotion_ledger["metrics"]["consecutive_allowed_count"] == 2
assert promotion_ledger["metrics"]["consecutive_blocked_count"] == 0
assert promotion_ledger["metrics"]["unique_source_artifact_count"] == 1
assert promotion_ledger["metrics"]["recommendation_counts"] == [{"count": 2, "id": "allow_promotion"}]
assert promotion_ledger["metrics"]["source_recommendation_counts"] == [{"count": 2, "id": "promote_iteration"}]
assert [record["path"] for record in promotion_ledger["records"]] == ["promotion_decision_previous.json", "promotion_decision.json"]
assert all(len(record["sha256"]) == 64 for record in promotion_ledger["records"])
assert all(len(record["source"]["artifact_sha256"]) == 64 for record in promotion_ledger["records"])
assert promotion_ledger_gate["schema_version"] == "hfr.promotion_ledger_gate.v1"
assert promotion_ledger_gate["passed"] is True
assert promotion_ledger_gate["decision"]["readiness"] == "ready"
assert promotion_ledger_gate["decision"]["recommendation"] == "promote_iteration"
assert promotion_ledger_gate["failed_check_count"] == 0
assert promotion_ledger_gate["metrics"]["decision_count"] == 2
assert promotion_ledger_gate["metrics"]["allowed_count"] == 2
assert promotion_ledger_gate["metrics"]["blocked_count"] == 0
assert promotion_ledger_gate["metrics"]["blocked_rate"] == 0.0
assert promotion_ledger_gate["metrics"]["latest_recommendation"] == "allow_promotion"
assert promotion_ledger_gate["metrics"]["latest_passed"] is True
assert promotion_ledger_gate["metrics"]["consecutive_allowed_count"] == 2
assert promotion_ledger_gate["metrics"]["consecutive_blocked_count"] == 0
assert promotion_ledger_gate["metrics"]["failed_decision_count"] == 0
assert promotion_ledger_gate["policy"]["schema_version"] == "hfr.promotion_ledger_gate.policy.v1"
assert promotion_ledger_gate["policy"]["effective"]["require_latest_recommendation"] == "allow_promotion"
assert promotion_ledger_gate["policy"]["effective"]["require_latest_passed"] is True
assert promotion_archive["schema_version"] == "hfr.promotion_archive.v1"
assert promotion_archive["passed"] is True
assert promotion_archive["self_contained"] is True
assert promotion_archive["require_self_contained"] is True
assert promotion_archive["metrics"]["missing_count"] == 0
assert promotion_archive["metrics"]["decision_gate_count"] == 1
assert promotion_archive["metrics"]["source_artifact_count"] == 1
archive_roles = {artifact["role"] for artifact in promotion_archive["artifacts"]}
assert archive_roles == {"promotion_ledger", "promotion_ledger_gate", "decision_gate", "source_artifact"}
assert all(len(artifact["sha256"]) == 64 for artifact in promotion_archive["artifacts"])
assert all(not artifact["original_path"].startswith("/") for artifact in promotion_archive["artifacts"])
assert all(not (len(artifact["original_path"]) > 2 and artifact["original_path"][1:3] == ":\\") for artifact in promotion_archive["artifacts"])
assert len(bundle["metrics"]["gates"]) == 4
assert {gate["id"] for gate in bundle["metrics"]["gates"]} == {
    "suite_gate",
    "compare_gate",
    "training_gate",
    "reviewed_gate",
}
assert bundle["metrics"]["compare_export"]["candidate_win_count"] == 1
assert bundle["metrics"]["compare_export"]["task_completion_improvement_count"] == 1
assert bundle["metrics"]["compare_export"]["task_completion_regression_count"] == 0
assert bundle["metrics"]["compare_export"]["task_completion_improvement_scenarios"] == ["email_reply_completion"]
assert bundle["metrics"]["compare_export"]["task_completion_regression_scenarios"] == []
assert bundle["metrics"]["compare_export"]["regressed_rule_counts"] == {}
assert bundle["metrics"]["compare_export"]["new_critical_failure_counts"] == {}
assert bundle["metrics"]["training_export"]["trainer_view_source_fingerprint_coverage"]["unverified"] == 0
assert bundle["metrics"]["live_smoke_summary"]["chat_completion_request_count"] == 1
assert bundle["metrics"]["live_smoke_summary"]["hermes_root"] == "<redacted:hermes-agent>"
assert bundle["metrics"]["live_smoke_summary"]["flight_recorder_root"] == "."
assert bundle["metrics"]["trace_observability"]["final_answer_rate"] == 1.0
assert bundle["metrics"]["review_export"]["item_count"] >= 6
assert bundle["metrics"]["reviewed_export"]["reviewed_label_count"] == bundle["metrics"]["review_export"]["item_count"]
assert bundle["metrics"]["review_calibration"]["agreement_rate"] == 1.0
assert bundle["metrics"]["review_calibration"]["disagreement_count"] == 0
assert preflight["passed"] is True
assert preflight["recommendation"] == "launch_allowed"
assert preflight["metadata"]["candidate"] == "offline-demo"
assert gate_validation["passed"] is True
assert {target["type"] for target in gate_validation["targets"]} == {"improvement_ledger_gate", "promotion_ledger_gate"}
assert preflight["passed_gate_count"] == 5
assert {gate["id"] for gate in preflight["gates"]} == {
    "training_gate",
    "compare_gate",
    "reviewed_gate",
    "improvement_ledger_gate",
    "promotion_ledger_gate",
}
assert all(gate["passed"] is True for gate in preflight["gates"])
improvement_preflight_gate = next(gate for gate in preflight["gates"] if gate["id"] == "improvement_ledger_gate")
assert improvement_preflight_gate["validation"]["available"] is True
assert improvement_preflight_gate["validation"]["passed"] is True
assert improvement_preflight_gate["validation"]["target_type"] == "improvement_ledger_gate"
assert preflight["validation_summaries"][0]["path"] == "gate_validation.json"
assert preflight["validation_summaries"][0]["target_count"] == 2
assert len(preflight["validation_summaries"][0]["sha256"]) == 64
assert sorted(gate["path"] for gate in preflight["gates"]) == [
    "compare_gate.json",
    "improvement_ledger_gate.json",
    "promotion_ledger_gate.json",
    "reviewed_gate.json",
    "training_gate.json",
]
assert preflight["schema_contracts"]["training_export_manifest_json"]["passed"] is True
assert preflight["schema_contracts"]["training_export_manifest_json"]["path"] == "training_export/manifest.json"
assert preflight["schema_contracts"]["training_export_sft_jsonl"]["schema_name"] == "rl_sft"
assert preflight["schema_contracts"]["training_export_sft_jsonl"]["passed"] is True
assert preflight["schema_contracts"]["compare_export_improvement_dpo_jsonl"]["schema_name"] == "compare_rl_dpo"
assert preflight["schema_contracts"]["compare_export_improvement_dpo_jsonl"]["passed"] is True
assert preflight["schema_contracts"]["reviewed_export_manifest_json"]["schema_name"] == "reviewed_manifest"
assert preflight["schema_contracts"]["reviewed_export_manifest_json"]["passed"] is True
assert preflight["schema_contracts"]["evidence_bundle"]["schema_name"] == "evidence_bundle"
assert preflight["schema_contracts"]["evidence_bundle"]["passed"] is True
assert preflight["schema_contracts"]["evidence_bundle"]["path"] == "evidence_bundle_full.json"
assert all(contract["passed"] is True for contract in preflight["schema_contracts"].values())
assert preflight["trainer_command"]["argv"][:2] == ["python", "train.py"]
assert len(preflight["artifacts"]["training_export_sft_jsonl"]["sha256"]) == 64
assert len(preflight["artifacts"]["training_export_dataset_splits_json"]["sha256"]) == 64
assert len(preflight["artifacts"]["training_export_splits_train_episodes_jsonl"]["sha256"]) == 64
assert len(preflight["artifacts"]["training_export_splits_validation_episodes_jsonl"]["sha256"]) == 64
assert len(preflight["artifacts"]["training_export_splits_test_episodes_jsonl"]["sha256"]) == 64
assert len(preflight["artifacts"]["compare_export_improvement_pairs_jsonl"]["sha256"]) == 64
assert launch_check["passed"] is True
assert launch_check["recommendation"] == "launch_allowed"
assert launch_check["validation"]["passed"] is True
assert launch_check["approved_command"]["approved"] is True
assert launch_check["approved_command"]["argv"][:2] == ["python", "train.py"]
assert {gate["id"] for gate in launch_check["gates"]} == {
    "training_gate",
    "compare_gate",
    "reviewed_gate",
    "improvement_ledger_gate",
    "promotion_ledger_gate",
}
PY
"$PYTHON" -m flightrecorder compare-suite \
  --baseline runs \
  --candidate runs \
  --out runs/suite_compare_check.json \
  --fail-on-regression \
  --fail-on-contract-drift \
  --fail-on-unverified-contracts >/dev/null
"$PYTHON" -m flightrecorder schemas --check runs/suite_compare_check.json >/dev/null

"$PYTHON" -m flightrecorder audit \
  --runs runs \
  --forbid-text hfr_fixture_secret_value_123 \
  --forbid-text DEMO_API_KEY=hfr_fixture \
  --fail-on-leak >/dev/null

INSTALL_DIR="$(mktemp -d)"
VENV_DIR="$(mktemp -d)"
"$PYTHON" -m venv --system-site-packages "$VENV_DIR"
if ! "$VENV_DIR/bin/python" -c "import setuptools" >/dev/null 2>&1; then
  "$VENV_DIR/bin/python" -m pip install "setuptools>=68" >/dev/null
fi
"$VENV_DIR/bin/python" -m pip install . --no-deps --no-build-isolation >/dev/null
"$VENV_DIR/bin/python" - <<'PY'
import importlib.metadata
import flightrecorder

assert flightrecorder.__version__ == importlib.metadata.version("hermes-flight-recorder")
PY
"$VENV_DIR/bin/flightrecorder" --help >/dev/null
"$VENV_DIR/bin/python" -m flightrecorder schemas --name scorecard \
  --out "$INSTALL_DIR/scorecard.schema.json" >/dev/null
test -f "$INSTALL_DIR/scorecard.schema.json"
"$VENV_DIR/bin/python" -m flightrecorder normalize \
  --trace fixtures/prompt_injection_good.trajectory.jsonl \
  --out "$INSTALL_DIR/normalized.json" >/dev/null
"$VENV_DIR/bin/python" -m flightrecorder schemas \
  --check "$INSTALL_DIR/normalized.json" >/dev/null
assert_help_contains "--check-jsonl" "$VENV_DIR/bin/python" -m flightrecorder schemas --help
"$VENV_DIR/bin/python" -m flightrecorder replay --help >/dev/null
"$VENV_DIR/bin/python" -m flightrecorder replay-bundle --help >/dev/null
"$VENV_DIR/bin/python" -m flightrecorder capture-state --help >/dev/null
assert_help_contains "--replay-bundle" "$VENV_DIR/bin/python" -m flightrecorder validate --help
assert_help_contains "--state-snapshot" "$VENV_DIR/bin/python" -m flightrecorder validate --help
assert_help_contains "--live-smoke-summary" "$VENV_DIR/bin/python" -m flightrecorder validate --help
assert_help_contains "--trainer-launch-check" "$VENV_DIR/bin/python" -m flightrecorder validate --help
assert_help_contains "--trainer-archive" "$VENV_DIR/bin/python" -m flightrecorder validate --help
assert_help_contains "--trainer-archive-check" "$VENV_DIR/bin/python" -m flightrecorder validate --help
assert_help_contains "--trainer-consumer-plan" "$VENV_DIR/bin/python" -m flightrecorder validate --help
assert_help_contains "--trainer-wrapper-dry-run" "$VENV_DIR/bin/python" -m flightrecorder validate --help
assert_help_contains "--harness-manifest" "$VENV_DIR/bin/python" -m flightrecorder validate --help
assert_help_contains "--harness-result" "$VENV_DIR/bin/python" -m flightrecorder validate --help
assert_help_contains "--action-ledger" "$VENV_DIR/bin/python" -m flightrecorder validate --help
assert_help_contains "--improvement-ledger-gate" "$VENV_DIR/bin/python" -m flightrecorder validate --help
assert_help_contains "--action-ledger-gate" "$VENV_DIR/bin/python" -m flightrecorder validate --help
assert_help_contains "--decision-gate" "$VENV_DIR/bin/python" -m flightrecorder validate --help
assert_help_contains "--promotion-ledger" "$VENV_DIR/bin/python" -m flightrecorder validate --help
assert_help_contains "--promotion-ledger-gate" "$VENV_DIR/bin/python" -m flightrecorder validate --help
assert_help_contains "--promotion-archive" "$VENV_DIR/bin/python" -m flightrecorder validate --help
"$VENV_DIR/bin/python" -m flightrecorder observer-template \
  --out "$INSTALL_DIR/flight_recorder_plugin.py" >/dev/null
"$VENV_DIR/bin/python" -m flightrecorder run-suite --help >/dev/null
"$VENV_DIR/bin/python" -m flightrecorder check-scenarios --help >/dev/null
"$VENV_DIR/bin/python" -m flightrecorder scenario-quality --help >/dev/null
"$VENV_DIR/bin/python" -m flightrecorder draft-scenario --help >/dev/null
"$VENV_DIR/bin/python" -m flightrecorder evidence-coverage --help >/dev/null
"$VENV_DIR/bin/python" -m flightrecorder trace-observability --help >/dev/null
"$VENV_DIR/bin/python" -m flightrecorder evidence-bundle --help >/dev/null
assert_help_contains "--live-smoke-summary" "$VENV_DIR/bin/python" -m flightrecorder evidence-bundle --help
assert_help_contains "--trainer-preflight" "$VENV_DIR/bin/python" -m flightrecorder evidence-bundle --help
assert_help_contains "--trainer-launch-check" "$VENV_DIR/bin/python" -m flightrecorder evidence-bundle --help
assert_help_contains "--trainer-archive" "$VENV_DIR/bin/python" -m flightrecorder evidence-bundle --help
assert_help_contains "--trainer-archive-check" "$VENV_DIR/bin/python" -m flightrecorder evidence-bundle --help
assert_help_contains "--trainer-consumer-plan" "$VENV_DIR/bin/python" -m flightrecorder evidence-bundle --help
assert_help_contains "--trainer-wrapper-dry-run" "$VENV_DIR/bin/python" -m flightrecorder evidence-bundle --help
assert_help_contains "--harness-manifest" "$VENV_DIR/bin/python" -m flightrecorder evidence-bundle --help
assert_help_contains "--harness-result" "$VENV_DIR/bin/python" -m flightrecorder evidence-bundle --help
assert_help_contains "--require-harness" "$VENV_DIR/bin/python" -m flightrecorder evidence-bundle --help
assert_help_contains "--require-gate" "$VENV_DIR/bin/python" -m flightrecorder evidence-bundle --help
"$VENV_DIR/bin/python" -m flightrecorder action-ledger --help >/dev/null
assert_help_contains "--bundle" "$VENV_DIR/bin/python" -m flightrecorder action-ledger --help
"$VENV_DIR/bin/python" -m flightrecorder gate-improvement-ledger --help >/dev/null
assert_help_contains "--max-recurring-work-items" "$VENV_DIR/bin/python" -m flightrecorder gate-improvement-ledger --help
"$VENV_DIR/bin/python" -m flightrecorder gate-action-ledger --help >/dev/null
assert_help_contains "--max-recurring-actions" "$VENV_DIR/bin/python" -m flightrecorder gate-action-ledger --help
"$VENV_DIR/bin/python" -m flightrecorder gate-decision --help >/dev/null
assert_help_contains "--expect-recommendation" "$VENV_DIR/bin/python" -m flightrecorder gate-decision --help
"$VENV_DIR/bin/python" -m flightrecorder promotion-ledger --help >/dev/null
assert_help_contains "--decision-gate" "$VENV_DIR/bin/python" -m flightrecorder promotion-ledger --help
"$VENV_DIR/bin/python" -m flightrecorder promotion-archive --help >/dev/null
assert_help_contains "--require-self-contained" "$VENV_DIR/bin/python" -m flightrecorder promotion-archive --help
"$VENV_DIR/bin/python" -m flightrecorder gate-promotion-ledger --help >/dev/null
assert_help_contains "--max-blocked-rate" "$VENV_DIR/bin/python" -m flightrecorder gate-promotion-ledger --help
test -f examples/promotion_ledger_gate_policy.demo.json
test -f examples/github-actions/action-ledger-promotion-gate.yml
grep -q "gate-decision" examples/github-actions/action-ledger-promotion-gate.yml
grep -q "decision-gate" examples/github-actions/action-ledger-promotion-gate.yml
grep -q "promotion-ledger" examples/github-actions/action-ledger-promotion-gate.yml
grep -q "gate-promotion-ledger" examples/github-actions/action-ledger-promotion-gate.yml
grep -q "promotion-archive" examples/github-actions/action-ledger-promotion-gate.yml
"$VENV_DIR/bin/python" -m flightrecorder gate-suite --help >/dev/null
"$VENV_DIR/bin/python" -m flightrecorder trend-suite --help >/dev/null
"$VENV_DIR/bin/python" -m flightrecorder gate-export --help >/dev/null
assert_help_contains "--min-task-completion-complete" "$VENV_DIR/bin/python" -m flightrecorder gate-export --help
assert_help_contains "--min-trace-average-events" "$VENV_DIR/bin/python" -m flightrecorder gate-export --help
assert_help_contains "--min-trainer-view-source-fingerprint-rate" "$VENV_DIR/bin/python" -m flightrecorder gate-export --help
assert_help_contains "--min-validation-episodes" "$VENV_DIR/bin/python" -m flightrecorder gate-export --help
assert_help_contains "--require-family-exclusive-splits" "$VENV_DIR/bin/python" -m flightrecorder gate-export --help
assert_help_contains "--strict-validation" "$VENV_DIR/bin/python" -m flightrecorder gate-export --help
assert_help_contains "--skip-validation" "$VENV_DIR/bin/python" -m flightrecorder gate-export --help
"$VENV_DIR/bin/python" -m flightrecorder gate-reviewed --help >/dev/null
assert_help_contains "--min-high-confidence-labels" "$VENV_DIR/bin/python" -m flightrecorder gate-reviewed --help
assert_help_contains "--min-medium-or-high-confidence-labels" "$VENV_DIR/bin/python" -m flightrecorder gate-reviewed --help
assert_help_contains "--max-low-confidence-labels" "$VENV_DIR/bin/python" -m flightrecorder gate-reviewed --help
assert_help_contains "--max-unknown-confidence-labels" "$VENV_DIR/bin/python" -m flightrecorder gate-reviewed --help
"$VENV_DIR/bin/python" -m flightrecorder gate-compare-export --help >/dev/null
assert_help_contains "--min-task-completion-improvements" "$VENV_DIR/bin/python" -m flightrecorder gate-compare-export --help
assert_help_contains "--strict-validation" "$VENV_DIR/bin/python" -m flightrecorder gate-compare-export --help
assert_help_contains "--skip-validation" "$VENV_DIR/bin/python" -m flightrecorder gate-compare-export --help
"$VENV_DIR/bin/python" -m flightrecorder trainer-preflight --help >/dev/null
assert_help_contains "--validation" "$VENV_DIR/bin/python" -m flightrecorder trainer-preflight --help
assert_help_contains "--allow-unvalidated-gates" "$VENV_DIR/bin/python" -m flightrecorder trainer-preflight --help
"$VENV_DIR/bin/python" -m flightrecorder trainer-launch-check --help >/dev/null
assert_help_contains "--print-command" "$VENV_DIR/bin/python" -m flightrecorder trainer-launch-check --help
"$VENV_DIR/bin/python" -m flightrecorder trainer-archive --help >/dev/null
assert_help_contains "--require-self-contained" "$VENV_DIR/bin/python" -m flightrecorder trainer-archive --help
"$VENV_DIR/bin/python" -m flightrecorder trainer-archive-check --help >/dev/null
assert_help_contains "--external-code-root" "$VENV_DIR/bin/python" -m flightrecorder trainer-archive-check --help
"$VENV_DIR/bin/python" -m flightrecorder trainer-consumer-plan --help >/dev/null
assert_help_contains "--archive-check" "$VENV_DIR/bin/python" -m flightrecorder trainer-consumer-plan --help
test -f examples/trainer-wrapper/consume_trainer_plan.py
"$VENV_DIR/bin/python" examples/trainer-wrapper/consume_trainer_plan.py --help >/dev/null
assert_help_contains "--print-command" "$VENV_DIR/bin/python" examples/trainer-wrapper/consume_trainer_plan.py --help
assert_help_contains "--metadata" "$VENV_DIR/bin/python" -m flightrecorder export-rl --help
"$VENV_DIR/bin/python" -m flightrecorder export-compare-rl --help >/dev/null
"$VENV_DIR/bin/python" -m flightrecorder export-review --help >/dev/null
"$VENV_DIR/bin/python" -m flightrecorder apply-review --help >/dev/null
"$VENV_DIR/bin/python" -m flightrecorder review-calibration --help >/dev/null

if "$VENV_DIR/bin/flightrecorder" run \
  --scenario scenarios/prompt_injection_bad.json \
  --out "$INSTALL_DIR/failing-run" \
  --fail-on-score >/dev/null; then
  echo "--fail-on-score did not fail a failing scenario" >&2
  exit 1
fi

rm -rf "$INSTALL_DIR" "$VENV_DIR"
echo "release check passed"
