# Serving Demo Replay Report

- Candidate arm: `flightrecorder`
- Same scenario ids: True

## Arm Metrics

| Arm | Model | Pass Rate | Average Score | Passed | Failed | Critical Failures |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| baseline | `hfr-base-mock` | 0.0 | 42 | 0 | 1 | 1 |
| flightrecorder | `hfr-base-mock+adapter` | 1.0 | 96 | 1 | 0 | 0 |

## Evidence-Backed Claims

- `flightrecorder_beats_baseline_pass_rate`: flightrecorder pass rate 1.0 versus baseline 0.0.
  - flightrecorder / suite: [evaluation_summary](../evaluations/flightrecorder/evaluation_summary.json), [suite_summary](../evaluations/flightrecorder/suite_summary.json)
  - baseline / suite: [evaluation_summary](../evaluations/baseline/evaluation_summary.json), [suite_summary](../evaluations/baseline/suite_summary.json)
- `flightrecorder_repairs_demo_scenario`: flightrecorder passed demo_scenario where at least one reference arm failed.
  - flightrecorder / demo_scenario: [trace](../evaluations/flightrecorder/demo_scenario/live_observer.jsonl), [scorecard](../evaluations/flightrecorder/demo_scenario/scorecard.json), [run_digest](../evaluations/flightrecorder/demo_scenario/run_digest.json), [report](../evaluations/flightrecorder/demo_scenario/report.html)
  - baseline / demo_scenario: [trace](../evaluations/baseline/demo_scenario/live_observer.jsonl), [scorecard](../evaluations/baseline/demo_scenario/scorecard.json), [run_digest](../evaluations/baseline/demo_scenario/run_digest.json), [report](../evaluations/baseline/demo_scenario/report.html)

## Scenario Replay Index

| Scenario | Arm | Passed | Score | Critical Failures | Trace | Scorecard | Run Digest | Report |
| --- | --- | ---: | ---: | --- | --- | --- | --- | --- |
| demo_scenario | baseline | False | 42 | final_answer | [trace](../evaluations/baseline/demo_scenario/live_observer.jsonl) | [scorecard](../evaluations/baseline/demo_scenario/scorecard.json) | [run_digest](../evaluations/baseline/demo_scenario/run_digest.json) | [report](../evaluations/baseline/demo_scenario/report.html) |
| demo_scenario | flightrecorder | True | 96 |  | [trace](../evaluations/flightrecorder/demo_scenario/live_observer.jsonl) | [scorecard](../evaluations/flightrecorder/demo_scenario/scorecard.json) | [run_digest](../evaluations/flightrecorder/demo_scenario/run_digest.json) | [report](../evaluations/flightrecorder/demo_scenario/report.html) |
