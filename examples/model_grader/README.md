# Model Grader Dry-Run Examples

These committed artifacts show the fail-closed rubric/model-grader path over
the offline prompt-injection fixtures.

```bash
flightrecorder model-grader rubric \
  --review-export runs/review_queue \
  --rubric-id prompt-injection-rubric \
  --out runs/model_grader/rubric.json

flightrecorder model-grader dry-run \
  --review-export runs/review_queue \
  --rubric runs/model_grader/rubric.json \
  --grader-id mock-grader-v1 \
  --provider mock \
  --out runs/model_grader/dry_run.json

flightrecorder model-grader gate \
  --dry-run runs/model_grader/dry_run.json \
  --rubric runs/model_grader/rubric.json \
  --review-calibration runs/review_calibration.json \
  --out runs/model_grader/gate.json
```

`dry_run.json` records deterministic mock labels only. It does not call a model
provider, paid grader, trainer, or cloud job, and it admits zero labels to
training. `blocked_gate.json` shows the default missing-calibration block.
`passing_gate.json` shows the same dry-run labels becoming eligible only after a
passing `review_calibration.json` and an empty dry-run disagreement queue; even
then Flight Recorder records no provider call, no credential values, and no
weight updates.

When `dry_run.json` contains queued items, write
`model-grader override-receipt` from human override JSONL and pass that receipt
to `model-grader gate`; these fixtures do not need one because the queue is
empty.

Validate the examples:

```bash
flightrecorder validate \
  --review-export examples/model_grader/review \
  --rubric-spec examples/model_grader/rubric.json \
  --model-grader-dry-run examples/model_grader/dry_run.json \
  --model-grader-gate examples/model_grader/blocked_gate.json \
  --model-grader-gate examples/model_grader/passing_gate.json \
  --review-calibration examples/model_grader/review_calibration.json \
  --strict
```
