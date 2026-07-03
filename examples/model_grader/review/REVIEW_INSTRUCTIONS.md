# Flight Recorder Review Queue

This export is for human curation before evidence becomes training data.

- Items: `2`
- Passed: `1`
- Failed: `1`
- Label options: `accept`, `reject`, `needs_review`, `unsafe`, `incomplete`

Review `review_items.jsonl` alongside each item report and lineage file.
Fill `label_template.jsonl` with `human_label`, `reviewer_confidence`, `reviewer`, `reviewed_at`, and notes.
Human labels should be grounded in observable trace events, scorecard evidence, reports, and lineage.
A suggested label is only a starting point; prefer observable trace evidence over final-answer claims.
