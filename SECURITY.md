# Security Notes

Hermes Flight Recorder is an audit and evaluation tool. It is not an isolation
boundary and does not prevent prompt injection, data exfiltration, or unsafe
tool execution by itself.

## Safe Defaults

- `run` scores raw trace evidence in memory, then writes a redacted
  `normalized_trace.json`.
- Raw evidence is written only when `--write-sensitive-trace` is set, and the
  file name ends with `.sensitive.json`.
- `report.html` and `scorecard.json` redact matched secret-like values.
- The optional Hermes observer collector is read-only and fail-open.

## Sensitive Artifacts

Treat these as sensitive in real deployments:

- Original Hermes trajectory/observer/ATOF/ATIF exports.
- `raw_trace.sensitive.json`.
- Any `runs/` directory generated with custom policies that do not cover local
  credential formats.

Do not publish raw traces from production systems without a separate review.

## Reporting Issues

For this hackathon project, report security issues privately to the project
owner before sharing live traces or credentials in public tickets.
