# Trainer Wrapper Dry Run Example

This directory shows how an external trainer launcher can consume
`trainer_consumer_plan.json` without making Flight Recorder a trainer.

```bash
python examples/trainer-wrapper/consume_trainer_plan.py \
  --plan runs/trainer_consumer_plan.json \
  --out runs/trainer_wrapper_dry_run.json \
  --strict
```

The script validates the plan, checks the expected ready recommendation, records
the command and input hashes it would hand to a trainer, and writes a dry-run
receipt. It does not execute `execution.command_argv`.

Real training infrastructure can reuse the same pattern, then take explicit
ownership of process execution outside Flight Recorder.
