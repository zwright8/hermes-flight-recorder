#!/usr/bin/env python3
# /// script
# dependencies = [
#   "accelerate==1.14.0",
#   "datasets==5.0.0",
#   "peft==0.19.1",
#   "torch==2.12.1",
#   "trackio==0.28.0",
#   "transformers==5.12.1",
#   "trl==1.7.0",
# ]
# ///
"""Train and optionally publish the self-improving-agent LoRA adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flightrecorder.schema_registry import check_schema_contract  # noqa: E402


RESULT_SCHEMA = "hfr.self_improving_agent_training_result.v1"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    result = check_schema_contract(value, artifact_path=path)
    if not result["passed"]:
        raise ValueError(f"schema validation failed for {path}: {'; '.join(result['errors'])}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _artifact_manifest(root: Path) -> dict[str, Any]:
    files = [path for path in sorted(root.rglob("*")) if path.is_file() and not path.is_symlink()]
    return {
        "directory": str(root),
        "file_count": len(files),
        "files": [
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in files
        ],
    }


def validate_inputs(data_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_manifest = _load_json(data_dir / "dataset_manifest.json")
    frozen = _load_json(data_dir / "frozen_heldout_manifest.json")
    contamination = _load_json(data_dir / "contamination_audit.json")
    train_path = data_dir / "train_trajectories.jsonl"
    development_path = data_dir / "development_tasks.jsonl"
    heldout_path = data_dir / "heldout_tasks.jsonl"
    checks = {
        "dataset_public_safe": dataset_manifest.get("public_safe") is True,
        "contamination_passed": contamination.get("passed") is True,
        "heldout_immutable": frozen.get("immutable") is True,
        "train_sha256": _sha256_file(train_path) == frozen["training_artifact"]["sha256"],
        "development_sha256": _sha256_file(development_path) == frozen["development_artifact"]["sha256"],
        "heldout_sha256": _sha256_file(heldout_path) == frozen["artifact"]["sha256"],
        "non_training_artifacts_are_distinct": len(
            {train_path.resolve(), development_path.resolve(), heldout_path.resolve()}
        ) == 3,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("input validation failed: " + ", ".join(failed))
    rows = _load_jsonl(train_path)
    if len(rows) != int(frozen["training_artifact"]["row_count"]):
        raise ValueError("training row count does not match frozen manifest")
    required = {"task_id", "messages", "tools", "expected", "governance"}
    for index, row in enumerate(rows):
        missing = sorted(required - set(row))
        if missing or row.get("split") != "train" or not row.get("messages") or not row.get("tools"):
            raise ValueError(f"invalid training row {index}: missing={missing}, split={row.get('split')!r}")
    return rows, {
        "checks": checks,
        "train_sha256": _sha256_file(train_path),
        "heldout_sha256": _sha256_file(heldout_path),
        "development_sha256": _sha256_file(development_path),
        "train_row_count": len(rows),
        "dataset_sha256": dataset_manifest["dataset_sha256"],
    }


def _dtype_policy(torch: Any) -> tuple[Any, bool, bool, str]:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, True, False, "cuda-bfloat16"
        return torch.float16, False, True, "cuda-float16"
    if torch.backends.mps.is_available():
        return torch.float16, False, False, "mps-float16"
    return torch.float32, False, False, "cpu-float32"


def train(args: argparse.Namespace) -> dict[str, Any]:
    import datasets
    import peft
    import torch
    import trackio
    import transformers
    import trl
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    rows, validation = validate_inputs(args.data_dir)
    if args.limit:
        rows = rows[: args.limit]
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"output directory is non-empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dtype, bf16, fp16, precision = _dtype_policy(torch)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision or None)
    chat_template_sha256 = hashlib.sha256(str(tokenizer.chat_template or "").encode("utf-8")).hexdigest()
    dataset = Dataset.from_list([{"messages": row["messages"], "tools": row["tools"]} for row in rows])
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    report_to: list[str] = [] if args.disable_trackio else ["trackio"]
    if not args.disable_trackio:
        os.environ.setdefault("TRACKIO_PROJECT_NAME", args.trackio_project)
        if args.trackio_space_id:
            os.environ.setdefault("TRACKIO_SPACE_ID", args.trackio_space_id)
    config = SFTConfig(
        output_dir=str(args.output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        max_length=args.max_length,
        assistant_only_loss=True,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        seed=args.seed,
        data_seed=args.seed,
        bf16=bf16,
        fp16=fp16,
        report_to=report_to,
        run_name=args.run_name,
        model_init_kwargs={"dtype": dtype, "revision": args.model_revision or None},
    )
    trainer = SFTTrainer(
        model=args.model,
        args=config,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    output = trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(args.output_dir)

    hub = None
    if args.push_to_hub:
        if not args.hub_model_id:
            raise ValueError("--push-to-hub requires --hub-model-id")
        commit = trainer.model.push_to_hub(
            args.hub_model_id,
            commit_message="Publish statistically evaluated Hermes Flight Recorder LoRA candidate",
            private=args.private,
        )
        tokenizer.push_to_hub(args.hub_model_id, private=args.private)
        hub = {
            "repo_id": args.hub_model_id,
            "commit_url": str(getattr(commit, "commit_url", "") or ""),
            "revision": str(getattr(commit, "oid", "") or ""),
            "private": args.private,
        }

    result = {
        "schema_version": RESULT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "succeeded",
        "base_model": args.model,
        "base_model_revision": args.model_revision or str(getattr(trainer.model.config, "_commit_hash", "") or ""),
        "chat_template_sha256": chat_template_sha256,
        "data_validation": validation,
        "training_row_count": len(rows),
        "hyperparameters": {
            "epochs": args.epochs,
            "max_steps": args.max_steps,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_length": args.max_length,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "seed": args.seed,
            "assistant_only_loss": True,
            "precision": precision,
        },
        "metrics": output.metrics,
        "libraries": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "trl": trl.__version__,
            "peft": peft.__version__,
            "datasets": datasets.__version__,
            "trackio": getattr(trackio, "__version__", "unknown"),
        },
        "tracking": {
            "enabled": not args.disable_trackio,
            "project": args.trackio_project,
            "space_id": args.trackio_space_id or None,
            "run_name": args.run_name,
        },
        "adapter_artifacts": _artifact_manifest(args.output_dir),
        "hub": hub,
    }
    _write_json(args.output_dir / "training_result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--model-revision", default="")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--trackio-project", default="hermes-flightrecorder-self-improving-proof")
    parser.add_argument("--trackio-space-id", default="")
    parser.add_argument("--run-name", default="qwen3-0.6b-hfr-self-improving-proof-v1")
    parser.add_argument("--disable-trackio", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id", default="")
    parser.add_argument("--private", action="store_true")
    return parser.parse_args()


def main() -> int:
    result = train(parse_args())
    print(json.dumps({"status": result["status"], "metrics": result["metrics"], "hub": result["hub"]}, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
