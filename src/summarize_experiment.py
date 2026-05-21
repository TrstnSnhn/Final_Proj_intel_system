from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


class ExperimentSummaryError(ValueError):
    """Raised when experiment summary inputs cannot be read."""


def build_experiment_summary(eval_summary_path: str | Path, training_log_path: str | Path) -> dict[str, Any]:
    eval_path = Path(eval_summary_path)
    log_path = Path(training_log_path)
    if not eval_path.exists():
        raise ExperimentSummaryError(f"Evaluation summary not found: {eval_path}")
    if not eval_path.is_file():
        raise ExperimentSummaryError(f"Evaluation summary path is not a file: {eval_path}")
    if not log_path.exists():
        raise ExperimentSummaryError(f"Training log not found: {log_path}")
    if not log_path.is_file():
        raise ExperimentSummaryError(f"Training log path is not a file: {log_path}")

    eval_payload = _load_eval_payload(eval_path)
    training_rows = _load_training_rows(log_path)
    if not training_rows:
        raise ExperimentSummaryError(f"Training log has no epoch rows: {log_path}")

    best_row = _best_row_by_metric(training_rows, "val_loss")
    last_row = training_rows[-1]
    return {
        "eval_summary": str(eval_path),
        "training_log": str(log_path),
        "epochs": len(training_rows),
        "best_val_loss": _float_from_row(best_row, "val_loss"),
        "best_val_loss_epoch": _int_from_row(best_row, "epoch"),
        "last_train_acc": _float_from_row(last_row, "train_acc"),
        "last_val_acc": _float_from_row(last_row, "val_acc"),
        "last_val_macro_f1": _float_from_row(last_row, "val_macro_f1"),
        "eval_samples": _int_from_payload(eval_payload, "num_samples"),
        "eval_correct": _int_from_payload(eval_payload, "correct"),
        "eval_accuracy": _float_from_payload(eval_payload, "accuracy"),
    }


def format_summary(summary: dict[str, Any]) -> str:
    lines = [
        "Experiment Summary",
        f"Training log: {summary['training_log']}",
        f"Evaluation summary: {summary['eval_summary']}",
        f"Epochs: {summary['epochs']}",
        f"Best val_loss: {summary['best_val_loss']:.4f} at epoch {summary['best_val_loss_epoch']}",
    ]
    if summary["last_train_acc"] is not None:
        lines.append(f"Last train accuracy: {summary['last_train_acc']:.2%}")
    if summary["last_val_acc"] is not None:
        lines.append(f"Last validation accuracy: {summary['last_val_acc']:.2%}")
    if summary["last_val_macro_f1"] is not None:
        lines.append(f"Last validation macro F1: {summary['last_val_macro_f1']:.4f}")
    lines.extend(
        [
            f"Test samples: {summary['eval_samples']}",
            f"Test correct: {summary['eval_correct']}",
            f"Accuracy: {summary['eval_accuracy']:.2%}",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print a concise PlantGuard experiment summary from generated local artifacts."
    )
    parser.add_argument("--eval-summary", required=True, help="Path to a generated evaluation summary JSON.")
    parser.add_argument("--training-log", required=True, help="Path to a generated training log CSV.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = build_experiment_summary(args.eval_summary, args.training_log)
    except (ExperimentSummaryError, json.JSONDecodeError, csv.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(format_summary(summary))
    return 0


def _load_eval_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ExperimentSummaryError(f"Evaluation summary must contain a JSON object: {path}")
    return payload


def _load_training_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _best_row_by_metric(rows: list[dict[str, str]], metric: str) -> dict[str, str]:
    try:
        return min(rows, key=lambda row: float(row[metric]))
    except KeyError as exc:
        raise ExperimentSummaryError(f"Training log is missing required column: {metric}") from exc
    except ValueError as exc:
        raise ExperimentSummaryError(f"Training log column must be numeric: {metric}") from exc


def _float_from_row(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    if value in {None, ""}:
        return None
    return float(value)


def _int_from_row(row: dict[str, str], key: str) -> int:
    value = row.get(key)
    if value in {None, ""}:
        raise ExperimentSummaryError(f"Training log is missing required column: {key}")
    return int(value)


def _float_from_payload(payload: dict[str, Any], key: str) -> float:
    try:
        return float(payload[key])
    except KeyError as exc:
        raise ExperimentSummaryError(f"Evaluation summary is missing required field: {key}") from exc


def _int_from_payload(payload: dict[str, Any], key: str) -> int:
    try:
        return int(payload[key])
    except KeyError as exc:
        raise ExperimentSummaryError(f"Evaluation summary is missing required field: {key}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
