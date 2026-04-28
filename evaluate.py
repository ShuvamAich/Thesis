"""
evaluate.py — Compute and persist evaluation metrics.
"""

import json
import os
import logging
from typing import List

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score

import config
from output_parser import ParsedOutput

logger = logging.getLogger(__name__)


def compute_metrics(samples_labels: List[int], predictions: List[ParsedOutput]) -> dict:
    y_true = np.array(samples_labels)
    valid_mask = np.array([prediction.predicted_int != -1 for prediction in predictions])
    if valid_mask.sum() == 0:
        logger.error("All predictions failed to parse. Cannot compute metrics.")
        return {
            "error": "All predictions failed to parse.",
            "num_samples_total": len(predictions),
            "num_samples_valid": 0,
            "num_samples_failed": len(predictions),
            "accuracy": None,
            "macro_f1": None,
            "auc_roc": None,
            "classification_report": None,
        }

    n_failed = int((~valid_mask).sum())
    y_true_v = y_true[valid_mask]
    y_pred_v = np.array([prediction.predicted_int for prediction in predictions])[valid_mask]
    y_conf_v = np.array([prediction.confidence for prediction in predictions])[valid_mask]

    accuracy = float(accuracy_score(y_true_v, y_pred_v))
    macro_f1 = float(f1_score(y_true_v, y_pred_v, average="macro", zero_division=0))

    auc_roc = None
    if len(np.unique(y_true_v)) == 2:
        try:
            auc_roc = float(roc_auc_score(y_true_v, y_conf_v))
        except ValueError as exc:
            logger.warning("Could not compute AUC-ROC: %s", exc)

    report = classification_report(
        y_true_v,
        y_pred_v,
        labels=[0, 1],
        target_names=["truthful", "deceptive"],
        zero_division=0,
    )

    return {
        "num_samples_total": len(predictions),
        "num_samples_valid": int(valid_mask.sum()),
        "num_samples_failed": n_failed,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "auc_roc": round(auc_roc, 4) if auc_roc is not None else None,
        "classification_report": report,
    }


def save_metrics(metrics: dict, path: str = config.METRICS_FILE) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    logger.info("Metrics saved to %s", path)


def save_predictions(sample_ids: List[str], ground_truths: List[int], predictions: List[ParsedOutput], path: str = config.PREDICTIONS_FILE) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    records = []
    for sample_id, ground_truth, prediction in zip(sample_ids, ground_truths, predictions):
        records.append(
            {
                "sample_id": sample_id,
                "ground_truth": config.LABEL_NAMES.get(ground_truth, "unknown"),
                "predicted_label": prediction.predicted_label,
                "confidence": prediction.confidence,
                "parse_status": prediction.parse_status,
                "correct": (prediction.predicted_int == ground_truth) if prediction.predicted_int != -1 else None,
                "raw_output": prediction.raw_output,
            }
        )
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
    logger.info("Per-sample predictions saved to %s", path)


def print_summary(metrics: dict) -> None:
    def fmt(value):
        return f"{value:.4f}" if isinstance(value, (int, float)) else "N/A"

    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Samples evaluated : {metrics.get('num_samples_valid')} / {metrics.get('num_samples_total')}")
    print(f"  Accuracy          : {fmt(metrics.get('accuracy'))}")
    print(f"  Macro-F1          : {fmt(metrics.get('macro_f1'))}")
    print(f"  AUC-ROC           : {fmt(metrics.get('auc_roc'))}")
    print("\n  Per-class report:")
    print(metrics.get("classification_report") or "N/A")
    if metrics.get("error"):
        print(f"\n  Error             : {metrics['error']}")
    print("=" * 60)