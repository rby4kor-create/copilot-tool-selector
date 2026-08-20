"""
src/training/trainer.py
Production training pipeline.
Trains Logistic Regression and Linear SVM.
Compares them. Selects the best.
"""
from __future__ import annotations
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, accuracy_score,
)
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def tune_thresholds(
    selector,
    X_val,
    y_val_df: pd.DataFrame,
    tool_list: List[str],
) -> Dict[str, float]:
    """Find best threshold per tool using validation set."""
    candidates = [round(t, 2) for t in np.arange(0.10, 0.95, 0.05)]
    probas = selector.predict_proba(X_val)
    best = {}
    logger.info("Tuning thresholds on validation set")
    for tool in tool_list:
        y_true = y_val_df[tool].values
        proba = probas[tool]
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            best[tool] = 0.50
            continue
        bf, bt = -1.0, 0.50
        for t in candidates:
            f = f1_score(y_true, (proba >= t).astype(int), zero_division=0)
            if f > bf:
                bf, bt = f, t
        # Safety: never let threshold go below 0.30 to prevent false positives
        bt = max(bt, 0.30)
        logger.info("  %-25s threshold=%.2f val_F1=%.4f", tool, bt, bf)
        best[tool] = bt
    return best


def evaluate_selector(
    selector,
    X: np.ndarray,
    y_df: pd.DataFrame,
    thresholds: Dict[str, float],
    tool_list: List[str],
    split_name: str = "test",
    verbose: bool = True,
) -> Dict:
    """Full multilabel evaluation."""
    probas = selector.predict_proba(X)
    all_yt, all_yp = [], []
    per_tool = {}

    if verbose:
        print(f"\n{'='*70}")
        print(f"EVALUATION — {split_name.upper()}")
        print(f"{'='*70}")
        print(f"  {'Tool':<25} {'T':>5} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7} {'TP':>5} {'FP':>5} {'FN':>5}")
        print("-" * 70)

    for tool in tool_list:
        y_true = y_df[tool].values
        proba = probas[tool]
        t = thresholds.get(tool, 0.50)
        y_pred = (proba >= t).astype(int)

        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_true, proba) if y_true.sum() > 0 and y_true.sum() < len(y_true) else float("nan")
        except Exception:
            auc = float("nan")

        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())

        if verbose:
            auc_str = f"{auc:.4f}" if not np.isnan(auc) else "  N/A "
            print(f"  {tool:<25} {t:>5.2f} {prec:>7.4f} {rec:>7.4f} {f1:>7.4f} {auc_str:>7} {tp:>5} {fp:>5} {fn:>5}")

        per_tool[tool] = {
            "threshold": t, "precision": round(prec, 4),
            "recall": round(rec, 4), "f1": round(f1, 4),
            "auc": round(auc, 4) if not np.isnan(auc) else None,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        }
        all_yt.append(y_true)
        all_yp.append(y_pred)

    Yt = np.stack(all_yt, axis=1)
    Yp = np.stack(all_yp, axis=1)

    micro_f1 = f1_score(Yt, Yp, average="micro", zero_division=0)
    macro_f1 = f1_score(Yt, Yp, average="macro", zero_division=0)
    weighted_f1 = f1_score(Yt, Yp, average="weighted", zero_division=0)
    micro_prec = precision_score(Yt, Yp, average="micro", zero_division=0)
    micro_rec = recall_score(Yt, Yp, average="micro", zero_division=0)
    exact = float((Yt == Yp).all(axis=1).mean())

    sm, rm = Yp == 1, Yp == 0
    sel_acc = float((Yt[sm] == 1).mean()) if sm.sum() > 0 else float("nan")
    rej_acc = float((Yt[rm] == 0).mean()) if rm.sum() > 0 else float("nan")

    total_possible = Yt.shape[0] * Yt.shape[1]
    total_selected = Yp.sum()
    tool_reduction = 1.0 - (total_selected / total_possible) if total_possible > 0 else 0.0
    avg_selected = Yp.sum(axis=1).mean()

    if verbose:
        print(f"{'─'*70}")
        print(f"  Micro  F1        : {micro_f1:.4f}")
        print(f"  Macro  F1        : {macro_f1:.4f}")
        print(f"  Weighted F1      : {weighted_f1:.4f}")
        print(f"  Micro  Precision : {micro_prec:.4f}")
        print(f"  Micro  Recall    : {micro_rec:.4f}")
        print(f"  Exact Match      : {exact:.4f}")
        print(f"  Selection Acc    : {sel_acc:.4f}  (when we SELECT, are we right?)")
        print(f"  Rejection Acc    : {rej_acc:.4f}  (when we DESELECT, are we right?)")
        print(f"  Tool Reduction   : {tool_reduction:.1%}  (fewer tools = less cost)")
        print(f"  Avg Tools/Prompt : {avg_selected:.2f}")
        print("=" * 70 + "\n")

    return {
        "micro_f1": round(micro_f1, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "micro_precision": round(micro_prec, 4),
        "micro_recall": round(micro_rec, 4),
        "exact_match": round(exact, 4),
        "tool_selection_accuracy": round(sel_acc, 4) if not np.isnan(sel_acc) else None,
        "tool_rejection_accuracy": round(rej_acc, 4) if not np.isnan(rej_acc) else None,
        "tool_reduction": round(tool_reduction, 4),
        "avg_tools_per_prompt": round(float(avg_selected), 4),
        "per_tool": per_tool,
    }


def run_experiment(
    experiment_id: str,
    algorithm: str,
    X_train, X_val, X_test,
    y_train_df, y_val_df, y_test_df,
    tool_list: List[str],
    lr_config: Dict,
    svm_config: Dict,
    experiments_dir: Path,
) -> Tuple[Dict, Dict, object]:
    """Run one complete experiment. Returns metrics, thresholds, selector."""
    from src.models.model_classes import MultiLabelToolSelector

    print(f"\n{'#'*70}")
    print(f"# EXPERIMENT: {experiment_id}  |  Algorithm: {algorithm}")
    print(f"{'#'*70}")

    t0 = time.time()
    selector = MultiLabelToolSelector(
        tools=tool_list,
        algorithm=algorithm,
        lr_config=lr_config,
        svm_config=svm_config,
    )
    selector.fit(X_train, y_train_df)
    train_time = time.time() - t0

    thresholds = tune_thresholds(selector, X_val, y_val_df, tool_list)

    t1 = time.time()
    metrics = evaluate_selector(
        selector, X_test, y_test_df, thresholds, tool_list,
        split_name=f"{experiment_id} ({algorithm})",
    )
    infer_time = (time.time() - t1) / len(y_test_df) * 1000

    metrics["training_time_seconds"] = round(train_time, 3)
    metrics["inference_latency_ms_per_prompt"] = round(infer_time, 3)
    metrics["algorithm"] = algorithm
    metrics["experiment_id"] = experiment_id

    # Save experiment record
    experiments_dir.mkdir(parents=True, exist_ok=True)
    exp_file = experiments_dir / f"{experiment_id}.json"
    with open(exp_file, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Experiment saved: {exp_file}")

    return metrics, thresholds, selector
