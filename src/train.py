"""
src/train.py
Production training pipeline.
Multilabel: one LogisticRegression binary classifier per tool.
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import Config
from src.feature_engineering import PromptFeatureTransformer
from src.preprocessing import describe_dataset, load_and_validate_dataset
from src.model_classes import MultiLabelToolSelector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("train")


def tune_thresholds(selector, X_val, y_val_df: pd.DataFrame) -> Dict[str, float]:
    candidates = [round(t, 2) for t in np.arange(0.10, 0.95, 0.05)]
    logger.info("Tuning thresholds on validation set")
    probas = selector.predict_proba(X_val)
    best = {}
    for tool in selector.tools:
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
        logger.info("  %-25s threshold=%.2f val_F1=%.4f", tool, bt, bf)
        best[tool] = bt
    return best


def evaluate_model(selector, X_test, y_test_df, thresholds, split_name="test") -> Dict:
    probas = selector.predict_proba(X_test)
    all_yt, all_yp = [], []
    per_tool = {}
    print(f"\n{'='*65}")
    print(f"EVALUATION - {split_name.upper()}")
    print(f"{'='*65}")
    print(f"  {'Tool':<25} {'Thresh':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print("-" * 65)
    for tool in selector.tools:
        y_true = y_test_df[tool].values
        proba = probas[tool]
        t = thresholds.get(tool, 0.50)
        y_pred = (proba >= t).astype(int)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_true, proba) if y_true.sum() > 0 else float("nan")
        except Exception:
            auc = float("nan")
        print(f"  {tool:<25} {t:>7.2f} {prec:>7.4f} {rec:>7.4f} {f1:>7.4f}")
        per_tool[tool] = {
            "threshold": t, "precision": round(prec, 4),
            "recall": round(rec, 4), "f1": round(f1, 4),
            "tp": int(((y_pred == 1) & (y_true == 1)).sum()),
            "fp": int(((y_pred == 1) & (y_true == 0)).sum()),
            "tn": int(((y_pred == 0) & (y_true == 0)).sum()),
            "fn": int(((y_pred == 0) & (y_true == 1)).sum()),
        }
        all_yt.append(y_true)
        all_yp.append(y_pred)
    Yt = np.stack(all_yt, axis=1)
    Yp = np.stack(all_yp, axis=1)
    micro = f1_score(Yt, Yp, average="micro", zero_division=0)
    macro = f1_score(Yt, Yp, average="macro", zero_division=0)
    exact = float((Yt == Yp).all(axis=1).mean())
    sm = Yp == 1
    rm = Yp == 0
    sa = float((Yt[sm] == 1).mean()) if sm.sum() > 0 else float("nan")
    ra = float((Yt[rm] == 0).mean()) if rm.sum() > 0 else float("nan")
    print(f"\nMicro F1={micro:.4f} | Macro F1={macro:.4f} | Exact={exact:.4f}")
    print(f"Selection Acc={sa:.4f} | Rejection Acc={ra:.4f}")
    print("=" * 65 + "\n")
    return {
        "micro_f1": round(micro, 4), "macro_f1": round(macro, 4),
        "exact_match_accuracy": round(exact, 4),
        "tool_selection_accuracy": round(sa, 4) if not np.isnan(sa) else None,
        "tool_rejection_accuracy": round(ra, 4) if not np.isnan(ra) else None,
        "per_tool": per_tool,
    }


def train(source: Optional[str] = None) -> None:
    t0 = time.time()
    run_id = str(uuid.uuid4())[:8]
    config = Config()
    logger.info("Training run: %s", run_id)

    effective = source or config.training_source
    if effective == "synthetic":
        df = load_and_validate_dataset(config.synthetic_data_path, "synthetic")
    elif effective == "manager":
        df = load_and_validate_dataset(config.manager_data_path, "manager")
    elif effective == "combined":
        d1 = load_and_validate_dataset(config.synthetic_data_path, "synthetic")
        d2 = load_and_validate_dataset(config.manager_data_path, "manager")
        df = pd.concat([d1, d2], ignore_index=True)
    else:
        raise ValueError(f"Unknown source: {effective}")

    describe_dataset(df)
    tools = config.all_tools()

    before = len(df)
    df = df.groupby("prompt", as_index=False)[tools].max()
    logger.info("Deduplicated %d -> %d", before, len(df))

    prompts = df["prompt"].tolist()
    y_df = df[tools]

    p_tv, p_te, y_tv, y_te = train_test_split(
        prompts, y_df, test_size=config.test_size,
        random_state=config.split_random_state)
    vf = config.validation_size / (1.0 - config.test_size)
    p_tr, p_va, y_tr, y_va = train_test_split(
        p_tv, y_tv, test_size=vf,
        random_state=config.split_random_state)

    y_tr_df = pd.DataFrame(y_tr, columns=tools)
    y_va_df = pd.DataFrame(y_va, columns=tools)
    y_te_df = pd.DataFrame(y_te, columns=tools)
    logger.info("Split: train=%d val=%d test=%d", len(p_tr), len(p_va), len(p_te))

    transformer = PromptFeatureTransformer(config)
    X_tr = transformer.fit_transform(p_tr)
    X_va = transformer.transform(p_va)
    X_te = transformer.transform(p_te)
    logger.info("Feature shape: %s", X_tr.shape)

    selector = MultiLabelToolSelector(tools=tools, lr_config=config.lr_config)
    selector.fit(X_tr, y_tr_df)

    thresholds = tune_thresholds(selector, X_va, y_va_df)
    metrics = evaluate_model(selector, X_te, y_te_df, thresholds)

    config.ensure_dirs()
    joblib.dump(
        {"transformer": transformer, "selector": selector,
         "thresholds": thresholds, "tools": tools},
        config.model_artifact_path,
    )
    logger.info("Model saved: %s", config.model_artifact_path)

    meta = {
        "run_id": run_id, "model_version": config.model_version,
        "training_source": effective, "tools": tools,
        "thresholds": thresholds, "train_size": len(p_tr),
        "val_size": len(p_va), "test_size": len(p_te),
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "training_duration_seconds": round(time.time() - t0, 2),
        "evaluation_metrics": metrics,
    }
    with open(config.model_metadata_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    logger.info("Done %.1fs | Micro F1=%.4f", time.time() - t0, metrics["micro_f1"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["synthetic", "manager", "combined"], default=None)
    args = parser.parse_args()
    train(source=args.source)
