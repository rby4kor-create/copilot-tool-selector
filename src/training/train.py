"""
src/training/train.py
A-to-Z production training pipeline.

Phases:
    1. Load and validate dataset
    2. Data quality check
    3. Train/Val/Test split
    4. Feature engineering (fit on train only)
    5. Train Logistic Regression
    6. Train Linear SVM
    7. Compare models objectively
    8. Select best model
    9. Save artifact + metadata

Usage:
    python src/training/train.py
    python src/training/train.py --source manager
    python src/training/train.py --algorithm logistic_regression
    python src/training/train.py --algorithm linear_svm
    python src/training/train.py --algorithm compare  (trains both, picks best)
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
from sklearn.model_selection import train_test_split

# Allow running from repo root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.preprocessing import load_and_validate_dataset, describe_dataset
from src.data.data_quality import run_data_quality_checks
from src.features.feature_engineering import PromptFeatureTransformer
from src.models.model_classes import MultiLabelToolSelector
from src.training.trainer import tune_thresholds, evaluate_selector, run_experiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("train")


def load_config() -> Dict:
    config_path = REPO_ROOT / "configs" / "config.yaml"
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def train(source: str = None, algorithm: str = "compare") -> None:
    t_start = time.time()
    run_id = str(uuid.uuid4())[:8]
    config = load_config()

    logger.info("=" * 60)
    logger.info("Training run: %s | Algorithm: %s", run_id, algorithm)
    logger.info("=" * 60)

    # ── Load tool catalog ─────────────────────────────────────────────────────
    catalog_path = REPO_ROOT / config["data"]["tool_catalog_path"]
    if not catalog_path.exists():
        logger.warning("Tool catalog not found — creating from config")
        _create_catalog(config, catalog_path)

    with open(catalog_path) as f:
        raw_catalog = json.load(f)
    tool_list = [k for k in raw_catalog if not k.startswith("__")]
    logger.info("Tools: %d → %s", len(tool_list), tool_list)

    # ── Load dataset ──────────────────────────────────────────────────────────
    effective_source = source or config["data"]["training_source"]
    if effective_source == "synthetic":
        data_path = REPO_ROOT / config["data"]["synthetic_data_path"]
    elif effective_source == "manager":
        data_path = REPO_ROOT / config["data"]["manager_data_path"]
    else:
        raise ValueError(f"Unknown source: {effective_source}")

    df = load_and_validate_dataset(data_path, tool_list, effective_source)
    describe_dataset(df, tool_list)

    # ── Data quality ──────────────────────────────────────────────────────────
    df, issues = run_data_quality_checks(df, tool_list)

    # ── Split ─────────────────────────────────────────────────────────────────
    prompts = df["prompt"].tolist()
    y_df = df[tool_list]
    test_size = config["split"]["test_size"]
    val_size = config["split"]["validation_size"]
    seed = config["split"]["random_state"]

    p_tv, p_te, y_tv, y_te = train_test_split(
        prompts, y_df, test_size=test_size, random_state=seed)
    vf = val_size / (1.0 - test_size)
    p_tr, p_va, y_tr, y_va = train_test_split(
        p_tv, y_tv, test_size=vf, random_state=seed)

    y_tr_df = pd.DataFrame(y_tr, columns=tool_list)
    y_va_df = pd.DataFrame(y_va, columns=tool_list)
    y_te_df = pd.DataFrame(y_te, columns=tool_list)

    logger.info(
        "Split: train=%d  val=%d  test=%d",
        len(p_tr), len(p_va), len(p_te),
    )

    # ── Feature engineering ───────────────────────────────────────────────────
    logger.info("Fitting feature transformer on TRAINING data only")
    transformer = PromptFeatureTransformer(
        tfidf_config=config["features"]["tfidf"],
        keyword_groups=config["features"]["keyword_groups"],
    )
    X_tr = transformer.fit_transform(p_tr)
    X_va = transformer.transform(p_va)
    X_te = transformer.transform(p_te)
    logger.info("Feature shape: train=%s val=%s test=%s",
                X_tr.shape, X_va.shape, X_te.shape)

    experiments_dir = REPO_ROOT / config["experiments"]["output_dir"]

    # ── Train models ──────────────────────────────────────────────────────────
    lr_config = config["model"]["logistic_regression"]
    svm_config = config["model"]["linear_svm"]

    results = {}

    if algorithm in ("logistic_regression", "compare"):
        lr_metrics, lr_thresholds, lr_selector = run_experiment(
            experiment_id=f"{run_id}_LR",
            algorithm="logistic_regression",
            X_train=X_tr, X_val=X_va, X_test=X_te,
            y_train_df=y_tr_df, y_val_df=y_va_df, y_test_df=y_te_df,
            tool_list=tool_list,
            lr_config=lr_config, svm_config=svm_config,
            experiments_dir=experiments_dir,
        )
        results["logistic_regression"] = {
            "metrics": lr_metrics,
            "thresholds": lr_thresholds,
            "selector": lr_selector,
        }

    if algorithm in ("linear_svm", "compare"):
        svm_metrics, svm_thresholds, svm_selector = run_experiment(
            experiment_id=f"{run_id}_SVM",
            algorithm="linear_svm",
            X_train=X_tr, X_val=X_va, X_test=X_te,
            y_train_df=y_tr_df, y_val_df=y_va_df, y_test_df=y_te_df,
            tool_list=tool_list,
            lr_config=lr_config, svm_config=svm_config,
            experiments_dir=experiments_dir,
        )
        results["linear_svm"] = {
            "metrics": svm_metrics,
            "thresholds": svm_thresholds,
            "selector": svm_selector,
        }

    # ── Select best model ─────────────────────────────────────────────────────
    if len(results) == 1:
        best_algo = list(results.keys())[0]
    else:
        # Primary: Micro F1. Tiebreaker: Macro F1.
        best_algo = max(
            results,
            key=lambda a: (
                results[a]["metrics"]["micro_f1"],
                results[a]["metrics"]["macro_f1"],
            ),
        )

    best = results[best_algo]
    best_metrics = best["metrics"]
    best_thresholds = best["thresholds"]
    best_selector = best["selector"]

    # ── Print comparison ──────────────────────────────────────────────────────
    if len(results) > 1:
        print("\n" + "=" * 70)
        print("MODEL COMPARISON")
        print("=" * 70)
        print(f"  {'Algorithm':<25} {'Micro F1':>10} {'Macro F1':>10} {'Precision':>10} {'Recall':>10}")
        print("-" * 70)
        for algo, r in results.items():
            m = r["metrics"]
            marker = " ← SELECTED" if algo == best_algo else ""
            print(f"  {algo:<25} {m['micro_f1']:>10.4f} {m['macro_f1']:>10.4f} "
                  f"{m['micro_precision']:>10.4f} {m['micro_recall']:>10.4f}{marker}")
        print("=" * 70)
        print(f"\nSelected: {best_algo}")
        print(f"Reason: Highest Micro F1 = {best_metrics['micro_f1']:.4f}")
        print()

    # ── Save artifact ─────────────────────────────────────────────────────────
    output_dir = REPO_ROOT / config["model"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = output_dir / config["model"]["artifact_name"]
    artifact = {
        "transformer": transformer,
        "selector": best_selector,
        "thresholds": best_thresholds,
        "tools": tool_list,
    }
    joblib.dump(artifact, artifact_path)
    logger.info("Model saved: %s", artifact_path)

    # ── Save metadata ─────────────────────────────────────────────────────────
    meta = {
        "run_id": run_id,
        "model_version": config["model"]["version"],
        "algorithm": best_algo,
        "training_source": effective_source,
        "tools": tool_list,
        "num_tools": len(tool_list),
        "thresholds": best_thresholds,
        "train_size": len(p_tr),
        "val_size": len(p_va),
        "test_size": len(p_te),
        "feature_count": transformer.get_feature_count(),
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_training_seconds": round(time.time() - t_start, 2),
        "data_quality_issues": issues,
        "evaluation_metrics": best_metrics,
        "all_experiment_results": {
            algo: r["metrics"] for algo, r in results.items()
        },
    }
    meta_path = output_dir / config["model"]["metadata_name"]
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    logger.info("Metadata saved: %s", meta_path)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Algorithm     : {best_algo}")
    print(f"  Tools         : {len(tool_list)}")
    print(f"  Train prompts : {len(p_tr)}")
    print(f"  Micro F1      : {best_metrics['micro_f1']:.4f}")
    print(f"  Macro F1      : {best_metrics['macro_f1']:.4f}")
    print(f"  Exact Match   : {best_metrics['exact_match']:.4f}")
    print(f"  Tool Reduction: {best_metrics['tool_reduction']:.1%}")
    print(f"  Model saved   : {artifact_path}")
    print("=" * 70 + "\n")


def _create_catalog(config: Dict, output_path: Path) -> None:
    """Create tool catalog from config keyword groups."""
    from datetime import datetime, timezone
    catalog = {}
    for tool, keywords in config["features"]["keyword_groups"].items():
        catalog[tool] = {
            "description": f"Tool: {tool}",
            "category": "auto",
            "keywords": keywords,
        }
    catalog["__meta__"] = {
        "version": config["data"]["tool_catalog_version"],
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        import json
        json.dump(catalog, f, indent=2)
    logger.info("Created tool catalog: %s", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML tool selector")
    parser.add_argument("--source", choices=["synthetic", "manager"], default=None)
    parser.add_argument(
        "--algorithm",
        choices=["logistic_regression", "linear_svm", "compare"],
        default="compare",
        help="compare trains both and picks best",
    )
    args = parser.parse_args()
    train(source=args.source, algorithm=args.algorithm)
