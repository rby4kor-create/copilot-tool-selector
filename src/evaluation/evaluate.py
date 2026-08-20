"""
src/evaluation/evaluate.py
Standalone evaluation on any dataset.
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

import joblib

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.preprocessing import load_and_validate_dataset, describe_dataset
from src.training.trainer import evaluate_selector
from src.models.model_classes import MultiLabelToolSelector
from src.features.feature_engineering import PromptFeatureTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("evaluate")


def evaluate(data_path: str = None) -> None:
    import yaml
    config_path = REPO_ROOT / "configs" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    artifact_path = REPO_ROOT / config["model"]["output_dir"] / config["model"]["artifact_name"]
    meta_path = REPO_ROOT / config["model"]["output_dir"] / config["model"]["metadata_name"]

    if not artifact_path.exists():
        logger.error("No model at: %s", artifact_path)
        logger.error("Run: python src/training/train.py")
        sys.exit(1)

    artifact = joblib.load(artifact_path)
    transformer = artifact["transformer"]
    selector = artifact["selector"]
    thresholds = artifact["thresholds"]
    tools = artifact["tools"]

    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"\nModel v{meta.get('model_version')} | {meta.get('algorithm')} | {meta.get('training_timestamp')}")

    eval_path = Path(data_path) if data_path else REPO_ROOT / config["data"]["synthetic_data_path"]
    df = load_and_validate_dataset(eval_path, tools)
    describe_dataset(df, tools)

    X = transformer.transform(df["prompt"].tolist())
    evaluate_selector(selector, X, df[tools], thresholds, tools, "full evaluation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None)
    args = parser.parse_args()
    evaluate(data_path=args.data)
