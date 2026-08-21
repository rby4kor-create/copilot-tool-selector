"""
src/evaluate.py
Standalone evaluation. Loads trained model. Evaluates on dataset.
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import Config
from src.preprocessing import load_and_validate_dataset, describe_dataset
from src.train import evaluate_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("evaluate")


def evaluate(data_path=None) -> None:
    config = Config()
    if not config.model_artifact_path.exists():
        logger.error("No model at: %s  Run: python src/train.py", config.model_artifact_path)
        sys.exit(1)
    artifact = joblib.load(config.model_artifact_path)
    transformer = artifact["transformer"]
    selector = artifact["selector"]
    thresholds = artifact["thresholds"]
    tools = artifact["tools"]
    if config.model_metadata_path.exists():
        with open(config.model_metadata_path) as f:
            meta = json.load(f)
        print(f"Model v{meta.get('model_version')} | {meta.get('training_timestamp')}")
    path = Path(data_path) if data_path else config.synthetic_data_path
    df = load_and_validate_dataset(path)
    describe_dataset(df)
    X = transformer.transform(df["prompt"].tolist())
    evaluate_model(selector, X, df[tools], thresholds, "evaluation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None)
    args = parser.parse_args()
    evaluate(data_path=args.data)
