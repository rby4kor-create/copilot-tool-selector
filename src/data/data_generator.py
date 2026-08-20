"""Synthetic data generator. Development only."""
from __future__ import annotations
import json
import logging
import sys
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_training_data() -> pd.DataFrame:
    print("\n[WARNING] Generating SYNTHETIC data — development only\n")
    import yaml
    with open(REPO_ROOT / "configs" / "config.yaml") as f:
        config = yaml.safe_load(f)

    tools = list(config["features"]["keyword_groups"].keys())
    prompts_path = REPO_ROOT / config["data"]["raw_dir"] / "prompts.json"

    with open(prompts_path) as f:
        prompts = json.load(f)
    logger.info("Loaded %d prompts", len(prompts))

    rows = []
    for item in prompts:
        text = item.get("prompt", "").strip()
        if not text:
            continue
        relevant = set(item.get("relevant_tools", []))
        for tool in tools:
            rows.append({"prompt": text, "tool": tool, "label": 1 if tool in relevant else 0})

    df = pd.DataFrame(rows)
    out = REPO_ROOT / config["data"]["synthetic_data_path"]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info("Generated %d rows -> %s", len(df), out)
    return df


if __name__ == "__main__":
    generate_training_data()
