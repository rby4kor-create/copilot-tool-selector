"""
src/data_generator.py
SYNTHETIC DATA GENERATOR - Development only.
For production use the manager-approved dataset.
"""
from __future__ import annotations
import json
import logging
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_training_data() -> pd.DataFrame:
    print("\n[WARNING] Generating SYNTHETIC data for development only.\n")
    config = Config()
    tools = config.all_tools()
    prompts_path = config.raw_data_dir / "prompts.json"
    if not prompts_path.exists():
        raise FileNotFoundError(f"Not found: {prompts_path}")
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
    out = config.synthetic_data_path
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info("Generated %d rows -> %s", len(df), out)
    return df


if __name__ == "__main__":
    generate_training_data()
