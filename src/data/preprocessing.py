"""
src/data/preprocessing.py
Dataset loading, schema detection, and canonical format conversion.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import List
import pandas as pd

logger = logging.getLogger(__name__)


def detect_schema(df: pd.DataFrame, tool_list: List[str]) -> str:
    cols = set(df.columns.str.lower().tolist())
    if "prompt" in cols and all(t in cols for t in tool_list):
        return "canonical"
    if "prompt" in cols and "tool" in cols and "label" in cols:
        return "paired_rows"
    if "prompt" in cols and ("tools" in cols or "relevant_tools" in cols):
        return "json_tools_col"
    return "unknown"


def from_paired_rows(df: pd.DataFrame, tool_list: List[str]) -> pd.DataFrame:
    logger.info("Converting paired-row format")
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()
    df = df[df["tool"].isin(set(tool_list))].copy()
    pivot = df.pivot_table(
        index="prompt", columns="tool",
        values="label", aggfunc="max", fill_value=0,
    ).reset_index()
    for t in tool_list:
        if t not in pivot.columns:
            pivot[t] = 0
    return pivot[["prompt"] + tool_list]


def from_json_tools_col(df: pd.DataFrame, tool_list: List[str]) -> pd.DataFrame:
    logger.info("Converting json-tools-col format")
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()
    col = "tools" if "tools" in df.columns else "relevant_tools"

    def parse(val):
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            try:
                return json.loads(val)
            except Exception:
                return [v.strip() for v in val.split(",") if v.strip()]
        return []

    df["_p"] = df[col].apply(parse)
    for t in tool_list:
        df[t] = df["_p"].apply(lambda x: int(t in x))
    return df[["prompt"] + tool_list].copy()


def validate_canonical(df: pd.DataFrame, tool_list: List[str]) -> pd.DataFrame:
    missing = [c for c in ["prompt"] + tool_list if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df.dropna(subset=["prompt"])
    df = df[df["prompt"].str.strip() != ""].copy()
    for t in tool_list:
        df[t] = pd.to_numeric(df[t], errors="coerce").fillna(0).astype(int).clip(0, 1)
    return df


def load_and_validate_dataset(
    path: str | Path,
    tool_list: List[str],
    source: str = "unknown",
) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    logger.info("Loading %s dataset: %s", source, path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows, cols=%s", len(df), df.columns.tolist())
    schema = detect_schema(df, tool_list)
    logger.info("Detected schema: %s", schema)
    if schema == "paired_rows":
        df = from_paired_rows(df, tool_list)
    elif schema == "json_tools_col":
        df = from_json_tools_col(df, tool_list)
    elif schema != "canonical":
        raise ValueError(f"Unknown schema. Columns: {df.columns.tolist()}")
    return validate_canonical(df, tool_list)


def describe_dataset(df: pd.DataFrame, tool_list: List[str]) -> None:
    print("\n" + "=" * 65)
    print("DATASET DESCRIPTION")
    print("=" * 65)
    print(f"Shape            : {df.shape}")
    print(f"Unique prompts   : {df['prompt'].nunique()}")
    print(f"Duplicate prompts: {df.duplicated(subset=['prompt']).sum()}")
    print(f"Missing values   : {df.isnull().sum().sum()}")
    print("\nTool label distribution:")
    print(f"  {'Tool':<25} {'SELECT':>8} {'DESELECT':>10} {'%SELECT':>9}")
    print("-" * 55)
    for t in tool_list:
        pos = (df[t] == 1).sum()
        neg = (df[t] == 0).sum()
        pct = 100.0 * pos / len(df) if len(df) > 0 else 0
        print(f"  {t:<25} {pos:>8} {neg:>10} {pct:>8.1f}%")
    n_tools_per_prompt = df[tool_list].sum(axis=1)
    print(f"\nAvg tools per prompt : {n_tools_per_prompt.mean():.2f}")
    print(f"Max tools per prompt : {n_tools_per_prompt.max()}")
    print(f"Multi-tool prompts   : {(n_tools_per_prompt > 1).sum()}")
    print("=" * 65 + "\n")
