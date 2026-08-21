"""
src/preprocessing.py
Manager dataset adapter. Auto-detects schema. Converts to canonical format.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
import pandas as pd
from src.config import Config

logger = logging.getLogger(__name__)
cfg = Config()
CANONICAL_TOOLS = cfg.all_tools()


def detect_schema(df: pd.DataFrame) -> str:
    cols = set(df.columns.str.lower().tolist())
    if "prompt" in cols and all(t in cols for t in CANONICAL_TOOLS):
        return "canonical"
    if "prompt" in cols and "tool" in cols and "label" in cols:
        return "paired_rows"
    if "prompt" in cols and ("tools" in cols or "relevant_tools" in cols):
        return "json_tools_col"
    return "unknown"


def from_paired_rows(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Converting paired-row format")
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()
    df = df[df["tool"].isin(set(CANONICAL_TOOLS))].copy()
    pivot = df.pivot_table(
        index="prompt", columns="tool",
        values="label", aggfunc="max", fill_value=0,
    ).reset_index()
    for t in CANONICAL_TOOLS:
        if t not in pivot.columns:
            pivot[t] = 0
    return pivot[["prompt"] + CANONICAL_TOOLS]


def from_json_tools_col(df: pd.DataFrame) -> pd.DataFrame:
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
    for t in CANONICAL_TOOLS:
        df[t] = df["_p"].apply(lambda x: int(t in x))
    return df[["prompt"] + CANONICAL_TOOLS].copy()


def validate_canonical(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in ["prompt"] + CANONICAL_TOOLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df.dropna(subset=["prompt"])
    df = df[df["prompt"].str.strip() != ""].copy()
    for t in CANONICAL_TOOLS:
        df[t] = pd.to_numeric(df[t], errors="coerce").fillna(0).astype(int).clip(0, 1)
    logger.info("Validated. Rows: %d", len(df))
    return df


def load_and_validate_dataset(path, source="synthetic") -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    logger.info("Loading %s dataset: %s", source, path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows, cols=%s", len(df), df.columns.tolist())
    schema = detect_schema(df)
    logger.info("Schema: %s", schema)
    if schema == "paired_rows":
        df = from_paired_rows(df)
    elif schema == "json_tools_col":
        df = from_json_tools_col(df)
    elif schema != "canonical":
        raise ValueError(f"Unknown schema. Columns: {df.columns.tolist()}")
    return validate_canonical(df)


def describe_dataset(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("DATASET DESCRIPTION")
    print("=" * 60)
    print(f"Shape : {df.shape}")
    print(f"Missing:\n{df.isnull().sum().to_string()}")
    print(f"Duplicate prompts: {df.duplicated(subset=['prompt']).sum()}")
    print("\nTool distribution:")
    for t in CANONICAL_TOOLS:
        pos = (df[t] == 1).sum()
        print(f"  {t:<25} SELECT={pos} ({100.0 * pos / len(df):.1f}%)")
    print("=" * 60 + "\n")
