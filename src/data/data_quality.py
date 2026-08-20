"""
src/data/data_quality.py
Data quality checks before training.
"""
from __future__ import annotations
import logging
from typing import List, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


def run_data_quality_checks(
    df: pd.DataFrame,
    tool_list: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Run all data quality checks.
    Returns cleaned DataFrame and list of issues found.
    """
    issues = []
    original_len = len(df)

    # Check 1: Empty prompts
    empty = df["prompt"].str.strip().eq("").sum()
    if empty > 0:
        issues.append(f"Found {empty} empty prompts — removed")
        df = df[df["prompt"].str.strip() != ""].copy()

    # Check 2: Duplicate prompts (keep max label per tool)
    dupes = df.duplicated(subset=["prompt"]).sum()
    if dupes > 0:
        issues.append(f"Found {dupes} duplicate prompts — merged with OR logic")
        df = df.groupby("prompt", as_index=False)[tool_list].max()

    # Check 3: Rows with no tools selected (all zeros)
    no_tool = (df[tool_list].sum(axis=1) == 0).sum()
    if no_tool > 0:
        issues.append(f"Found {no_tool} prompts with no tool selected — kept (valid negative examples)")

    # Check 4: Tools with no positive examples
    for t in tool_list:
        pos = (df[t] == 1).sum()
        if pos == 0:
            issues.append(f"Tool '{t}' has ZERO positive examples — model cannot learn it")
        elif pos < 3:
            issues.append(f"Tool '{t}' has only {pos} positive examples — may underperform")

    # Check 5: Tools with no negative examples
    for t in tool_list:
        neg = (df[t] == 0).sum()
        if neg == 0:
            issues.append(f"Tool '{t}' has ZERO negative examples — model cannot learn it")

    # Check 6: Label validity
    for t in tool_list:
        invalid = (~df[t].isin([0, 1])).sum()
        if invalid > 0:
            issues.append(f"Tool '{t}' has {invalid} invalid labels — clipped to 0/1")
            df[t] = df[t].clip(0, 1)

    removed = original_len - len(df)
    if removed > 0:
        issues.append(f"Total rows removed: {removed}")

    print("\n" + "=" * 65)
    print("DATA QUALITY REPORT")
    print("=" * 65)
    if not issues:
        print("  No issues found.")
    for issue in issues:
        print(f"  [CHECK] {issue}")
    print(f"\n  Final dataset: {len(df)} prompts")
    print("=" * 65 + "\n")

    return df, issues
