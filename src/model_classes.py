"""
src/model_classes.py
Contains MultiLabelToolSelector class.
Must be a separate importable module so joblib can deserialize it
regardless of which script is the __main__ entry point.
"""
from __future__ import annotations
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class MultiLabelToolSelector:
    """One LogisticRegression binary classifier per tool."""

    def __init__(self, tools: List[str], lr_config: Dict) -> None:
        self.tools = tools
        self.lr_config = lr_config
        self.classifiers_: Dict[str, LogisticRegression] = {}

    def fit(self, X, y_df: pd.DataFrame):
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Training %d classifiers", len(self.tools))
        for tool in self.tools:
            y = y_df[tool].values
            logger.info("  %-25s pos=%d neg=%d", tool, y.sum(), len(y) - y.sum())
            clf = LogisticRegression(
                C=self.lr_config["C"],
                max_iter=self.lr_config["max_iter"],
                solver=self.lr_config["solver"],
                class_weight=self.lr_config.get("class_weight", "balanced"),
                random_state=self.lr_config["random_state"],
            )
            clf.fit(X, y)
            self.classifiers_[tool] = clf
        return self

    def predict_proba(self, X) -> Dict[str, np.ndarray]:
        return {t: clf.predict_proba(X)[:, 1] for t, clf in self.classifiers_.items()}

    def predict(self, X, thresholds: Dict[str, float]) -> Dict[str, np.ndarray]:
        p = self.predict_proba(X)
        return {t: (p[t] >= thresholds.get(t, 0.5)).astype(int) for t in self.tools}
