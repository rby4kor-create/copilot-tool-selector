"""
src/models/model_classes.py
ML model classes. Separate module so joblib can always deserialize
regardless of which script is __main__.
"""
from __future__ import annotations
import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

logger = logging.getLogger(__name__)


class MultiLabelToolSelector:
    """
    One binary classifier per tool (One-vs-Rest multilabel).

    Supports:
        - LogisticRegression  (provides probabilities natively)
        - LinearSVC           (calibrated for probability estimates)

    Each classifier independently answers:
        'Should this tool be selected for this prompt?'
    """

    def __init__(
        self,
        tools: List[str],
        algorithm: str = "logistic_regression",
        lr_config: Optional[Dict] = None,
        svm_config: Optional[Dict] = None,
    ) -> None:
        self.tools = tools
        self.algorithm = algorithm
        self.lr_config = lr_config or {}
        self.svm_config = svm_config or {}
        self.classifiers_: Dict[str, object] = {}

    def _make_classifier(self):
        if self.algorithm == "logistic_regression":
            return LogisticRegression(
                C=self.lr_config.get("C", 1.0),
                max_iter=self.lr_config.get("max_iter", 2000),
                solver=self.lr_config.get("solver", "lbfgs"),
                class_weight=self.lr_config.get("class_weight", "balanced"),
                random_state=self.lr_config.get("random_state", 42),
            )
        elif self.algorithm == "linear_svm":
            base = LinearSVC(
                C=self.svm_config.get("C", 1.0),
                max_iter=self.svm_config.get("max_iter", 2000),
                class_weight=self.svm_config.get("class_weight", "balanced"),
                random_state=self.svm_config.get("random_state", 42),
            )
            # Calibrate SVM to produce probability estimates
            return CalibratedClassifierCV(base, cv=3, method="sigmoid")
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def fit(self, X, y_df: pd.DataFrame):
        logger.info(
            "Training %d %s classifiers (one per tool)",
            len(self.tools), self.algorithm,
        )
        for tool in self.tools:
            y = y_df[tool].values
            pos = y.sum()
            neg = len(y) - pos
            logger.info("  %-25s pos=%d neg=%d", tool, pos, neg)
            if pos == 0:
                logger.warning("  %-25s has NO positive examples — skipping", tool)
                self.classifiers_[tool] = None
                continue
            if neg == 0:
                logger.warning("  %-25s has NO negative examples — skipping", tool)
                self.classifiers_[tool] = None
                continue
            clf = self._make_classifier()
            clf.fit(X, y)
            self.classifiers_[tool] = clf
        return self

    def predict_proba(self, X) -> Dict[str, np.ndarray]:
        result = {}
        for tool, clf in self.classifiers_.items():
            if clf is None:
                result[tool] = np.zeros(X.shape[0])
            else:
                result[tool] = clf.predict_proba(X)[:, 1]
        return result

    def predict(self, X, thresholds: Dict[str, float]) -> Dict[str, np.ndarray]:
        probas = self.predict_proba(X)
        return {
            t: (probas[t] >= thresholds.get(t, 0.5)).astype(int)
            for t in self.tools
        }
