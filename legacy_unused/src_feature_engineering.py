"""
src/feature_engineering.py
Production feature pipeline.
Fit ONCE during training. Load and reuse during inference. Never refit.
"""
from __future__ import annotations
import logging
import re
from typing import Dict, List
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from src.config import Config

logger = logging.getLogger(__name__)


def normalize_prompt(text: str) -> str:
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


class KeywordFeatureExtractor(BaseEstimator, TransformerMixin):
    """One binary flag per tool keyword group."""

    def __init__(self, keyword_groups: Dict[str, List[str]]) -> None:
        self.keyword_groups = keyword_groups
        self.feature_names_: List[str] = []

    def fit(self, X: List[str], y=None):
        self.feature_names_ = list(self.keyword_groups.keys())
        return self

    def transform(self, X: List[str]) -> np.ndarray:
        n = len(X)
        k = len(self.keyword_groups)
        result = np.zeros((n, k), dtype=np.float32)
        tool_names = list(self.keyword_groups.keys())
        for i, prompt in enumerate(X):
            pl = prompt.lower()
            for j, tool in enumerate(tool_names):
                if any(kw.lower() in pl for kw in self.keyword_groups[tool]):
                    result[i, j] = 1.0
        return result

    def get_feature_names_out(self) -> List[str]:
        return [f"keyword__{n}" for n in self.feature_names_]


class StructuralFeatureExtractor(BaseEstimator, TransformerMixin):
    """Hand-crafted structural features from the prompt text."""

    ACTION_VERBS = {
        "find", "search", "get", "show", "list", "read", "run", "execute",
        "edit", "create", "write", "update", "delete", "modify", "check",
        "view", "open", "look", "locate", "what", "how", "where", "display",
    }
    FILE_EXT = re.compile(
        r"\.(py|js|ts|java|cpp|c|cs|go|rb|rs|md|json|yaml|yml|toml|txt)$",
        re.IGNORECASE,
    )

    def fit(self, X, y=None):
        return self

    def transform(self, X: List[str]) -> np.ndarray:
        result = np.zeros((len(X), 8), dtype=np.float32)
        for i, prompt in enumerate(X):
            words = prompt.split()
            result[i, 0] = min(len(prompt) / 200.0, 1.0)
            result[i, 1] = min(len(words) / 30.0, 1.0)
            result[i, 2] = float("(" in prompt or ")" in prompt)
            result[i, 3] = float("." in prompt)
            result[i, 4] = float("_" in prompt)
            result[i, 5] = float("?" in prompt)
            result[i, 6] = float(bool(self.FILE_EXT.search(prompt)))
            result[i, 7] = float(bool(words) and words[0].lower() in self.ACTION_VERBS)
        return result

    def get_feature_names_out(self) -> List[str]:
        return [
            "struct__char_length", "struct__word_count",
            "struct__has_parentheses", "struct__has_dot_notation",
            "struct__has_underscore", "struct__has_question_mark",
            "struct__has_file_extension", "struct__starts_with_verb",
        ]


class PromptNormalizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X: List[str]) -> List[str]:
        return [normalize_prompt(p) for p in X]


class PromptFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Full feature pipeline: TF-IDF + Keyword + Structural.
    Fit once during training. Persist. Reuse during inference.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._build()

    def _build(self) -> None:
        tc = self.config.tfidf_config
        ngram = tuple(tc["ngram_range"])
        self.normalizer_ = PromptNormalizer()
        self.tfidf_ = TfidfVectorizer(
            max_features=tc["max_features"],
            ngram_range=ngram,
            min_df=tc["min_df"],
            sublinear_tf=tc["sublinear_tf"],
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b",
        )
        self.keyword_ = KeywordFeatureExtractor(self.config.keyword_groups)
        self.structural_ = StructuralFeatureExtractor()

    def fit(self, X: List[str], y=None):
        norm = self.normalizer_.transform(X)
        self.tfidf_.fit(norm)
        self.keyword_.fit(norm)
        self.structural_.fit(norm)
        logger.info(
            "Transformer fitted. tfidf_vocab=%d keyword_groups=%d",
            len(self.tfidf_.vocabulary_), len(self.config.keyword_groups),
        )
        return self

    def transform(self, X: List[str]) -> sp.csr_matrix:
        norm = self.normalizer_.transform(X)
        return sp.hstack([
            self.tfidf_.transform(norm),
            sp.csr_matrix(self.keyword_.transform(norm)),
            sp.csr_matrix(self.structural_.transform(norm)),
        ], format="csr")

    def fit_transform(self, X: List[str], y=None) -> sp.csr_matrix:
        return self.fit(X).transform(X)

    def get_feature_count(self) -> int:
        return len(self.tfidf_.vocabulary_) + len(self.config.keyword_groups) + 8

    def get_feature_names(self) -> List[str]:
        return (
            self.tfidf_.get_feature_names_out().tolist()
            + self.keyword_.get_feature_names_out()
            + self.structural_.get_feature_names_out()
        )
