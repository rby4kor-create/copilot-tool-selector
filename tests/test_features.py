"""Tests for feature engineering."""
import sys
from pathlib import Path
import numpy as np
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

from src.features.feature_engineering import (
    PromptFeatureTransformer, KeywordFeatureExtractor,
    StructuralFeatureExtractor, normalize_prompt,
)

TRAIN_PROMPTS = [
    "find all functions that call authenticate_user()",
    "read the main configuration file",
    "list all files in the src directory",
    "run the test suite",
    "edit the config file",
    "search the web for documentation",
    "find code similar to this implementation",
    "show me the contents of README.md",
    "create a new Python file called helper.py",
    "find all classes that extend BaseModel",
]


class TestNormalize:
    def test_lowercase(self):
        assert normalize_prompt("FIND") == "find"

    def test_strips(self):
        assert normalize_prompt("  find  ") == "find"

    def test_none(self):
        assert isinstance(normalize_prompt(None), str)

    def test_empty(self):
        assert normalize_prompt("") == ""


class TestKeywordExtractor:
    def setup_method(self):
        self.ext = KeywordFeatureExtractor(cfg["features"]["keyword_groups"])
        self.ext.fit(TRAIN_PROMPTS)

    def test_shape(self):
        r = self.ext.transform(["find all", "read file"])
        assert r.shape == (2, len(cfg["features"]["keyword_groups"]))

    def test_binary(self):
        r = self.ext.transform(["find something"])
        assert set(r.flatten()).issubset({0.0, 1.0})

    def test_grep_detected(self):
        r = self.ext.transform(["find all occurrences"])
        idx = list(cfg["features"]["keyword_groups"].keys()).index("grep_search")
        assert r[0, idx] == 1.0


class TestStructuralExtractor:
    def setup_method(self):
        self.ext = StructuralFeatureExtractor()
        self.ext.fit([])

    def test_shape(self):
        r = self.ext.transform(["find authenticate_user()", "list files"])
        assert r.shape == (2, 8)

    def test_parens(self):
        r = self.ext.transform(["find func()"])
        assert r[0, 2] == 1.0

    def test_underscore(self):
        r = self.ext.transform(["find my_func"])
        assert r[0, 4] == 1.0

    def test_empty(self):
        r = self.ext.transform([""])
        assert r.shape == (1, 8)


class TestTransformer:
    def setup_method(self):
        self.t = PromptFeatureTransformer(
            cfg["features"]["tfidf"],
            cfg["features"]["keyword_groups"],
        )
        self.t.fit(TRAIN_PROMPTS)

    def test_sparse(self):
        assert sp.issparse(self.t.fit_transform(TRAIN_PROMPTS))

    def test_consistent_shape(self):
        X1 = self.t.fit_transform(TRAIN_PROMPTS)
        X2 = self.t.transform(["find something"])
        assert X1.shape[1] == X2.shape[1]

    def test_no_nan(self):
        X = self.t.transform(["find all functions"])
        assert not np.any(np.isnan(X.toarray()))

    def test_empty_no_crash(self):
        assert self.t.transform([""]).shape[0] == 1
