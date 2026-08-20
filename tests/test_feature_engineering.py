"""Tests for feature engineering pipeline."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import Config
from src.feature_engineering import (
    KeywordFeatureExtractor, PromptFeatureTransformer,
    StructuralFeatureExtractor, normalize_prompt,
)

cfg = Config()

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


class TestNormalizePrompt:
    def test_lowercase(self):
        assert normalize_prompt("FIND ALL FUNCTIONS") == "find all functions"

    def test_strips_whitespace(self):
        assert normalize_prompt("  find function  ") == "find function"

    def test_collapses_spaces(self):
        result = normalize_prompt("find   all   functions")
        assert result == "find all functions"

    def test_none_input(self):
        assert isinstance(normalize_prompt(None), str)

    def test_empty_string(self):
        assert normalize_prompt("") == ""

    def test_preserves_underscores(self):
        assert "authenticate_user" in normalize_prompt("Find authenticate_user()")


class TestKeywordFeatureExtractor:
    def setup_method(self):
        self.ext = KeywordFeatureExtractor(cfg.keyword_groups)
        self.ext.fit(TRAIN_PROMPTS)

    def test_output_shape(self):
        r = self.ext.transform(["find all functions", "read the file"])
        assert r.shape == (2, len(cfg.keyword_groups))

    def test_grep_search_detected(self):
        r = self.ext.transform(["find all occurrences of this pattern"])
        idx = list(cfg.keyword_groups.keys()).index("grep_search")
        assert r[0, idx] == 1.0

    def test_binary_output(self):
        r = self.ext.transform(["find all functions"])
        assert set(r.flatten().tolist()).issubset({0.0, 1.0})

    def test_feature_names_length(self):
        assert len(self.ext.get_feature_names_out()) == len(cfg.keyword_groups)


class TestStructuralFeatureExtractor:
    def setup_method(self):
        self.ext = StructuralFeatureExtractor()
        self.ext.fit([])

    def test_output_shape(self):
        r = self.ext.transform(["find authenticate_user()", "list files"])
        assert r.shape == (2, 8)

    def test_parentheses_detected(self):
        r = self.ext.transform(["find authenticate_user()"])
        assert r[0, 2] == 1.0

    def test_underscore_detected(self):
        r = self.ext.transform(["find authenticate_user"])
        assert r[0, 4] == 1.0

    def test_question_mark_detected(self):
        r = self.ext.transform(["how do I do this?"])
        assert r[0, 5] == 1.0

    def test_empty_prompt_no_crash(self):
        r = self.ext.transform([""])
        assert r.shape == (1, 8)

    def test_feature_count(self):
        assert len(self.ext.get_feature_names_out()) == 8


class TestPromptFeatureTransformer:
    def setup_method(self):
        self.t = PromptFeatureTransformer(cfg)
        self.t.fit(TRAIN_PROMPTS)

    def test_returns_sparse(self):
        assert sp.issparse(self.t.fit_transform(TRAIN_PROMPTS))

    def test_shape_consistent(self):
        X_train = self.t.fit_transform(TRAIN_PROMPTS)
        X_new = self.t.transform(["find something"])
        assert X_new.shape[1] == X_train.shape[1]

    def test_no_nan(self):
        X = self.t.transform(["find all functions"])
        assert not np.any(np.isnan(X.toarray()))

    def test_no_inf(self):
        X = self.t.transform(["find all functions"])
        assert not np.any(np.isinf(X.toarray()))

    def test_empty_prompt_no_crash(self):
        assert self.t.transform([""]).shape[0] == 1

    def test_feature_count_positive(self):
        assert self.t.get_feature_count() > 0

    def test_names_match_count(self):
        assert len(self.t.get_feature_names()) == self.t.get_feature_count()

    def test_different_prompts_different_vectors(self):
        X = self.t.transform([
            "find all functions that call authenticate_user()",
            "search the web for python documentation",
        ])
        assert not np.allclose(X.getrow(0).toarray(), X.getrow(1).toarray())
