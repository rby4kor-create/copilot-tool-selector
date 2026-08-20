"""Tests for production inference."""
from __future__ import annotations
import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.predict import ModelNotFoundError, ToolSelectorModel, select_tools
from src.config import Config

cfg = Config()


def model_is_trained():
    return cfg.model_artifact_path.exists()


SKIP = pytest.mark.skipif(
    not model_is_trained(),
    reason="No trained model - run: python src/train.py",
)


class TestModelNotFound:
    def test_raises_when_model_missing(self, tmp_path):
        fake = MagicMock()
        fake.model_artifact_path = tmp_path / "none.joblib"
        fake.model_metadata_path = tmp_path / "none.json"
        with pytest.raises(ModelNotFoundError):
            ToolSelectorModel(config=fake)


@SKIP
class TestValidPredictions:
    def setup_method(self):
        self.model = ToolSelectorModel()

    def test_predict_not_none(self):
        assert self.model.predict("Find all functions calling authenticate_user()") is not None

    def test_has_selected_tools(self):
        r = self.model.predict("Find all functions calling authenticate_user()")
        assert isinstance(r.selected_tools, list)

    def test_has_rejected_tools(self):
        r = self.model.predict("Find all functions calling authenticate_user()")
        assert isinstance(r.rejected_tools, list)

    def test_all_tools_accounted(self):
        r = self.model.predict("Find all functions calling authenticate_user()")
        returned = (
            {t["tool"] for t in r.selected_tools}
            | {t["tool"] for t in r.rejected_tools}
        )
        assert returned == set(self.model.tools)

    def test_scores_valid(self):
        r = self.model.predict("Find all functions calling authenticate_user()")
        for t in r.selected_tools + r.rejected_tools:
            assert -1.0 <= t["score"] <= 1.0

    def test_select_tools_returns_dict(self):
        r = select_tools("Find all functions calling authenticate_user()")
        assert isinstance(r, dict)
        assert "selected_tools" in r
        assert "rejected_tools" in r

    def test_model_not_reloaded(self):
        m = ToolSelectorModel()
        v1 = m.predict("find functions").model_version
        v2 = m.predict("list all files").model_version
        assert v1 == v2


@SKIP
class TestEdgeCases:
    def setup_method(self):
        self.model = ToolSelectorModel()

    def test_empty_prompt_fallback(self):
        assert self.model.predict("").fallback_used is True

    def test_none_prompt_fallback(self):
        assert self.model.predict(None).fallback_used is True

    def test_whitespace_prompt_fallback(self):
        assert self.model.predict("   ").fallback_used is True

    def test_single_word_no_crash(self):
        assert self.model.predict("find") is not None

    def test_special_chars_no_crash(self):
        assert self.model.predict("!@#$%^&*()") is not None

    def test_unseen_prompt_no_crash(self):
        assert self.model.predict("xyzzy frob blorb quux") is not None
