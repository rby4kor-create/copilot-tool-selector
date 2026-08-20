"""Tests for production inference."""
import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.predict import ModelNotFoundError, ToolSelectorModel, select_tools

MODEL_PATH = Path("models/tool_selector_pipeline.joblib")
META_PATH = Path("models/metadata.json")
MODEL_TRAINED = MODEL_PATH.exists()

SKIP = pytest.mark.skipif(not MODEL_TRAINED, reason="Run train.py first")


class TestModelNotFound:
    def test_raises(self, tmp_path):
        with pytest.raises(ModelNotFoundError):
            ToolSelectorModel(tmp_path / "none.joblib", tmp_path / "none.json")


@SKIP
class TestPredictions:
    def setup_method(self):
        self.model = ToolSelectorModel(MODEL_PATH, META_PATH)

    def test_returns_result(self):
        r = self.model.predict("Find all functions calling authenticate_user()")
        assert r is not None

    def test_all_tools_accounted(self):
        r = self.model.predict("Find all functions calling authenticate_user()")
        returned = {t["tool"] for t in r.selected_tools} | {t["tool"] for t in r.rejected_tools}
        assert returned == set(self.model.tools)

    def test_scores_valid(self):
        r = self.model.predict("Find all functions calling authenticate_user()")
        for t in r.selected_tools + r.rejected_tools:
            assert -1.0 <= t["score"] <= 1.0

    def test_select_tools_dict(self):
        r = select_tools("find something")
        assert isinstance(r, dict)
        assert "selected_tools" in r

    def test_empty_fallback(self):
        r = self.model.predict("")
        assert r.fallback_used is True

    def test_none_fallback(self):
        r = self.model.predict(None)
        assert r.fallback_used is True

    def test_no_crash_unseen(self):
        assert self.model.predict("xyzzy frob blorb") is not None


@SKIP
class TestThreshold:
    def setup_method(self):
        self.model = ToolSelectorModel(MODEL_PATH, META_PATH)

    def test_select_above_threshold(self):
        import joblib
        art = joblib.load(MODEL_PATH)
        thresholds = art["thresholds"]
        r = self.model.predict("Find all functions calling authenticate_user()")
        for t in r.selected_tools:
            thresh = thresholds.get(t["tool"], 0.5)
            assert t["score"] >= thresh

    def test_reject_below_threshold(self):
        import joblib
        art = joblib.load(MODEL_PATH)
        thresholds = art["thresholds"]
        r = self.model.predict("Find all functions calling authenticate_user()")
        for t in r.rejected_tools:
            thresh = thresholds.get(t["tool"], 0.5)
            assert t["score"] < thresh
