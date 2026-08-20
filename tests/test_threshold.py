"""Tests for threshold logic."""
from __future__ import annotations
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import Config

cfg = Config()


class TestThresholdConfig:
    def test_default_threshold_exists(self):
        assert cfg.default_threshold is not None

    def test_default_threshold_valid(self):
        assert 0.0 < cfg.default_threshold < 1.0

    def test_per_tool_thresholds_exist(self):
        assert len(cfg.per_tool_thresholds) > 0

    def test_per_tool_thresholds_valid(self):
        for tool, t in cfg.per_tool_thresholds.items():
            assert 0.0 < t < 1.0, f"Bad threshold for {tool}: {t}"

    def test_unknown_tool_returns_default(self):
        assert cfg.per_tool_threshold("nonexistent_xyz") == cfg.default_threshold


class TestThresholdLogic:
    @pytest.mark.parametrize("score,threshold,expected", [
        (0.91, 0.50, "SELECT"),
        (0.84, 0.50, "SELECT"),
        (0.50, 0.50, "SELECT"),
        (0.499, 0.50, "DESELECT"),
        (0.13, 0.50, "DESELECT"),
        (0.70, 0.70, "SELECT"),
        (0.699, 0.70, "DESELECT"),
        (1.0, 0.50, "SELECT"),
        (0.0, 0.50, "DESELECT"),
    ])
    def test_decision(self, score, threshold, expected):
        result = "SELECT" if score >= threshold else "DESELECT"
        assert result == expected

    def test_all_tools_have_threshold(self):
        for tool in cfg.all_tools():
            t = cfg.per_tool_threshold(tool)
            assert isinstance(t, float)
            assert 0.0 < t <= 1.0
