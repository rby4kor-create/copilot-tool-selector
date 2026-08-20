"""Tests for tool router."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

from src.routing.router import ToolRouter, RoutingDecision

ALL_TOOLS = list(cfg["features"]["keyword_groups"].keys())
OTHER = [t for t in ALL_TOOLS if t != "grep_search"]


def ml_result(selected, rejected):
    return {
        "request_id": "test-001",
        "model_version": "2.0.0",
        "algorithm": "logistic_regression",
        "selected_tools": [{"tool": t, "score": 0.9} for t in selected],
        "rejected_tools": [{"tool": t, "score": 0.1} for t in rejected],
        "fallback_used": False,
    }


class TestBasic:
    def test_returns_decision(self):
        r = ToolRouter(ALL_TOOLS).route(ml_result(["grep_search"], OTHER))
        assert isinstance(r, RoutingDecision)

    def test_selected_in_execute(self):
        r = ToolRouter(ALL_TOOLS).route(ml_result(["grep_search"], OTHER))
        assert "grep_search" in r.execute

    def test_rejected_in_skip(self):
        r = ToolRouter(ALL_TOOLS).route(ml_result(["grep_search"], OTHER))
        assert "codebase_search" in r.skip

    def test_all_accounted(self):
        r = ToolRouter(ALL_TOOLS).route(ml_result(["grep_search"], OTHER))
        assert set(r.execute) | set(r.skip) == set(ALL_TOOLS)


class TestPolicies:
    def test_mandatory(self):
        router = ToolRouter(ALL_TOOLS, mandatory_tools=["grep_search"])
        r = router.route(ml_result([], ALL_TOOLS))
        assert "grep_search" in r.execute

    def test_banned(self):
        router = ToolRouter(ALL_TOOLS, banned_tools=["web_search"])
        r = router.route(ml_result(ALL_TOOLS, []))
        assert "web_search" not in r.execute

    def test_dependency(self):
        router = ToolRouter(ALL_TOOLS, tool_dependencies={"grep_search": ["codebase_search"]})
        non_gs = [t for t in ALL_TOOLS if t != "grep_search"]
        r = router.route(ml_result(["grep_search"], non_gs))
        assert "codebase_search" in r.execute
