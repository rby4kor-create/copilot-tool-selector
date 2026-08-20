"""Tests for tool router."""
from __future__ import annotations
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.router import RoutingDecision, ToolRouter
from src.config import Config

cfg = Config()
ALL_TOOLS = cfg.all_tools()


def make_result(selected, rejected):
    return {
        "request_id": "test-001",
        "model_version": "1.0.0",
        "selected_tools": [{"tool": t, "score": 0.9} for t in selected],
        "rejected_tools": [{"tool": t, "score": 0.1} for t in rejected],
        "fallback_used": False,
    }


OTHER = [t for t in ALL_TOOLS if t != "grep_search"]


class TestBasic:
    def test_returns_routing_decision(self):
        r = ToolRouter().route(make_result(["grep_search"], OTHER))
        assert isinstance(r, RoutingDecision)

    def test_selected_in_execute(self):
        non_gs_cs = [t for t in ALL_TOOLS if t not in ["grep_search", "codebase_search"]]
        r = ToolRouter().route(make_result(["grep_search", "codebase_search"], non_gs_cs))
        assert "grep_search" in r.execute
        assert "codebase_search" in r.execute

    def test_rejected_in_skip(self):
        r = ToolRouter().route(make_result(["grep_search"], OTHER))
        assert "codebase_search" in r.skip

    def test_all_tools_accounted(self):
        r = ToolRouter().route(make_result(["grep_search"], OTHER))
        assert set(r.execute) | set(r.skip) == set(ALL_TOOLS)

    def test_to_dict_has_keys(self):
        r = ToolRouter().route(make_result(["grep_search"], OTHER))
        d = r.to_dict()
        assert "execute" in d
        assert "skip" in d


class TestMandatory:
    def test_mandatory_always_executed(self):
        router = ToolRouter(mandatory_tools=["grep_search"])
        r = router.route(make_result([], ALL_TOOLS))
        assert "grep_search" in r.execute

    def test_mandatory_override_logged(self):
        router = ToolRouter(mandatory_tools=["grep_search"])
        r = router.route(make_result([], ALL_TOOLS))
        assert any("MANDATORY" in o for o in r.policy_overrides)


class TestBanned:
    def test_banned_never_executed(self):
        router = ToolRouter(banned_tools=["web_search"])
        r = router.route(make_result(ALL_TOOLS, []))
        assert "web_search" not in r.execute
        assert "web_search" in r.skip

    def test_banned_override_logged(self):
        router = ToolRouter(banned_tools=["web_search"])
        r = router.route(make_result(ALL_TOOLS, []))
        assert any("BANNED" in o for o in r.policy_overrides)


class TestDependencies:
    def test_dependency_added(self):
        router = ToolRouter(tool_dependencies={"grep_search": ["codebase_search"]})
        non_gs = [t for t in ALL_TOOLS if t != "grep_search"]
        r = router.route(make_result(["grep_search"], non_gs))
        assert "codebase_search" in r.execute

    def test_dependency_not_added_when_parent_absent(self):
        router = ToolRouter(tool_dependencies={"grep_search": ["codebase_search"]})
        non_rf = [t for t in ALL_TOOLS if t != "read_file"]
        r = router.route(make_result(["read_file"], non_rf))
        assert "grep_search" not in r.execute
