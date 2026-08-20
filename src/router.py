"""
src/router.py
Tool routing layer. Separate from ML logic.
ML decides relevance. Router applies business policies.
"""
from __future__ import annotations
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import Config

logger = logging.getLogger(__name__)


class RoutingDecision:
    def __init__(self, execute, skip, policy_overrides, ml_result):
        self.execute = execute
        self.skip = skip
        self.policy_overrides = policy_overrides
        self.ml_result = ml_result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execute": self.execute,
            "skip": self.skip,
            "policy_overrides": self.policy_overrides,
            "ml_selected": [t["tool"] for t in self.ml_result.get("selected_tools", [])],
            "request_id": self.ml_result.get("request_id"),
            "fallback_used": self.ml_result.get("fallback_used", False),
        }

    def __repr__(self):
        return f"RoutingDecision(execute={self.execute}, skip={self.skip})"


class ToolRouter:
    """Applies post-ML routing policies on top of ML scores."""

    def __init__(self, config=None, mandatory_tools=None,
                 banned_tools=None, tool_dependencies=None):
        self.config = config or Config()
        self.mandatory_tools: Set[str] = set(mandatory_tools or [])
        self.banned_tools: Set[str] = set(banned_tools or [])
        self.tool_dependencies: Dict[str, List[str]] = tool_dependencies or {}

    def route(self, ml_result: Dict[str, Any]) -> RoutingDecision:
        overrides: List[str] = []
        selected: Set[str] = {t["tool"] for t in ml_result.get("selected_tools", [])}
        all_tools: Set[str] = selected | {t["tool"] for t in ml_result.get("rejected_tools", [])}

        for tool, deps in self.tool_dependencies.items():
            if tool in selected:
                for dep in deps:
                    if dep not in selected and dep in all_tools:
                        selected.add(dep)
                        overrides.append(f"DEPENDENCY: {dep} added because {tool} selected")

        for tool in self.mandatory_tools:
            if tool in all_tools and tool not in selected:
                selected.add(tool)
                overrides.append(f"MANDATORY: {tool} force-selected")

        for tool in self.banned_tools:
            if tool in selected:
                selected.discard(tool)
                overrides.append(f"BANNED: {tool} force-rejected")

        if not selected:
            if self.config.fallback_strategy == "select_all":
                selected = all_tools.copy()
                overrides.append("MINIMUM: selected all tools")
            elif self.config.fallback_strategy == "select_default_tools":
                selected = set(self.config.fallback_default_tools) & all_tools
                overrides.append(f"MINIMUM: selected defaults={selected}")

        known = self.config.all_tools()
        execute = [t for t in known if t in selected]
        skip = [t for t in known if t not in selected]

        if overrides:
            logger.info("[%s] Overrides: %s", ml_result.get("request_id", "?"), overrides)
        logger.info("[%s] EXECUTE=%s SKIP=%s", ml_result.get("request_id", "?"), execute, skip)
        return RoutingDecision(execute=execute, skip=skip,
                               policy_overrides=overrides, ml_result=ml_result)
