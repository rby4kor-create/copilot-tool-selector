"""
src/routing/router.py
Tool routing layer. Completely separate from ML logic.

ML answers:  Which tools are relevant?
Router answers: Which tools should actually execute?
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Set

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
            "model_version": self.ml_result.get("model_version"),
            "algorithm": self.ml_result.get("algorithm"),
            "fallback_used": self.ml_result.get("fallback_used", False),
        }

    def summary(self) -> str:
        lines = [
            f"EXECUTE ({len(self.execute)}): {self.execute}",
            f"SKIP    ({len(self.skip)}):    {self.skip}",
        ]
        if self.policy_overrides:
            lines.append(f"POLICY OVERRIDES: {self.policy_overrides}")
        total = len(self.execute) + len(self.skip)
        reduction = 1.0 - len(self.execute) / total if total > 0 else 0
        lines.append(f"TOOL REDUCTION: {reduction:.1%}")
        return "\n".join(lines)


class ToolRouter:
    """
    Applies business routing policies on top of ML predictions.

    Policies (all optional):
        mandatory_tools    — always execute regardless of ML score
        banned_tools       — never execute regardless of ML score
        tool_dependencies  — if A selected, also select B
    """

    def __init__(
        self,
        all_tools: List[str],
        mandatory_tools: List[str] = None,
        banned_tools: List[str] = None,
        tool_dependencies: Dict[str, List[str]] = None,
        fallback_strategy: str = "select_all",
    ):
        self.all_tools = all_tools
        self.mandatory_tools: Set[str] = set(mandatory_tools or [])
        self.banned_tools: Set[str] = set(banned_tools or [])
        self.tool_dependencies: Dict[str, List[str]] = tool_dependencies or {}
        self.fallback_strategy = fallback_strategy

    def route(self, ml_result: Dict[str, Any]) -> RoutingDecision:
        overrides: List[str] = []
        selected: Set[str] = {t["tool"] for t in ml_result.get("selected_tools", [])}
        all_known: Set[str] = set(self.all_tools)

        # Apply dependencies
        for tool, deps in self.tool_dependencies.items():
            if tool in selected:
                for dep in deps:
                    if dep not in selected and dep in all_known:
                        selected.add(dep)
                        overrides.append(f"DEPENDENCY: {dep} added because {tool} selected")

        # Apply mandatory
        for tool in self.mandatory_tools:
            if tool in all_known and tool not in selected:
                selected.add(tool)
                overrides.append(f"MANDATORY: {tool} force-selected")

        # Apply banned
        for tool in self.banned_tools:
            if tool in selected:
                selected.discard(tool)
                overrides.append(f"BANNED: {tool} force-rejected")

        # Minimum guarantee
        if not selected and self.fallback_strategy == "select_all":
            selected = all_known.copy()
            overrides.append("MINIMUM: no tools selected — selected all as fallback")

        execute = [t for t in self.all_tools if t in selected]
        skip = [t for t in self.all_tools if t not in selected]

        if overrides:
            logger.info("[%s] Policy overrides: %s",
                        ml_result.get("request_id", "?"), overrides)

        decision = RoutingDecision(
            execute=execute, skip=skip,
            policy_overrides=overrides, ml_result=ml_result,
        )

        total = len(execute) + len(skip)
        reduction = 1.0 - len(execute) / total if total > 0 else 0
        logger.info(
            "[%s] EXECUTE=%s SKIP=%s REDUCTION=%.1f%%",
            ml_result.get("request_id", "?"), execute, skip, reduction * 100,
        )
        return decision
