"""
src/catalog/tool_catalog.py
Dynamic tool registry. Never hardcode tool names in business logic.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ToolCatalog:
    """
    Dynamic tool registry loaded from disk.
    All tool names come from here — never hardcoded.
    """

    def __init__(self, catalog_path: str | Path) -> None:
        self.catalog_path = Path(catalog_path)
        self._tools: Dict[str, Dict] = {}
        self._version: str = "unknown"
        self._load()

    def _load(self) -> None:
        if not self.catalog_path.exists():
            raise FileNotFoundError(f"Tool catalog not found: {self.catalog_path}")
        with open(self.catalog_path) as f:
            raw = json.load(f)
        meta = raw.pop("__meta__", {})
        self._version = meta.get("version", "unknown")
        self._tools = {k: v for k, v in raw.items() if not k.startswith("__")}
        logger.info("Loaded tool catalog v%s with %d tools", self._version, len(self._tools))

    @property
    def tools(self) -> List[str]:
        return list(self._tools.keys())

    @property
    def version(self) -> str:
        return self._version

    def get_description(self, tool: str) -> str:
        return self._tools.get(tool, {}).get("description", "")

    def get_category(self, tool: str) -> str:
        return self._tools.get(tool, {}).get("category", "unknown")

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, tool: str) -> bool:
        return tool in self._tools
