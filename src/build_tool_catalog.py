"""
src/build_tool_catalog.py
Builds tool catalog from raw copilot_tools.json.
Falls back to create_tool_catalog if raw file missing.
"""
from __future__ import annotations
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_catalog() -> dict:
    config = Config()
    raw = config.raw_data_dir / "copilot_tools.json"
    if not raw.exists():
        logger.warning("copilot_tools.json not found - using create_tool_catalog")
        from src.create_tool_catalog import create_catalog
        return create_catalog()
    with open(raw) as f:
        data = json.load(f)
    catalog = {}
    for tool in data.get("tools", []):
        name = tool.get("name", "").strip()
        if name:
            catalog[name] = {
                "description": tool.get("description", ""),
                "category": tool.get("category", "unknown"),
            }
    catalog["__meta__"] = {
        "version": config.tool_catalog_version,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    out = config.tool_catalog_path
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(catalog, f, indent=2)
    logger.info("Catalog built: %d tools -> %s", len(catalog) - 1, out)
    return catalog


if __name__ == "__main__":
    build_catalog()
