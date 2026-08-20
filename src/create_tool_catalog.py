"""
src/create_tool_catalog.py
Creates the tool catalog. Keep in sync with config.yaml keyword_groups.
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

TOOL_DEFINITIONS = {
    "grep_search": {
        "description": "Exact text pattern or regex search across files",
        "category": "search",
        "use_when": "Finding specific strings or patterns in code files",
    },
    "codebase_search": {
        "description": "Semantic search across the entire codebase",
        "category": "search",
        "use_when": "Finding code definitions, symbols, or semantic matches",
    },
    "read_file": {
        "description": "Read and display the contents of a specific file",
        "category": "file_ops",
        "use_when": "User wants to see the contents of a known file",
    },
    "list_dir": {
        "description": "List files and directories in a path",
        "category": "file_ops",
        "use_when": "User wants to see what files or directories exist",
    },
    "run_terminal_cmd": {
        "description": "Execute a terminal or shell command",
        "category": "execution",
        "use_when": "User wants to run a command, script, build, or test",
    },
    "edit_file": {
        "description": "Create, edit, or modify a file",
        "category": "file_ops",
        "use_when": "User wants to create or modify code or files",
    },
    "web_search": {
        "description": "Search the web for documentation or external information",
        "category": "search",
        "use_when": "User wants information from external web sources",
    },
}


def create_catalog() -> dict:
    config = Config()
    catalog = dict(TOOL_DEFINITIONS)
    catalog["__meta__"] = {
        "version": config.tool_catalog_version,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    out = config.tool_catalog_path
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(catalog, f, indent=2)
    logger.info("Catalog created: %d tools -> %s", len(TOOL_DEFINITIONS), out)
    return catalog


if __name__ == "__main__":
    create_catalog()
