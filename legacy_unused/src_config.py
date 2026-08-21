"""
src/config.py
Single source of truth for all configuration.
All other modules import from here.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List
import yaml

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
CONFIG_PATH: Path = REPO_ROOT / "config" / "config.yaml"


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class Config:
    def __init__(self, config_path: Path = CONFIG_PATH) -> None:
        self._raw: Dict[str, Any] = _load_yaml(config_path)

    @property
    def repo_root(self) -> Path:
        return REPO_ROOT

    @property
    def raw_data_dir(self) -> Path:
        return REPO_ROOT / self._raw["data"]["raw_dir"]

    @property
    def processed_data_dir(self) -> Path:
        return REPO_ROOT / self._raw["data"]["processed_dir"]

    @property
    def synthetic_data_path(self) -> Path:
        return REPO_ROOT / self._raw["data"]["synthetic_data_path"]

    @property
    def manager_data_path(self) -> Path:
        return REPO_ROOT / self._raw["data"]["manager_data_path"]

    @property
    def tool_catalog_path(self) -> Path:
        return REPO_ROOT / self._raw["data"]["tool_catalog_path"]

    @property
    def tool_catalog_version(self) -> str:
        return self._raw["data"]["tool_catalog_version"]

    @property
    def training_source(self) -> str:
        return self._raw["data"]["training_source"]

    @property
    def model_output_dir(self) -> Path:
        return REPO_ROOT / self._raw["model"]["output_dir"]

    @property
    def model_artifact_path(self) -> Path:
        return self.model_output_dir / self._raw["model"]["artifact_name"]

    @property
    def model_metadata_path(self) -> Path:
        return self.model_output_dir / self._raw["model"]["metadata_name"]

    @property
    def model_version(self) -> str:
        return self._raw["model"]["version"]

    @property
    def tfidf_config(self) -> Dict[str, Any]:
        return self._raw["features"]["tfidf"]

    @property
    def keyword_groups(self) -> Dict[str, List[str]]:
        return self._raw["features"]["keyword_groups"]

    @property
    def test_size(self) -> float:
        return float(self._raw["split"]["test_size"])

    @property
    def validation_size(self) -> float:
        return float(self._raw["split"]["validation_size"])

    @property
    def split_random_state(self) -> int:
        return int(self._raw["split"]["random_state"])

    @property
    def default_threshold(self) -> float:
        return float(self._raw["thresholds"]["default"])

    @property
    def per_tool_thresholds(self) -> Dict[str, float]:
        return {k: float(v) for k, v in self._raw["thresholds"]["per_tool"].items()}

    def per_tool_threshold(self, tool_name: str) -> float:
        return self.per_tool_thresholds.get(tool_name, self.default_threshold)

    @property
    def fallback_strategy(self) -> str:
        return self._raw["fallback"]["strategy"]

    @property
    def fallback_default_tools(self) -> List[str]:
        return self._raw["fallback"]["default_tools"]

    @property
    def lr_config(self) -> Dict[str, Any]:
        return self._raw["model"]["logistic_regression"]

    @property
    def log_level(self) -> str:
        return self._raw["logging"]["level"]

    @property
    def log_prompt_content(self) -> bool:
        return bool(self._raw["logging"]["log_prompt_content"])

    def all_tools(self) -> List[str]:
        return list(self._raw["features"]["keyword_groups"].keys())

    def ensure_dirs(self) -> None:
        self.model_output_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)


cfg = Config()
