"""
src/inference/predict.py
Production inference. Model loaded ONCE. Never reloaded per prediction.

Usage:
    from src.inference.predict import select_tools
    result = select_tools("Find all functions calling authenticate_user()")
"""
from __future__ import annotations
import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib

logger = logging.getLogger(__name__)


class ModelNotFoundError(RuntimeError):
    pass


class ModelLoadError(RuntimeError):
    pass


class ToolSelectionResult:
    def __init__(self, request_id, prompt, selected_tools, rejected_tools,
                 thresholds_used, model_version, algorithm,
                 fallback_used=False, latency_ms=0.0):
        self.request_id = request_id
        self.prompt = prompt
        self.selected_tools = selected_tools
        self.rejected_tools = rejected_tools
        self.thresholds_used = thresholds_used
        self.model_version = model_version
        self.algorithm = algorithm
        self.fallback_used = fallback_used
        self.latency_ms = latency_ms

    def to_dict(self, include_prompt=True) -> Dict[str, Any]:
        d = {
            "request_id": self.request_id,
            "model_version": self.model_version,
            "algorithm": self.algorithm,
            "selected_tools": self.selected_tools,
            "rejected_tools": self.rejected_tools,
            "thresholds_used": self.thresholds_used,
            "fallback_used": self.fallback_used,
            "latency_ms": round(self.latency_ms, 2),
        }
        if include_prompt:
            d["prompt"] = self.prompt
        return d

    def explain(self) -> str:
        """Human-readable explanation of the routing decision."""
        lines = [
            f"Prompt      : {self.prompt}",
            f"Model       : v{self.model_version} ({self.algorithm})",
            f"Latency     : {self.latency_ms:.1f}ms",
            f"Fallback    : {self.fallback_used}",
            "",
            "SELECTED TOOLS:",
        ]
        for t in self.selected_tools:
            thresh = self.thresholds_used.get(t["tool"], 0.5)
            lines.append(f"  [SELECT]   {t['tool']:<25} score={t['score']:.4f}  threshold={thresh:.2f}")
        lines.append("")
        lines.append("REJECTED TOOLS:")
        for t in self.rejected_tools:
            thresh = self.thresholds_used.get(t["tool"], 0.5)
            lines.append(f"  [DESELECT] {t['tool']:<25} score={t['score']:.4f}  threshold={thresh:.2f}")
        return "\n".join(lines)


class ToolSelectorModel:
    """Production inference model. Loads once. Never reloads."""

    def __init__(self, model_path: str | Path, metadata_path: str | Path):
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self._transformer = None
        self._selector = None
        self._thresholds = {}
        self._tools = []
        self._model_version = "unknown"
        self._algorithm = "unknown"
        self._loaded = False
        self._load()

    def _load(self):
        if not self.model_path.exists():
            raise ModelNotFoundError(
                f"Model not found: {self.model_path}\n"
                "Run: python src/training/train.py"
            )
        try:
            # Import model classes so joblib can deserialize
            from src.models.model_classes import MultiLabelToolSelector
            from src.features.feature_engineering import PromptFeatureTransformer

            art = joblib.load(self.model_path)
            self._transformer = art["transformer"]
            self._selector = art["selector"]
            self._thresholds = art["thresholds"]
            self._tools = art["tools"]
            self._loaded = True

            if self.metadata_path.exists():
                with open(self.metadata_path) as f:
                    meta = json.load(f)
                self._model_version = meta.get("model_version", "unknown")
                self._algorithm = meta.get("algorithm", "unknown")

            logger.info(
                "Model loaded v%s algorithm=%s tools=%d",
                self._model_version, self._algorithm, len(self._tools),
            )
        except ModelNotFoundError:
            raise
        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {e}") from e

    def predict(self, prompt: str, request_id: str = None) -> ToolSelectionResult:
        t0 = time.perf_counter()
        rid = request_id or str(uuid.uuid4())[:12]

        if not prompt or not isinstance(prompt, str) or not prompt.strip():
            logger.warning("[%s] Empty/invalid prompt — fallback", rid)
            return self._fallback(rid, prompt or "", "empty_prompt",
                                  (time.perf_counter() - t0) * 1000)
        prompt = prompt.strip()
        try:
            X = self._transformer.transform([prompt])
            probas = self._selector.predict_proba(X)
            selected, rejected = [], []
            for tool in self._tools:
                score = float(probas[tool][0])
                thresh = self._thresholds.get(tool, 0.50)
                entry = {"tool": tool, "score": round(score, 4)}
                (selected if score >= thresh else rejected).append(entry)
            selected.sort(key=lambda x: x["score"], reverse=True)
            rejected.sort(key=lambda x: x["score"], reverse=True)
            lat = (time.perf_counter() - t0) * 1000
            logger.info(
                "[%s] selected=%s lat=%.1fms",
                rid, [t["tool"] for t in selected], lat,
            )
            return ToolSelectionResult(
                request_id=rid, prompt=prompt,
                selected_tools=selected, rejected_tools=rejected,
                thresholds_used=self._thresholds.copy(),
                model_version=self._model_version,
                algorithm=self._algorithm,
                fallback_used=False, latency_ms=lat,
            )
        except Exception as e:
            logger.error("[%s] Prediction failed: %s", rid, e, exc_info=True)
            return self._fallback(rid, prompt, str(e),
                                  (time.perf_counter() - t0) * 1000)

    def _fallback(self, rid, prompt, reason, latency_ms) -> ToolSelectionResult:
        logger.warning("[%s] Fallback activated. reason=%s", rid, reason)
        selected = [{"tool": t, "score": -1.0} for t in self._tools]
        return ToolSelectionResult(
            request_id=rid, prompt=prompt,
            selected_tools=selected, rejected_tools=[],
            thresholds_used=self._thresholds,
            model_version=self._model_version,
            algorithm=self._algorithm,
            fallback_used=True, latency_ms=latency_ms,
        )

    @property
    def tools(self):
        return self._tools.copy()

    @property
    def is_loaded(self):
        return self._loaded


# Module-level singleton
_singleton: Optional[ToolSelectorModel] = None
_MODEL_PATH = Path("models/tool_selector_pipeline.joblib")
_META_PATH = Path("models/metadata.json")


def _get_model() -> ToolSelectorModel:
    global _singleton
    if _singleton is None:
        _singleton = ToolSelectorModel(_MODEL_PATH, _META_PATH)
    return _singleton


def select_tools(prompt: str, request_id: str = None) -> Dict[str, Any]:
    """Primary production entry point."""
    return _get_model().predict(prompt, request_id=request_id).to_dict()


def explain_selection(prompt: str) -> str:
    """Return human-readable explanation of the routing decision."""
    result = _get_model().predict(prompt)
    return result.explain()
