"""
src/predict.py
Production inference. Model loaded ONCE. Never reloaded per prediction.

Usage:
    from src.predict import select_tools
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import Config
from src.model_classes import MultiLabelToolSelector  # needed for joblib deserialization

logger = logging.getLogger(__name__)


class ModelNotFoundError(RuntimeError):
    pass


class ModelLoadError(RuntimeError):
    pass


class ToolSelectionResult:
    def __init__(self, request_id, prompt, selected_tools, rejected_tools,
                 thresholds_used, model_version, fallback_used=False, latency_ms=0.0):
        self.request_id = request_id
        self.prompt = prompt
        self.selected_tools = selected_tools
        self.rejected_tools = rejected_tools
        self.thresholds_used = thresholds_used
        self.model_version = model_version
        self.fallback_used = fallback_used
        self.latency_ms = latency_ms

    def to_dict(self, include_prompt=True) -> Dict[str, Any]:
        d = {
            "request_id": self.request_id,
            "model_version": self.model_version,
            "selected_tools": self.selected_tools,
            "rejected_tools": self.rejected_tools,
            "thresholds_used": self.thresholds_used,
            "fallback_used": self.fallback_used,
            "latency_ms": round(self.latency_ms, 2),
        }
        if include_prompt:
            d["prompt"] = self.prompt
        return d

    def __repr__(self):
        sel = [t["tool"] for t in self.selected_tools]
        return f"ToolSelectionResult(selected={sel}, fallback={self.fallback_used})"


class ToolSelectorModel:
    """Loads model once. Reuses for all predictions."""

    def __init__(self, config=None):
        self.config = config or Config()
        self._loaded = False
        self._transformer = None
        self._selector = None
        self._thresholds = {}
        self._tools = []
        self._model_version = "unknown"
        self._load_model()

    def _load_model(self):
        path = self.config.model_artifact_path
        if not path.exists():
            raise ModelNotFoundError(
                f"Model not found: {path}\nRun: python src/train.py"
            )
        try:
            art = joblib.load(path)
            self._transformer = art["transformer"]
            self._selector = art["selector"]
            self._thresholds = art["thresholds"]
            self._tools = art["tools"]
            self._loaded = True
            mp = self.config.model_metadata_path
            if mp.exists():
                with open(mp) as f:
                    self._model_version = json.load(f).get("model_version", "unknown")
            logger.info("Model loaded v%s tools=%s", self._model_version, self._tools)
        except ModelNotFoundError:
            raise
        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {e}") from e

    def predict(self, prompt, request_id=None) -> ToolSelectionResult:
        t0 = time.perf_counter()
        rid = request_id or str(uuid.uuid4())[:12]
        if not prompt or not isinstance(prompt, str) or not prompt.strip():
            logger.warning("[%s] Empty prompt - fallback", rid)
            return self._fallback(rid, prompt or "", "empty_prompt",
                                  (time.perf_counter() - t0) * 1000)
        prompt = prompt.strip()
        try:
            X = self._transformer.transform([prompt])
            probas = self._selector.predict_proba(X)
            selected, rejected = [], []
            for tool in self._tools:
                score = float(probas[tool][0])
                thresh = self._thresholds.get(tool, self.config.default_threshold)
                entry = {"tool": tool, "score": round(score, 4)}
                (selected if score >= thresh else rejected).append(entry)
            selected.sort(key=lambda x: x["score"], reverse=True)
            rejected.sort(key=lambda x: x["score"], reverse=True)
            lat = (time.perf_counter() - t0) * 1000
            lp = prompt if self.config.log_prompt_content else f"[{len(prompt)} chars]"
            logger.info("[%s] %s selected=%s lat=%.1fms",
                        rid, lp, [t["tool"] for t in selected], lat)
            return ToolSelectionResult(
                request_id=rid, prompt=prompt,
                selected_tools=selected, rejected_tools=rejected,
                thresholds_used=self._thresholds.copy(),
                model_version=self._model_version,
                fallback_used=False, latency_ms=lat,
            )
        except Exception as e:
            logger.error("[%s] Prediction failed: %s", rid, e, exc_info=True)
            return self._fallback(rid, prompt, str(e),
                                  (time.perf_counter() - t0) * 1000)

    def _fallback(self, rid, prompt, reason, latency_ms) -> ToolSelectionResult:
        strategy = self.config.fallback_strategy
        tools = self._tools or self.config.all_tools()
        thresh = self._thresholds or {t: self.config.default_threshold for t in tools}
        logger.warning("[%s] Fallback strategy=%s reason=%s", rid, strategy, reason)
        if strategy == "select_all":
            selected = [{"tool": t, "score": -1.0} for t in tools]
            rejected = []
        elif strategy == "select_none":
            selected = []
            rejected = [{"tool": t, "score": -1.0} for t in tools]
        else:
            defs = set(self.config.fallback_default_tools)
            selected = [{"tool": t, "score": -1.0} for t in tools if t in defs]
            rejected = [{"tool": t, "score": -1.0} for t in tools if t not in defs]
        return ToolSelectionResult(
            request_id=rid, prompt=prompt,
            selected_tools=selected, rejected_tools=rejected,
            thresholds_used=thresh, model_version=self._model_version,
            fallback_used=True, latency_ms=latency_ms,
        )

    @property
    def is_loaded(self):
        return self._loaded

    @property
    def tools(self):
        return self._tools.copy()


_singleton: Optional[ToolSelectorModel] = None


def _get_model() -> ToolSelectorModel:
    global _singleton
    if _singleton is None:
        _singleton = ToolSelectorModel()
    return _singleton


def select_tools(prompt: str, request_id=None) -> Dict[str, Any]:
    """Primary production entry point."""
    return _get_model().predict(prompt, request_id=request_id).to_dict()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = select_tools(args.prompt)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"\nPrompt : {result['prompt']}")
            print(f"Model  : {result['model_version']}")
            print("\nSELECTED:")
            for t in result["selected_tools"]:
                print(f"  [SELECT]   {t['tool']:<25} score={t['score']:.4f}")
            print("\nREJECTED:")
            for t in result["rejected_tools"]:
                print(f"  [DESELECT] {t['tool']:<25} score={t['score']:.4f}")
            print(f"\nLatency: {result['latency_ms']:.1f}ms | Fallback: {result['fallback_used']}")
    except ModelNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
