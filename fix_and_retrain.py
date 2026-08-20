"""
fix_and_retrain.py
Fixes the pickle import error and retrains the model cleanly.
Run from repo root: python fix_and_retrain.py
"""

from pathlib import Path
import shutil

# Step 1: Remove old broken model
models_dir = Path("models")
if models_dir.exists():
    shutil.rmtree(models_dir)
    print("Deleted old models/ directory")

models_dir.mkdir(exist_ok=True)
print("Created fresh models/ directory")

# Step 2: The fix - train.py must be run as a module, not a script
# But the real fix is to move MultiLabelToolSelector to its own file
# so joblib can always find it regardless of how predict.py is run.

# Write the fixed model_classes.py
Path("src/model_classes.py").write_text('''\
"""
src/model_classes.py
Contains MultiLabelToolSelector class.
Must be a separate importable module so joblib can deserialize it
regardless of which script is the __main__ entry point.
"""
from __future__ import annotations
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class MultiLabelToolSelector:
    """One LogisticRegression binary classifier per tool."""

    def __init__(self, tools: List[str], lr_config: Dict) -> None:
        self.tools = tools
        self.lr_config = lr_config
        self.classifiers_: Dict[str, LogisticRegression] = {}

    def fit(self, X, y_df: pd.DataFrame):
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Training %d classifiers", len(self.tools))
        for tool in self.tools:
            y = y_df[tool].values
            logger.info("  %-25s pos=%d neg=%d", tool, y.sum(), len(y) - y.sum())
            clf = LogisticRegression(
                C=self.lr_config["C"],
                max_iter=self.lr_config["max_iter"],
                solver=self.lr_config["solver"],
                class_weight=self.lr_config.get("class_weight", "balanced"),
                random_state=self.lr_config["random_state"],
            )
            clf.fit(X, y)
            self.classifiers_[tool] = clf
        return self

    def predict_proba(self, X) -> Dict[str, np.ndarray]:
        return {t: clf.predict_proba(X)[:, 1] for t, clf in self.classifiers_.items()}

    def predict(self, X, thresholds: Dict[str, float]) -> Dict[str, np.ndarray]:
        p = self.predict_proba(X)
        return {t: (p[t] >= thresholds.get(t, 0.5)).astype(int) for t in self.tools}
''')
print("Created src/model_classes.py")

# Step 3: Update train.py to import from model_classes
train_content = '''\
"""
src/train.py
Production training pipeline.
Multilabel: one LogisticRegression binary classifier per tool.
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import Config
from src.feature_engineering import PromptFeatureTransformer
from src.preprocessing import describe_dataset, load_and_validate_dataset
from src.model_classes import MultiLabelToolSelector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("train")


def tune_thresholds(selector, X_val, y_val_df: pd.DataFrame) -> Dict[str, float]:
    candidates = [round(t, 2) for t in np.arange(0.10, 0.95, 0.05)]
    logger.info("Tuning thresholds on validation set")
    probas = selector.predict_proba(X_val)
    best = {}
    for tool in selector.tools:
        y_true = y_val_df[tool].values
        proba = probas[tool]
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            best[tool] = 0.50
            continue
        bf, bt = -1.0, 0.50
        for t in candidates:
            f = f1_score(y_true, (proba >= t).astype(int), zero_division=0)
            if f > bf:
                bf, bt = f, t
        logger.info("  %-25s threshold=%.2f val_F1=%.4f", tool, bt, bf)
        best[tool] = bt
    return best


def evaluate_model(selector, X_test, y_test_df, thresholds, split_name="test") -> Dict:
    probas = selector.predict_proba(X_test)
    all_yt, all_yp = [], []
    per_tool = {}
    print(f"\\n{\'=\'*65}")
    print(f"EVALUATION - {split_name.upper()}")
    print(f"{\'=\'*65}")
    print(f"  {\'Tool\':<25} {\'Thresh\':>7} {\'Prec\':>7} {\'Rec\':>7} {\'F1\':>7}")
    print("-" * 65)
    for tool in selector.tools:
        y_true = y_test_df[tool].values
        proba = probas[tool]
        t = thresholds.get(tool, 0.50)
        y_pred = (proba >= t).astype(int)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_true, proba) if y_true.sum() > 0 else float("nan")
        except Exception:
            auc = float("nan")
        print(f"  {tool:<25} {t:>7.2f} {prec:>7.4f} {rec:>7.4f} {f1:>7.4f}")
        per_tool[tool] = {
            "threshold": t, "precision": round(prec, 4),
            "recall": round(rec, 4), "f1": round(f1, 4),
            "tp": int(((y_pred == 1) & (y_true == 1)).sum()),
            "fp": int(((y_pred == 1) & (y_true == 0)).sum()),
            "tn": int(((y_pred == 0) & (y_true == 0)).sum()),
            "fn": int(((y_pred == 0) & (y_true == 1)).sum()),
        }
        all_yt.append(y_true)
        all_yp.append(y_pred)
    Yt = np.stack(all_yt, axis=1)
    Yp = np.stack(all_yp, axis=1)
    micro = f1_score(Yt, Yp, average="micro", zero_division=0)
    macro = f1_score(Yt, Yp, average="macro", zero_division=0)
    exact = float((Yt == Yp).all(axis=1).mean())
    sm = Yp == 1
    rm = Yp == 0
    sa = float((Yt[sm] == 1).mean()) if sm.sum() > 0 else float("nan")
    ra = float((Yt[rm] == 0).mean()) if rm.sum() > 0 else float("nan")
    print(f"\\nMicro F1={micro:.4f} | Macro F1={macro:.4f} | Exact={exact:.4f}")
    print(f"Selection Acc={sa:.4f} | Rejection Acc={ra:.4f}")
    print("=" * 65 + "\\n")
    return {
        "micro_f1": round(micro, 4), "macro_f1": round(macro, 4),
        "exact_match_accuracy": round(exact, 4),
        "tool_selection_accuracy": round(sa, 4) if not np.isnan(sa) else None,
        "tool_rejection_accuracy": round(ra, 4) if not np.isnan(ra) else None,
        "per_tool": per_tool,
    }


def train(source: Optional[str] = None) -> None:
    t0 = time.time()
    run_id = str(uuid.uuid4())[:8]
    config = Config()
    logger.info("Training run: %s", run_id)

    effective = source or config.training_source
    if effective == "synthetic":
        df = load_and_validate_dataset(config.synthetic_data_path, "synthetic")
    elif effective == "manager":
        df = load_and_validate_dataset(config.manager_data_path, "manager")
    elif effective == "combined":
        d1 = load_and_validate_dataset(config.synthetic_data_path, "synthetic")
        d2 = load_and_validate_dataset(config.manager_data_path, "manager")
        df = pd.concat([d1, d2], ignore_index=True)
    else:
        raise ValueError(f"Unknown source: {effective}")

    describe_dataset(df)
    tools = config.all_tools()

    before = len(df)
    df = df.groupby("prompt", as_index=False)[tools].max()
    logger.info("Deduplicated %d -> %d", before, len(df))

    prompts = df["prompt"].tolist()
    y_df = df[tools]

    p_tv, p_te, y_tv, y_te = train_test_split(
        prompts, y_df, test_size=config.test_size,
        random_state=config.split_random_state)
    vf = config.validation_size / (1.0 - config.test_size)
    p_tr, p_va, y_tr, y_va = train_test_split(
        p_tv, y_tv, test_size=vf,
        random_state=config.split_random_state)

    y_tr_df = pd.DataFrame(y_tr, columns=tools)
    y_va_df = pd.DataFrame(y_va, columns=tools)
    y_te_df = pd.DataFrame(y_te, columns=tools)
    logger.info("Split: train=%d val=%d test=%d", len(p_tr), len(p_va), len(p_te))

    transformer = PromptFeatureTransformer(config)
    X_tr = transformer.fit_transform(p_tr)
    X_va = transformer.transform(p_va)
    X_te = transformer.transform(p_te)
    logger.info("Feature shape: %s", X_tr.shape)

    selector = MultiLabelToolSelector(tools=tools, lr_config=config.lr_config)
    selector.fit(X_tr, y_tr_df)

    thresholds = tune_thresholds(selector, X_va, y_va_df)
    metrics = evaluate_model(selector, X_te, y_te_df, thresholds)

    config.ensure_dirs()
    joblib.dump(
        {"transformer": transformer, "selector": selector,
         "thresholds": thresholds, "tools": tools},
        config.model_artifact_path,
    )
    logger.info("Model saved: %s", config.model_artifact_path)

    meta = {
        "run_id": run_id, "model_version": config.model_version,
        "training_source": effective, "tools": tools,
        "thresholds": thresholds, "train_size": len(p_tr),
        "val_size": len(p_va), "test_size": len(p_te),
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "training_duration_seconds": round(time.time() - t0, 2),
        "evaluation_metrics": metrics,
    }
    with open(config.model_metadata_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    logger.info("Done %.1fs | Micro F1=%.4f", time.time() - t0, metrics["micro_f1"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["synthetic", "manager", "combined"], default=None)
    args = parser.parse_args()
    train(source=args.source)
'''

Path("src/train.py").write_text(train_content)
print("Updated src/train.py")

# Step 4: Update predict.py to import from model_classes
predict_content = '''\
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
                f"Model not found: {path}\\nRun: python src/train.py"
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
            print(f"\\nPrompt : {result[\'prompt\']}")
            print(f"Model  : {result[\'model_version\']}")
            print("\\nSELECTED:")
            for t in result["selected_tools"]:
                print(f"  [SELECT]   {t[\'tool\']:<25} score={t[\'score\']:.4f}")
            print("\\nREJECTED:")
            for t in result["rejected_tools"]:
                print(f"  [DESELECT] {t[\'tool\']:<25} score={t[\'score\']:.4f}")
            print(f"\\nLatency: {result[\'latency_ms\']:.1f}ms | Fallback: {result[\'fallback_used\']}")
    except ModelNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
'''

Path("src/predict.py").write_text(predict_content)
print("Updated src/predict.py")

print("\nNow retraining the model...")

# Step 5: Retrain
import subprocess
result = subprocess.run(
    ["python", "src/train.py"],
    capture_output=False,
    text=True
)

if result.returncode == 0:
    print("\nTraining complete!")
    print("\nNow run your prediction:")
    print('  python src/predict.py --prompt "Find all functions that call authenticate_user()"')
else:
    print("\nTraining failed. Check error above.")