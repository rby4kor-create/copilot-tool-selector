"""
setup_project.py
Run from your repository root:
    python setup_project.py
"""

import os
from pathlib import Path


def write(path, content):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  created: {path}")


print("\nCreating production ML tool selector files...\n")

# =============================================================================
write("config/config.yaml", """\
data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  training_source: "synthetic"
  synthetic_data_path: "data/processed/training_data.csv"
  manager_data_path: "data/raw/manager_dataset.csv"
  tool_catalog_path: "data/processed/tool_catalog.json"
  tool_catalog_version: "1.0.0"

model:
  output_dir: "models"
  artifact_name: "tool_selector_pipeline.joblib"
  metadata_name: "metadata.json"
  version: "1.0.0"
  algorithm: "logistic_regression"
  logistic_regression:
    C: 1.0
    max_iter: 1000
    solver: "lbfgs"
    class_weight: "balanced"
    random_state: 42

features:
  tfidf:
    max_features: 2000
    ngram_range: [1, 2]
    min_df: 1
    sublinear_tf: true
  keyword_groups:
    grep_search:
      - "find"
      - "search"
      - "grep"
      - "pattern"
      - "text"
      - "string"
      - "match"
      - "contain"
      - "occurrence"
      - "usage"
      - "where is"
      - "locate"
      - "exact"
    codebase_search:
      - "codebase"
      - "semantic"
      - "similar"
      - "related"
      - "function"
      - "class"
      - "method"
      - "symbol"
      - "definition"
      - "implementation"
      - "across"
      - "project"
    read_file:
      - "read"
      - "open"
      - "view"
      - "content"
      - "show file"
      - "display"
      - "contents of"
      - "what is in"
      - "look at"
    list_dir:
      - "list"
      - "directory"
      - "folder"
      - "files"
      - "structure"
      - "what files"
      - "show files"
      - "ls"
      - "tree"
    run_terminal_cmd:
      - "run"
      - "execute"
      - "command"
      - "terminal"
      - "shell"
      - "bash"
      - "script"
      - "install"
      - "build"
      - "test"
      - "deploy"
    edit_file:
      - "edit"
      - "create"
      - "write"
      - "modify"
      - "update"
      - "change"
      - "add"
      - "delete"
      - "refactor"
      - "fix"
      - "rename"
    web_search:
      - "web"
      - "internet"
      - "documentation"
      - "docs"
      - "how to"
      - "tutorial"
      - "example"
      - "library"
      - "package"
      - "api"
      - "guide"
      - "best practice"
      - "external"

split:
  test_size: 0.20
  validation_size: 0.10
  random_state: 42

thresholds:
  default: 0.50
  per_tool:
    grep_search: 0.50
    codebase_search: 0.50
    read_file: 0.50
    list_dir: 0.50
    run_terminal_cmd: 0.50
    edit_file: 0.50
    web_search: 0.50

fallback:
  strategy: "select_all"
  default_tools: []

logging:
  level: "INFO"
  log_prompt_content: false
""")

# =============================================================================
write("data/raw/prompts.json", """\
[
  {"prompt": "Find all functions that call authenticate_user()", "relevant_tools": ["grep_search", "codebase_search"]},
  {"prompt": "Search for all usages of login() in the codebase", "relevant_tools": ["grep_search", "codebase_search"]},
  {"prompt": "Find all occurrences of TODO comments", "relevant_tools": ["grep_search"]},
  {"prompt": "Locate the definition of UserService class", "relevant_tools": ["codebase_search"]},
  {"prompt": "Find code similar to this authentication logic", "relevant_tools": ["codebase_search"]},
  {"prompt": "Search for all imports of pandas library", "relevant_tools": ["grep_search"]},
  {"prompt": "Find all methods that return None", "relevant_tools": ["grep_search", "codebase_search"]},
  {"prompt": "What files are in the src directory?", "relevant_tools": ["list_dir"]},
  {"prompt": "List all files in the project root", "relevant_tools": ["list_dir"]},
  {"prompt": "Show me the directory structure", "relevant_tools": ["list_dir"]},
  {"prompt": "What is in the config folder?", "relevant_tools": ["list_dir"]},
  {"prompt": "Show all folders in data/", "relevant_tools": ["list_dir"]},
  {"prompt": "Read the contents of README.md", "relevant_tools": ["read_file"]},
  {"prompt": "Show me what is in config.yaml", "relevant_tools": ["read_file"]},
  {"prompt": "Open the requirements.txt file", "relevant_tools": ["read_file"]},
  {"prompt": "Display the contents of src/train.py", "relevant_tools": ["read_file"]},
  {"prompt": "View the main configuration file", "relevant_tools": ["read_file"]},
  {"prompt": "Run the test suite", "relevant_tools": ["run_terminal_cmd"]},
  {"prompt": "Execute the build script", "relevant_tools": ["run_terminal_cmd"]},
  {"prompt": "Install the dependencies", "relevant_tools": ["run_terminal_cmd"]},
  {"prompt": "Run python train.py", "relevant_tools": ["run_terminal_cmd"]},
  {"prompt": "Execute all unit tests with pytest", "relevant_tools": ["run_terminal_cmd"]},
  {"prompt": "Deploy the application", "relevant_tools": ["run_terminal_cmd"]},
  {"prompt": "Edit the configuration file", "relevant_tools": ["edit_file"]},
  {"prompt": "Create a new Python file called helper.py", "relevant_tools": ["edit_file"]},
  {"prompt": "Modify the train.py to add logging", "relevant_tools": ["edit_file"]},
  {"prompt": "Fix the bug in predict.py", "relevant_tools": ["edit_file", "codebase_search"]},
  {"prompt": "Refactor the UserService class", "relevant_tools": ["edit_file", "codebase_search"]},
  {"prompt": "Update the README with new instructions", "relevant_tools": ["edit_file"]},
  {"prompt": "Add a new function to utils.py", "relevant_tools": ["edit_file"]},
  {"prompt": "How do I use the requests library?", "relevant_tools": ["web_search"]},
  {"prompt": "What is the best way to handle exceptions in Python?", "relevant_tools": ["web_search"]},
  {"prompt": "Find documentation for scikit-learn LogisticRegression", "relevant_tools": ["web_search"]},
  {"prompt": "How to install numpy on Windows?", "relevant_tools": ["web_search"]},
  {"prompt": "What are best practices for REST API design?", "relevant_tools": ["web_search"]},
  {"prompt": "Find all callers of process_payment() and run the payment tests", "relevant_tools": ["grep_search", "codebase_search", "run_terminal_cmd"]},
  {"prompt": "Refactor the database module and run tests after", "relevant_tools": ["edit_file", "codebase_search", "run_terminal_cmd"]},
  {"prompt": "Find the UserModel definition and show me the file", "relevant_tools": ["codebase_search", "read_file"]},
  {"prompt": "List all test files and run them", "relevant_tools": ["list_dir", "run_terminal_cmd"]},
  {"prompt": "Search for deprecated functions and fix them", "relevant_tools": ["grep_search", "edit_file"]},
  {"prompt": "Find similar authentication code and refactor it", "relevant_tools": ["codebase_search", "edit_file"]},
  {"prompt": "Read the error log and search for the exception", "relevant_tools": ["read_file", "grep_search"]},
  {"prompt": "Show me the src folder then read main.py", "relevant_tools": ["list_dir", "read_file"]},
  {"prompt": "Find all test functions that use mock objects", "relevant_tools": ["grep_search", "codebase_search"]},
  {"prompt": "Create a new config file and update the imports", "relevant_tools": ["edit_file"]},
  {"prompt": "Search the web for how to use pandas groupby", "relevant_tools": ["web_search"]},
  {"prompt": "Find the implementation of sort_users and optimize it", "relevant_tools": ["codebase_search", "edit_file"]},
  {"prompt": "List the models directory", "relevant_tools": ["list_dir"]},
  {"prompt": "Run the linter on the entire project", "relevant_tools": ["run_terminal_cmd"]},
  {"prompt": "Find all classes that inherit from BaseModel", "relevant_tools": ["grep_search", "codebase_search"]}
]
""")

# =============================================================================
write("requirements.txt", """\
scikit-learn>=1.3.0
numpy>=1.24.0
scipy>=1.11.0
pandas>=2.0.0
joblib>=1.3.0
pyyaml>=6.0
pytest>=7.4.0
pytest-cov>=4.1.0
""")

# =============================================================================
write("src/__init__.py", """\
\"\"\"
copilot-ml-tool-selector
ML-based tool selection for GitHub Copilot / MCP-style tools.

Primary API:
    from src.predict import select_tools
    result = select_tools("Find all functions calling authenticate_user()")
\"\"\"
""")

# =============================================================================
write("src/config.py", """\
\"\"\"
src/config.py
Single source of truth for all configuration.
All other modules import from here.
\"\"\"
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
""")

# =============================================================================
write("src/feature_engineering.py", """\
\"\"\"
src/feature_engineering.py
Production feature pipeline.
Fit ONCE during training. Load and reuse during inference. Never refit.
\"\"\"
from __future__ import annotations
import logging
import re
from typing import Dict, List
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from src.config import Config

logger = logging.getLogger(__name__)


def normalize_prompt(text: str) -> str:
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = text.lower().strip()
    text = re.sub(r"\\s+", " ", text)
    return text


class KeywordFeatureExtractor(BaseEstimator, TransformerMixin):
    \"\"\"One binary flag per tool keyword group.\"\"\"

    def __init__(self, keyword_groups: Dict[str, List[str]]) -> None:
        self.keyword_groups = keyword_groups
        self.feature_names_: List[str] = []

    def fit(self, X: List[str], y=None):
        self.feature_names_ = list(self.keyword_groups.keys())
        return self

    def transform(self, X: List[str]) -> np.ndarray:
        n = len(X)
        k = len(self.keyword_groups)
        result = np.zeros((n, k), dtype=np.float32)
        tool_names = list(self.keyword_groups.keys())
        for i, prompt in enumerate(X):
            pl = prompt.lower()
            for j, tool in enumerate(tool_names):
                if any(kw.lower() in pl for kw in self.keyword_groups[tool]):
                    result[i, j] = 1.0
        return result

    def get_feature_names_out(self) -> List[str]:
        return [f"keyword__{n}" for n in self.feature_names_]


class StructuralFeatureExtractor(BaseEstimator, TransformerMixin):
    \"\"\"Hand-crafted structural features from the prompt text.\"\"\"

    ACTION_VERBS = {
        "find", "search", "get", "show", "list", "read", "run", "execute",
        "edit", "create", "write", "update", "delete", "modify", "check",
        "view", "open", "look", "locate", "what", "how", "where", "display",
    }
    FILE_EXT = re.compile(
        r"\\.(py|js|ts|java|cpp|c|cs|go|rb|rs|md|json|yaml|yml|toml|txt)$",
        re.IGNORECASE,
    )

    def fit(self, X, y=None):
        return self

    def transform(self, X: List[str]) -> np.ndarray:
        result = np.zeros((len(X), 8), dtype=np.float32)
        for i, prompt in enumerate(X):
            words = prompt.split()
            result[i, 0] = min(len(prompt) / 200.0, 1.0)
            result[i, 1] = min(len(words) / 30.0, 1.0)
            result[i, 2] = float("(" in prompt or ")" in prompt)
            result[i, 3] = float("." in prompt)
            result[i, 4] = float("_" in prompt)
            result[i, 5] = float("?" in prompt)
            result[i, 6] = float(bool(self.FILE_EXT.search(prompt)))
            result[i, 7] = float(bool(words) and words[0].lower() in self.ACTION_VERBS)
        return result

    def get_feature_names_out(self) -> List[str]:
        return [
            "struct__char_length", "struct__word_count",
            "struct__has_parentheses", "struct__has_dot_notation",
            "struct__has_underscore", "struct__has_question_mark",
            "struct__has_file_extension", "struct__starts_with_verb",
        ]


class PromptNormalizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X: List[str]) -> List[str]:
        return [normalize_prompt(p) for p in X]


class PromptFeatureTransformer(BaseEstimator, TransformerMixin):
    \"\"\"
    Full feature pipeline: TF-IDF + Keyword + Structural.
    Fit once during training. Persist. Reuse during inference.
    \"\"\"

    def __init__(self, config: Config) -> None:
        self.config = config
        self._build()

    def _build(self) -> None:
        tc = self.config.tfidf_config
        ngram = tuple(tc["ngram_range"])
        self.normalizer_ = PromptNormalizer()
        self.tfidf_ = TfidfVectorizer(
            max_features=tc["max_features"],
            ngram_range=ngram,
            min_df=tc["min_df"],
            sublinear_tf=tc["sublinear_tf"],
            analyzer="word",
            token_pattern=r"(?u)\\b\\w+\\b",
        )
        self.keyword_ = KeywordFeatureExtractor(self.config.keyword_groups)
        self.structural_ = StructuralFeatureExtractor()

    def fit(self, X: List[str], y=None):
        norm = self.normalizer_.transform(X)
        self.tfidf_.fit(norm)
        self.keyword_.fit(norm)
        self.structural_.fit(norm)
        logger.info(
            "Transformer fitted. tfidf_vocab=%d keyword_groups=%d",
            len(self.tfidf_.vocabulary_), len(self.config.keyword_groups),
        )
        return self

    def transform(self, X: List[str]) -> sp.csr_matrix:
        norm = self.normalizer_.transform(X)
        return sp.hstack([
            self.tfidf_.transform(norm),
            sp.csr_matrix(self.keyword_.transform(norm)),
            sp.csr_matrix(self.structural_.transform(norm)),
        ], format="csr")

    def fit_transform(self, X: List[str], y=None) -> sp.csr_matrix:
        return self.fit(X).transform(X)

    def get_feature_count(self) -> int:
        return len(self.tfidf_.vocabulary_) + len(self.config.keyword_groups) + 8

    def get_feature_names(self) -> List[str]:
        return (
            self.tfidf_.get_feature_names_out().tolist()
            + self.keyword_.get_feature_names_out()
            + self.structural_.get_feature_names_out()
        )
""")

# =============================================================================
write("src/preprocessing.py", """\
\"\"\"
src/preprocessing.py
Manager dataset adapter. Auto-detects schema. Converts to canonical format.
\"\"\"
from __future__ import annotations
import json
import logging
from pathlib import Path
import pandas as pd
from src.config import Config

logger = logging.getLogger(__name__)
cfg = Config()
CANONICAL_TOOLS = cfg.all_tools()


def detect_schema(df: pd.DataFrame) -> str:
    cols = set(df.columns.str.lower().tolist())
    if "prompt" in cols and all(t in cols for t in CANONICAL_TOOLS):
        return "canonical"
    if "prompt" in cols and "tool" in cols and "label" in cols:
        return "paired_rows"
    if "prompt" in cols and ("tools" in cols or "relevant_tools" in cols):
        return "json_tools_col"
    return "unknown"


def from_paired_rows(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Converting paired-row format")
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()
    df = df[df["tool"].isin(set(CANONICAL_TOOLS))].copy()
    pivot = df.pivot_table(
        index="prompt", columns="tool",
        values="label", aggfunc="max", fill_value=0,
    ).reset_index()
    for t in CANONICAL_TOOLS:
        if t not in pivot.columns:
            pivot[t] = 0
    return pivot[["prompt"] + CANONICAL_TOOLS]


def from_json_tools_col(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Converting json-tools-col format")
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()
    col = "tools" if "tools" in df.columns else "relevant_tools"

    def parse(val):
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            try:
                return json.loads(val)
            except Exception:
                return [v.strip() for v in val.split(",") if v.strip()]
        return []

    df["_p"] = df[col].apply(parse)
    for t in CANONICAL_TOOLS:
        df[t] = df["_p"].apply(lambda x: int(t in x))
    return df[["prompt"] + CANONICAL_TOOLS].copy()


def validate_canonical(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in ["prompt"] + CANONICAL_TOOLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df.dropna(subset=["prompt"])
    df = df[df["prompt"].str.strip() != ""].copy()
    for t in CANONICAL_TOOLS:
        df[t] = pd.to_numeric(df[t], errors="coerce").fillna(0).astype(int).clip(0, 1)
    logger.info("Validated. Rows: %d", len(df))
    return df


def load_and_validate_dataset(path, source="synthetic") -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    logger.info("Loading %s dataset: %s", source, path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows, cols=%s", len(df), df.columns.tolist())
    schema = detect_schema(df)
    logger.info("Schema: %s", schema)
    if schema == "paired_rows":
        df = from_paired_rows(df)
    elif schema == "json_tools_col":
        df = from_json_tools_col(df)
    elif schema != "canonical":
        raise ValueError(f"Unknown schema. Columns: {df.columns.tolist()}")
    return validate_canonical(df)


def describe_dataset(df: pd.DataFrame) -> None:
    print("\\n" + "=" * 60)
    print("DATASET DESCRIPTION")
    print("=" * 60)
    print(f"Shape : {df.shape}")
    print(f"Missing:\\n{df.isnull().sum().to_string()}")
    print(f"Duplicate prompts: {df.duplicated(subset=['prompt']).sum()}")
    print("\\nTool distribution:")
    for t in CANONICAL_TOOLS:
        pos = (df[t] == 1).sum()
        print(f"  {t:<25} SELECT={pos} ({100.0 * pos / len(df):.1f}%)")
    print("=" * 60 + "\\n")
""")

# =============================================================================
write("src/data_generator.py", """\
\"\"\"
src/data_generator.py
SYNTHETIC DATA GENERATOR - Development only.
For production use the manager-approved dataset.
\"\"\"
from __future__ import annotations
import json
import logging
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_training_data() -> pd.DataFrame:
    print("\\n[WARNING] Generating SYNTHETIC data for development only.\\n")
    config = Config()
    tools = config.all_tools()
    prompts_path = config.raw_data_dir / "prompts.json"
    if not prompts_path.exists():
        raise FileNotFoundError(f"Not found: {prompts_path}")
    with open(prompts_path) as f:
        prompts = json.load(f)
    logger.info("Loaded %d prompts", len(prompts))
    rows = []
    for item in prompts:
        text = item.get("prompt", "").strip()
        if not text:
            continue
        relevant = set(item.get("relevant_tools", []))
        for tool in tools:
            rows.append({"prompt": text, "tool": tool, "label": 1 if tool in relevant else 0})
    df = pd.DataFrame(rows)
    out = config.synthetic_data_path
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info("Generated %d rows -> %s", len(df), out)
    return df


if __name__ == "__main__":
    generate_training_data()
""")

# =============================================================================
write("src/train.py", """\
\"\"\"
src/train.py
Production training pipeline.
Multilabel: one LogisticRegression binary classifier per tool.
\"\"\"
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import Config
from src.feature_engineering import PromptFeatureTransformer
from src.preprocessing import describe_dataset, load_and_validate_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("train")


class MultiLabelToolSelector:
    \"\"\"One LogisticRegression binary classifier per tool.\"\"\"

    def __init__(self, tools: List[str], lr_config: Dict) -> None:
        self.tools = tools
        self.lr_config = lr_config
        self.classifiers_: Dict[str, LogisticRegression] = {}

    def fit(self, X, y_df: pd.DataFrame):
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
    print(f"\\n{'='*65}")
    print(f"EVALUATION - {split_name.upper()}")
    print(f"{'='*65}")
    print(f"  {'Tool':<25} {'Thresh':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
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
        prompts, y_df, test_size=config.test_size, random_state=config.split_random_state)
    vf = config.validation_size / (1.0 - config.test_size)
    p_tr, p_va, y_tr, y_va = train_test_split(
        p_tv, y_tv, test_size=vf, random_state=config.split_random_state)

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
""")

# =============================================================================
write("src/evaluate.py", """\
\"\"\"
src/evaluate.py
Standalone evaluation. Loads trained model. Evaluates on dataset.
\"\"\"
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import Config
from src.preprocessing import load_and_validate_dataset, describe_dataset
from src.train import evaluate_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("evaluate")


def evaluate(data_path=None) -> None:
    config = Config()
    if not config.model_artifact_path.exists():
        logger.error("No model at: %s  Run: python src/train.py", config.model_artifact_path)
        sys.exit(1)
    artifact = joblib.load(config.model_artifact_path)
    transformer = artifact["transformer"]
    selector = artifact["selector"]
    thresholds = artifact["thresholds"]
    tools = artifact["tools"]
    if config.model_metadata_path.exists():
        with open(config.model_metadata_path) as f:
            meta = json.load(f)
        print(f"Model v{meta.get('model_version')} | {meta.get('training_timestamp')}")
    path = Path(data_path) if data_path else config.synthetic_data_path
    df = load_and_validate_dataset(path)
    describe_dataset(df)
    X = transformer.transform(df["prompt"].tolist())
    evaluate_model(selector, X, df[tools], thresholds, "evaluation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None)
    args = parser.parse_args()
    evaluate(data_path=args.data)
""")

# =============================================================================
write("src/predict.py", """\
\"\"\"
src/predict.py
Production inference. Model loaded ONCE. Never reloaded per prediction.

Usage:
    from src.predict import select_tools
    result = select_tools("Find all functions calling authenticate_user()")
\"\"\"
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
    \"\"\"Loads model once. Reuses for all predictions.\"\"\"

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
            raise ModelNotFoundError(f"Model not found: {path}\\nRun: python src/train.py")
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
            return self._fallback(rid, prompt or "", "empty_prompt", (time.perf_counter()-t0)*1000)
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
            lat = (time.perf_counter()-t0)*1000
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
            return self._fallback(rid, prompt, str(e), (time.perf_counter()-t0)*1000)

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
    \"\"\"Primary production entry point.\"\"\"
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
            print(f"\\nPrompt : {result['prompt']}")
            print(f"Model  : {result['model_version']}")
            print("\\nSELECTED:")
            for t in result["selected_tools"]:
                print(f"  [SELECT]   {t['tool']:<25} score={t['score']:.4f}")
            print("\\nREJECTED:")
            for t in result["rejected_tools"]:
                print(f"  [DESELECT] {t['tool']:<25} score={t['score']:.4f}")
            print(f"\\nLatency: {result['latency_ms']:.1f}ms | Fallback: {result['fallback_used']}")
    except ModelNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
""")

# =============================================================================
write("src/router.py", """\
\"\"\"
src/router.py
Tool routing layer. Separate from ML logic.
ML decides relevance. Router applies business policies.
\"\"\"
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
    \"\"\"Applies post-ML routing policies on top of ML scores.\"\"\"

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
""")

# =============================================================================
write("src/create_tool_catalog.py", """\
\"\"\"
src/create_tool_catalog.py
Creates the tool catalog. Keep in sync with config.yaml keyword_groups.
\"\"\"
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
""")

# =============================================================================
write("src/build_tool_catalog.py", """\
\"\"\"
src/build_tool_catalog.py
Builds tool catalog from raw copilot_tools.json.
Falls back to create_tool_catalog if raw file missing.
\"\"\"
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
""")

# =============================================================================
write("tests/__init__.py", "# tests package\n")

# =============================================================================
write("tests/test_feature_engineering.py", """\
\"\"\"Tests for feature engineering pipeline.\"\"\"
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import Config
from src.feature_engineering import (
    KeywordFeatureExtractor, PromptFeatureTransformer,
    StructuralFeatureExtractor, normalize_prompt,
)

cfg = Config()

TRAIN_PROMPTS = [
    "find all functions that call authenticate_user()",
    "read the main configuration file",
    "list all files in the src directory",
    "run the test suite",
    "edit the config file",
    "search the web for documentation",
    "find code similar to this implementation",
    "show me the contents of README.md",
    "create a new Python file called helper.py",
    "find all classes that extend BaseModel",
]


class TestNormalizePrompt:
    def test_lowercase(self):
        assert normalize_prompt("FIND ALL FUNCTIONS") == "find all functions"

    def test_strips_whitespace(self):
        assert normalize_prompt("  find function  ") == "find function"

    def test_collapses_spaces(self):
        result = normalize_prompt("find   all   functions")
        assert result == "find all functions"

    def test_none_input(self):
        assert isinstance(normalize_prompt(None), str)

    def test_empty_string(self):
        assert normalize_prompt("") == ""

    def test_preserves_underscores(self):
        assert "authenticate_user" in normalize_prompt("Find authenticate_user()")


class TestKeywordFeatureExtractor:
    def setup_method(self):
        self.ext = KeywordFeatureExtractor(cfg.keyword_groups)
        self.ext.fit(TRAIN_PROMPTS)

    def test_output_shape(self):
        r = self.ext.transform(["find all functions", "read the file"])
        assert r.shape == (2, len(cfg.keyword_groups))

    def test_grep_search_detected(self):
        r = self.ext.transform(["find all occurrences of this pattern"])
        idx = list(cfg.keyword_groups.keys()).index("grep_search")
        assert r[0, idx] == 1.0

    def test_binary_output(self):
        r = self.ext.transform(["find all functions"])
        assert set(r.flatten().tolist()).issubset({0.0, 1.0})

    def test_feature_names_length(self):
        assert len(self.ext.get_feature_names_out()) == len(cfg.keyword_groups)


class TestStructuralFeatureExtractor:
    def setup_method(self):
        self.ext = StructuralFeatureExtractor()
        self.ext.fit([])

    def test_output_shape(self):
        r = self.ext.transform(["find authenticate_user()", "list files"])
        assert r.shape == (2, 8)

    def test_parentheses_detected(self):
        r = self.ext.transform(["find authenticate_user()"])
        assert r[0, 2] == 1.0

    def test_underscore_detected(self):
        r = self.ext.transform(["find authenticate_user"])
        assert r[0, 4] == 1.0

    def test_question_mark_detected(self):
        r = self.ext.transform(["how do I do this?"])
        assert r[0, 5] == 1.0

    def test_empty_prompt_no_crash(self):
        r = self.ext.transform([""])
        assert r.shape == (1, 8)

    def test_feature_count(self):
        assert len(self.ext.get_feature_names_out()) == 8


class TestPromptFeatureTransformer:
    def setup_method(self):
        self.t = PromptFeatureTransformer(cfg)
        self.t.fit(TRAIN_PROMPTS)

    def test_returns_sparse(self):
        assert sp.issparse(self.t.fit_transform(TRAIN_PROMPTS))

    def test_shape_consistent(self):
        X_train = self.t.fit_transform(TRAIN_PROMPTS)
        X_new = self.t.transform(["find something"])
        assert X_new.shape[1] == X_train.shape[1]

    def test_no_nan(self):
        X = self.t.transform(["find all functions"])
        assert not np.any(np.isnan(X.toarray()))

    def test_no_inf(self):
        X = self.t.transform(["find all functions"])
        assert not np.any(np.isinf(X.toarray()))

    def test_empty_prompt_no_crash(self):
        assert self.t.transform([""]).shape[0] == 1

    def test_feature_count_positive(self):
        assert self.t.get_feature_count() > 0

    def test_names_match_count(self):
        assert len(self.t.get_feature_names()) == self.t.get_feature_count()

    def test_different_prompts_different_vectors(self):
        X = self.t.transform([
            "find all functions that call authenticate_user()",
            "search the web for python documentation",
        ])
        assert not np.allclose(X.getrow(0).toarray(), X.getrow(1).toarray())
""")

# =============================================================================
write("tests/test_predict.py", """\
\"\"\"Tests for production inference.\"\"\"
from __future__ import annotations
import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.predict import ModelNotFoundError, ToolSelectorModel, select_tools
from src.config import Config

cfg = Config()


def model_is_trained():
    return cfg.model_artifact_path.exists()


SKIP = pytest.mark.skipif(
    not model_is_trained(),
    reason="No trained model - run: python src/train.py",
)


class TestModelNotFound:
    def test_raises_when_model_missing(self, tmp_path):
        fake = MagicMock()
        fake.model_artifact_path = tmp_path / "none.joblib"
        fake.model_metadata_path = tmp_path / "none.json"
        with pytest.raises(ModelNotFoundError):
            ToolSelectorModel(config=fake)


@SKIP
class TestValidPredictions:
    def setup_method(self):
        self.model = ToolSelectorModel()

    def test_predict_not_none(self):
        assert self.model.predict("Find all functions calling authenticate_user()") is not None

    def test_has_selected_tools(self):
        r = self.model.predict("Find all functions calling authenticate_user()")
        assert isinstance(r.selected_tools, list)

    def test_has_rejected_tools(self):
        r = self.model.predict("Find all functions calling authenticate_user()")
        assert isinstance(r.rejected_tools, list)

    def test_all_tools_accounted(self):
        r = self.model.predict("Find all functions calling authenticate_user()")
        returned = (
            {t["tool"] for t in r.selected_tools}
            | {t["tool"] for t in r.rejected_tools}
        )
        assert returned == set(self.model.tools)

    def test_scores_valid(self):
        r = self.model.predict("Find all functions calling authenticate_user()")
        for t in r.selected_tools + r.rejected_tools:
            assert -1.0 <= t["score"] <= 1.0

    def test_select_tools_returns_dict(self):
        r = select_tools("Find all functions calling authenticate_user()")
        assert isinstance(r, dict)
        assert "selected_tools" in r
        assert "rejected_tools" in r

    def test_model_not_reloaded(self):
        m = ToolSelectorModel()
        v1 = m.predict("find functions").model_version
        v2 = m.predict("list all files").model_version
        assert v1 == v2


@SKIP
class TestEdgeCases:
    def setup_method(self):
        self.model = ToolSelectorModel()

    def test_empty_prompt_fallback(self):
        assert self.model.predict("").fallback_used is True

    def test_none_prompt_fallback(self):
        assert self.model.predict(None).fallback_used is True

    def test_whitespace_prompt_fallback(self):
        assert self.model.predict("   ").fallback_used is True

    def test_single_word_no_crash(self):
        assert self.model.predict("find") is not None

    def test_special_chars_no_crash(self):
        assert self.model.predict("!@#$%^&*()") is not None

    def test_unseen_prompt_no_crash(self):
        assert self.model.predict("xyzzy frob blorb quux") is not None
""")

# =============================================================================
write("tests/test_threshold.py", """\
\"\"\"Tests for threshold logic.\"\"\"
from __future__ import annotations
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import Config

cfg = Config()


class TestThresholdConfig:
    def test_default_threshold_exists(self):
        assert cfg.default_threshold is not None

    def test_default_threshold_valid(self):
        assert 0.0 < cfg.default_threshold < 1.0

    def test_per_tool_thresholds_exist(self):
        assert len(cfg.per_tool_thresholds) > 0

    def test_per_tool_thresholds_valid(self):
        for tool, t in cfg.per_tool_thresholds.items():
            assert 0.0 < t < 1.0, f"Bad threshold for {tool}: {t}"

    def test_unknown_tool_returns_default(self):
        assert cfg.per_tool_threshold("nonexistent_xyz") == cfg.default_threshold


class TestThresholdLogic:
    @pytest.mark.parametrize("score,threshold,expected", [
        (0.91, 0.50, "SELECT"),
        (0.84, 0.50, "SELECT"),
        (0.50, 0.50, "SELECT"),
        (0.499, 0.50, "DESELECT"),
        (0.13, 0.50, "DESELECT"),
        (0.70, 0.70, "SELECT"),
        (0.699, 0.70, "DESELECT"),
        (1.0, 0.50, "SELECT"),
        (0.0, 0.50, "DESELECT"),
    ])
    def test_decision(self, score, threshold, expected):
        result = "SELECT" if score >= threshold else "DESELECT"
        assert result == expected

    def test_all_tools_have_threshold(self):
        for tool in cfg.all_tools():
            t = cfg.per_tool_threshold(tool)
            assert isinstance(t, float)
            assert 0.0 < t <= 1.0
""")

# =============================================================================
write("tests/test_router.py", """\
\"\"\"Tests for tool router.\"\"\"
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
""")

# =============================================================================
