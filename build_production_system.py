"""
build_production_system.py
Complete A-to-Z production ML tool selector.
Run from repo root: python build_production_system.py
"""

import os
import shutil
from pathlib import Path


def write(path, content):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  created: {path}")


print("\nBuilding production ML tool selector system...\n")

# =============================================================================
# CONFIGURATION
# =============================================================================
write("configs/config.yaml", """\
# =============================================================================
# Production ML Tool Selector Configuration
# =============================================================================

data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  training_source: "synthetic"
  synthetic_data_path: "data/processed/training_data.csv"
  manager_data_path: "data/raw/manager_dataset.csv"
  tool_catalog_path: "data/processed/tool_catalog.json"
  tool_catalog_version: "2.0.0"

model:
  output_dir: "models"
  artifact_name: "tool_selector_pipeline.joblib"
  metadata_name: "metadata.json"
  version: "2.0.0"
  algorithm: "logistic_regression"
  logistic_regression:
    C: 1.0
    max_iter: 2000
    solver: "lbfgs"
    class_weight: "balanced"
    random_state: 42
  linear_svm:
    C: 1.0
    max_iter: 2000
    random_state: 42
    class_weight: "balanced"

features:
  tfidf:
    max_features: 3000
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
      - "all places"
      - "all files"
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
      - "who calls"
      - "callers"
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
      - "show me"
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
      - "what is in the"
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
      - "deploy"
      - "start"
      - "launch"
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
      - "implement"
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
      - "api reference"
      - "guide"
      - "best practice"
      - "external"
      - "stackoverflow"

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

experiments:
  output_dir: "experiments"
  track: true
""")

# =============================================================================
# EXPANDED TRAINING DATA
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
  {"prompt": "Find all occurrences of the word password in the codebase", "relevant_tools": ["grep_search"]},
  {"prompt": "Search for all print statements in the project", "relevant_tools": ["grep_search"]},
  {"prompt": "Find all files containing the word deprecated", "relevant_tools": ["grep_search"]},
  {"prompt": "Locate all uses of os.path.join", "relevant_tools": ["grep_search"]},
  {"prompt": "Find all lines that import numpy", "relevant_tools": ["grep_search"]},
  {"prompt": "Search for the string connection_string in all files", "relevant_tools": ["grep_search"]},
  {"prompt": "Find all functions that call send_email()", "relevant_tools": ["grep_search", "codebase_search"]},
  {"prompt": "Find all callers of validate_token() function", "relevant_tools": ["grep_search", "codebase_search"]},
  {"prompt": "Search for all usages of the UserRepository class", "relevant_tools": ["grep_search", "codebase_search"]},
  {"prompt": "Find all places where DatabaseError is caught", "relevant_tools": ["grep_search", "codebase_search"]},
  {"prompt": "Locate all calls to the render_template function", "relevant_tools": ["grep_search", "codebase_search"]},
  {"prompt": "Find the implementation of the payment processor", "relevant_tools": ["codebase_search"]},
  {"prompt": "Where is the database connection initialized?", "relevant_tools": ["codebase_search"]},
  {"prompt": "Find the class that handles user authentication", "relevant_tools": ["codebase_search"]},
  {"prompt": "Locate the function that sends emails", "relevant_tools": ["codebase_search"]},
  {"prompt": "Find the method responsible for token validation", "relevant_tools": ["codebase_search"]},
  {"prompt": "Where is the logging configuration set up?", "relevant_tools": ["codebase_search"]},
  {"prompt": "Find the UserController class definition", "relevant_tools": ["codebase_search"]},
  {"prompt": "Find all exception handlers in the codebase", "relevant_tools": ["grep_search", "codebase_search"]},
  {"prompt": "What files are in the src directory?", "relevant_tools": ["list_dir"]},
  {"prompt": "List all files in the project root", "relevant_tools": ["list_dir"]},
  {"prompt": "Show me the directory structure", "relevant_tools": ["list_dir"]},
  {"prompt": "What is in the config folder?", "relevant_tools": ["list_dir"]},
  {"prompt": "Show all folders in data/", "relevant_tools": ["list_dir"]},
  {"prompt": "What files exist in the tests folder?", "relevant_tools": ["list_dir"]},
  {"prompt": "Show me the contents of the models directory", "relevant_tools": ["list_dir"]},
  {"prompt": "List all Python files in the project", "relevant_tools": ["list_dir"]},
  {"prompt": "What is in the scripts folder?", "relevant_tools": ["list_dir"]},
  {"prompt": "Show me the project structure", "relevant_tools": ["list_dir"]},
  {"prompt": "List the files in the data directory", "relevant_tools": ["list_dir"]},
  {"prompt": "What folders are in the root directory?", "relevant_tools": ["list_dir"]},
  {"prompt": "List the models directory", "relevant_tools": ["list_dir"]},
  {"prompt": "Read the contents of README.md", "relevant_tools": ["read_file"]},
  {"prompt": "Show me what is in config.yaml", "relevant_tools": ["read_file"]},
  {"prompt": "Open the requirements.txt file", "relevant_tools": ["read_file"]},
  {"prompt": "Display the contents of src/train.py", "relevant_tools": ["read_file"]},
  {"prompt": "View the main configuration file", "relevant_tools": ["read_file"]},
  {"prompt": "Show me the content of the .env file", "relevant_tools": ["read_file"]},
  {"prompt": "What does the Dockerfile contain?", "relevant_tools": ["read_file"]},
  {"prompt": "Display the contents of setup.py", "relevant_tools": ["read_file"]},
  {"prompt": "Open the pyproject.toml file", "relevant_tools": ["read_file"]},
  {"prompt": "Read the contents of the main.py file", "relevant_tools": ["read_file"]},
  {"prompt": "Show me what is inside the constants.py file", "relevant_tools": ["read_file"]},
  {"prompt": "View the migration script", "relevant_tools": ["read_file"]},
  {"prompt": "What functions are defined in utils.py?", "relevant_tools": ["read_file"]},
  {"prompt": "Show me the requirements.txt file", "relevant_tools": ["read_file"]},
  {"prompt": "Run the test suite", "relevant_tools": ["run_terminal_cmd"]},
  {"prompt": "Execute the build script", "relevant_tools": ["run_terminal_cmd"]},
  {"prompt": "Install the dependencies", "relevant_tools": ["run_terminal_cmd"]},
  {"prompt": "Run python train.py", "relevant_tools": ["run_terminal_cmd"]},
  {"prompt": "Execute all unit tests with pytest", "relevant_tools": ["run_terminal_cmd"]},
  {"prompt": "Deploy the application", "relevant_tools": ["run_terminal_cmd"]},
  {"prompt": "Run the database migrations", "relevant_tools": ["run_terminal_cmd"]},
  {"prompt": "Execute the setup script", "relevant_tools": ["run_terminal_cmd"]},
  {"prompt": "Run pytest with coverage", "relevant_tools": ["run_terminal_cmd"]},
  {"prompt": "Start the development server", "relevant_tools": ["run_terminal_cmd"]},
  {"prompt": "Run the formatter on all Python files", "relevant_tools": ["run_terminal_cmd"]},
  {"prompt": "Execute the deployment pipeline", "relevant_tools": ["run_terminal_cmd"]},
  {"prompt": "Run mypy type checking", "relevant_tools": ["run_terminal_cmd"]},
  {"prompt": "Install all project dependencies", "relevant_tools": ["run_terminal_cmd"]},
  {"prompt": "Run the security audit on the project", "relevant_tools": ["run_terminal_cmd"]},
  {"prompt": "Run the linter on the entire project", "relevant_tools": ["run_terminal_cmd"]},
  {"prompt": "Edit the configuration file", "relevant_tools": ["edit_file"]},
  {"prompt": "Create a new Python file called helper.py", "relevant_tools": ["edit_file"]},
  {"prompt": "Modify the train.py to add logging", "relevant_tools": ["edit_file"]},
  {"prompt": "Fix the bug in predict.py", "relevant_tools": ["edit_file", "codebase_search"]},
  {"prompt": "Update the README with new instructions", "relevant_tools": ["edit_file"]},
  {"prompt": "Add a new function to utils.py", "relevant_tools": ["edit_file"]},
  {"prompt": "Create a new file called constants.py", "relevant_tools": ["edit_file"]},
  {"prompt": "Add error handling to the login function", "relevant_tools": ["edit_file"]},
  {"prompt": "Update the version number in setup.py", "relevant_tools": ["edit_file"]},
  {"prompt": "Write a new test for the payment module", "relevant_tools": ["edit_file"]},
  {"prompt": "Delete the old migration file", "relevant_tools": ["edit_file"]},
  {"prompt": "Rename the config file to settings.py", "relevant_tools": ["edit_file"]},
  {"prompt": "Add a docstring to the UserService class", "relevant_tools": ["edit_file"]},
  {"prompt": "Create a Docker compose file for the project", "relevant_tools": ["edit_file"]},
  {"prompt": "How do I use the requests library?", "relevant_tools": ["web_search"]},
  {"prompt": "What is the best way to handle exceptions in Python?", "relevant_tools": ["web_search"]},
  {"prompt": "Find documentation for scikit-learn LogisticRegression", "relevant_tools": ["web_search"]},
  {"prompt": "How to install numpy on Windows?", "relevant_tools": ["web_search"]},
  {"prompt": "What are best practices for REST API design?", "relevant_tools": ["web_search"]},
  {"prompt": "What is the difference between OAuth and JWT?", "relevant_tools": ["web_search"]},
  {"prompt": "How do I configure CORS in Flask?", "relevant_tools": ["web_search"]},
  {"prompt": "Find the documentation for the boto3 library", "relevant_tools": ["web_search"]},
  {"prompt": "What are the best practices for Python logging?", "relevant_tools": ["web_search"]},
  {"prompt": "How to deploy a FastAPI application to AWS?", "relevant_tools": ["web_search"]},
  {"prompt": "What is the latest version of scikit-learn?", "relevant_tools": ["web_search"]},
  {"prompt": "How do I use async await in Python?", "relevant_tools": ["web_search"]},
  {"prompt": "How to use Redis for caching in Python?", "relevant_tools": ["web_search"]},
  {"prompt": "Search the web for how to use pandas groupby", "relevant_tools": ["web_search"]},
  {"prompt": "Find all callers of process_payment() and run the payment tests", "relevant_tools": ["grep_search", "codebase_search", "run_terminal_cmd"]},
  {"prompt": "Refactor the database module and run tests after", "relevant_tools": ["edit_file", "codebase_search", "run_terminal_cmd"]},
  {"prompt": "Find the UserModel definition and show me the file", "relevant_tools": ["codebase_search", "read_file"]},
  {"prompt": "List all test files and run them", "relevant_tools": ["list_dir", "run_terminal_cmd"]},
  {"prompt": "Search for deprecated functions and fix them", "relevant_tools": ["grep_search", "edit_file"]},
  {"prompt": "Find similar authentication code and refactor it", "relevant_tools": ["codebase_search", "edit_file"]},
  {"prompt": "Read the error log and search for the exception", "relevant_tools": ["read_file", "grep_search"]},
  {"prompt": "Show me the src folder then read main.py", "relevant_tools": ["list_dir", "read_file"]},
  {"prompt": "Find all test functions that use mock objects", "relevant_tools": ["grep_search", "codebase_search"]},
  {"prompt": "Find the send_notification function and update it", "relevant_tools": ["codebase_search", "edit_file"]},
  {"prompt": "Search for all TODO comments and fix them", "relevant_tools": ["grep_search", "edit_file"]},
  {"prompt": "Find the test files and run them", "relevant_tools": ["list_dir", "run_terminal_cmd"]},
  {"prompt": "Find the database models and add a new field", "relevant_tools": ["codebase_search", "edit_file"]},
  {"prompt": "Search for all API endpoints and document them", "relevant_tools": ["grep_search", "codebase_search", "edit_file"]},
  {"prompt": "Find the auth middleware and run the auth tests", "relevant_tools": ["codebase_search", "run_terminal_cmd"]},
  {"prompt": "List all config files and read the main one", "relevant_tools": ["list_dir", "read_file"]},
  {"prompt": "Find all functions that use deprecated methods and update them", "relevant_tools": ["grep_search", "codebase_search", "edit_file"]},
  {"prompt": "Create a new utility module and run the tests to verify", "relevant_tools": ["edit_file", "run_terminal_cmd"]},
  {"prompt": "Search for slow database queries and optimize them", "relevant_tools": ["grep_search", "codebase_search", "edit_file"]},
  {"prompt": "Refactor the UserService class and run the tests", "relevant_tools": ["edit_file", "codebase_search", "run_terminal_cmd"]},
  {"prompt": "Find the implementation of sort_users and optimize it", "relevant_tools": ["codebase_search", "edit_file"]},
  {"prompt": "Find all classes that inherit from BaseModel", "relevant_tools": ["grep_search", "codebase_search"]},
  {"prompt": "Create a new config file and update the imports", "relevant_tools": ["edit_file"]},
  {"prompt": "Find all classes that extend BaseModel", "relevant_tools": ["grep_search", "codebase_search"]}
]
""")

# =============================================================================
# SOURCE FILES
# =============================================================================

write("src/__init__.py", """\
\"\"\"
copilot-ml-tool-selector v2.0
Production ML-based tool selection.

Primary API:
    from src.inference.predict import select_tools
    result = select_tools("Find all functions calling authenticate_user()")
\"\"\"
""")

# =============================================================================
write("src/catalog/__init__.py", "")

write("src/catalog/tool_catalog.py", """\
\"\"\"
src/catalog/tool_catalog.py
Dynamic tool registry. Never hardcode tool names in business logic.
\"\"\"
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ToolCatalog:
    \"\"\"
    Dynamic tool registry loaded from disk.
    All tool names come from here — never hardcoded.
    \"\"\"

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
""")

# =============================================================================
write("src/features/__init__.py", "")

write("src/features/feature_engineering.py", """\
\"\"\"
src/features/feature_engineering.py
Production feature pipeline.

CRITICAL RULE:
    Fit ONCE on training data.
    Persist transformer.
    Load and transform (never refit) during inference.

Features produced per prompt:
    1. TF-IDF (up to 3000 features)
    2. Keyword group indicators (one per tool)
    3. Structural features (8 hand-crafted)
\"\"\"
from __future__ import annotations
import logging
import re
from typing import Dict, List
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


def normalize_prompt(text: str) -> str:
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = text.lower().strip()
    text = re.sub(r"\\s+", " ", text)
    return text


class KeywordFeatureExtractor(BaseEstimator, TransformerMixin):
    \"\"\"Binary flag per tool keyword group.\"\"\"

    def __init__(self, keyword_groups: Dict[str, List[str]]) -> None:
        self.keyword_groups = keyword_groups
        self.feature_names_: List[str] = []

    def fit(self, X: List[str], y=None):
        self.feature_names_ = list(self.keyword_groups.keys())
        return self

    def transform(self, X: List[str]) -> np.ndarray:
        n, k = len(X), len(self.keyword_groups)
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
    \"\"\"8 hand-crafted structural features.\"\"\"

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
    Complete feature pipeline: TF-IDF + Keyword + Structural.
    Single object persisted to disk. Loaded for inference. Never refit.
    \"\"\"

    def __init__(self, tfidf_config: Dict, keyword_groups: Dict) -> None:
        self.tfidf_config = tfidf_config
        self.keyword_groups = keyword_groups
        self._build()

    def _build(self) -> None:
        tc = self.tfidf_config
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
        self.keyword_ = KeywordFeatureExtractor(self.keyword_groups)
        self.structural_ = StructuralFeatureExtractor()

    def fit(self, X: List[str], y=None):
        norm = self.normalizer_.transform(X)
        self.tfidf_.fit(norm)
        self.keyword_.fit(norm)
        self.structural_.fit(norm)
        logger.info(
            "Transformer fitted. tfidf_vocab=%d keyword_groups=%d structural=8",
            len(self.tfidf_.vocabulary_), len(self.keyword_groups),
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
        return len(self.tfidf_.vocabulary_) + len(self.keyword_groups) + 8

    def get_feature_names(self) -> List[str]:
        return (
            self.tfidf_.get_feature_names_out().tolist()
            + self.keyword_.get_feature_names_out()
            + self.structural_.get_feature_names_out()
        )
""")

# =============================================================================
write("src/models/__init__.py", "")

write("src/models/model_classes.py", """\
\"\"\"
src/models/model_classes.py
ML model classes. Separate module so joblib can always deserialize
regardless of which script is __main__.
\"\"\"
from __future__ import annotations
import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

logger = logging.getLogger(__name__)


class MultiLabelToolSelector:
    \"\"\"
    One binary classifier per tool (One-vs-Rest multilabel).

    Supports:
        - LogisticRegression  (provides probabilities natively)
        - LinearSVC           (calibrated for probability estimates)

    Each classifier independently answers:
        'Should this tool be selected for this prompt?'
    \"\"\"

    def __init__(
        self,
        tools: List[str],
        algorithm: str = "logistic_regression",
        lr_config: Optional[Dict] = None,
        svm_config: Optional[Dict] = None,
    ) -> None:
        self.tools = tools
        self.algorithm = algorithm
        self.lr_config = lr_config or {}
        self.svm_config = svm_config or {}
        self.classifiers_: Dict[str, object] = {}

    def _make_classifier(self):
        if self.algorithm == "logistic_regression":
            return LogisticRegression(
                C=self.lr_config.get("C", 1.0),
                max_iter=self.lr_config.get("max_iter", 2000),
                solver=self.lr_config.get("solver", "lbfgs"),
                class_weight=self.lr_config.get("class_weight", "balanced"),
                random_state=self.lr_config.get("random_state", 42),
            )
        elif self.algorithm == "linear_svm":
            base = LinearSVC(
                C=self.svm_config.get("C", 1.0),
                max_iter=self.svm_config.get("max_iter", 2000),
                class_weight=self.svm_config.get("class_weight", "balanced"),
                random_state=self.svm_config.get("random_state", 42),
            )
            # Calibrate SVM to produce probability estimates
            return CalibratedClassifierCV(base, cv=3, method="sigmoid")
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def fit(self, X, y_df: pd.DataFrame):
        logger.info(
            "Training %d %s classifiers (one per tool)",
            len(self.tools), self.algorithm,
        )
        for tool in self.tools:
            y = y_df[tool].values
            pos = y.sum()
            neg = len(y) - pos
            logger.info("  %-25s pos=%d neg=%d", tool, pos, neg)
            if pos == 0:
                logger.warning("  %-25s has NO positive examples — skipping", tool)
                self.classifiers_[tool] = None
                continue
            if neg == 0:
                logger.warning("  %-25s has NO negative examples — skipping", tool)
                self.classifiers_[tool] = None
                continue
            clf = self._make_classifier()
            clf.fit(X, y)
            self.classifiers_[tool] = clf
        return self

    def predict_proba(self, X) -> Dict[str, np.ndarray]:
        result = {}
        for tool, clf in self.classifiers_.items():
            if clf is None:
                result[tool] = np.zeros(X.shape[0])
            else:
                result[tool] = clf.predict_proba(X)[:, 1]
        return result

    def predict(self, X, thresholds: Dict[str, float]) -> Dict[str, np.ndarray]:
        probas = self.predict_proba(X)
        return {
            t: (probas[t] >= thresholds.get(t, 0.5)).astype(int)
            for t in self.tools
        }
""")

# =============================================================================
write("src/data/__init__.py", "")

write("src/data/preprocessing.py", """\
\"\"\"
src/data/preprocessing.py
Dataset loading, schema detection, and canonical format conversion.
\"\"\"
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import List
import pandas as pd

logger = logging.getLogger(__name__)


def detect_schema(df: pd.DataFrame, tool_list: List[str]) -> str:
    cols = set(df.columns.str.lower().tolist())
    if "prompt" in cols and all(t in cols for t in tool_list):
        return "canonical"
    if "prompt" in cols and "tool" in cols and "label" in cols:
        return "paired_rows"
    if "prompt" in cols and ("tools" in cols or "relevant_tools" in cols):
        return "json_tools_col"
    return "unknown"


def from_paired_rows(df: pd.DataFrame, tool_list: List[str]) -> pd.DataFrame:
    logger.info("Converting paired-row format")
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()
    df = df[df["tool"].isin(set(tool_list))].copy()
    pivot = df.pivot_table(
        index="prompt", columns="tool",
        values="label", aggfunc="max", fill_value=0,
    ).reset_index()
    for t in tool_list:
        if t not in pivot.columns:
            pivot[t] = 0
    return pivot[["prompt"] + tool_list]


def from_json_tools_col(df: pd.DataFrame, tool_list: List[str]) -> pd.DataFrame:
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
    for t in tool_list:
        df[t] = df["_p"].apply(lambda x: int(t in x))
    return df[["prompt"] + tool_list].copy()


def validate_canonical(df: pd.DataFrame, tool_list: List[str]) -> pd.DataFrame:
    missing = [c for c in ["prompt"] + tool_list if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df.dropna(subset=["prompt"])
    df = df[df["prompt"].str.strip() != ""].copy()
    for t in tool_list:
        df[t] = pd.to_numeric(df[t], errors="coerce").fillna(0).astype(int).clip(0, 1)
    return df


def load_and_validate_dataset(
    path: str | Path,
    tool_list: List[str],
    source: str = "unknown",
) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    logger.info("Loading %s dataset: %s", source, path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows, cols=%s", len(df), df.columns.tolist())
    schema = detect_schema(df, tool_list)
    logger.info("Detected schema: %s", schema)
    if schema == "paired_rows":
        df = from_paired_rows(df, tool_list)
    elif schema == "json_tools_col":
        df = from_json_tools_col(df, tool_list)
    elif schema != "canonical":
        raise ValueError(f"Unknown schema. Columns: {df.columns.tolist()}")
    return validate_canonical(df, tool_list)


def describe_dataset(df: pd.DataFrame, tool_list: List[str]) -> None:
    print("\\n" + "=" * 65)
    print("DATASET DESCRIPTION")
    print("=" * 65)
    print(f"Shape            : {df.shape}")
    print(f"Unique prompts   : {df['prompt'].nunique()}")
    print(f"Duplicate prompts: {df.duplicated(subset=['prompt']).sum()}")
    print(f"Missing values   : {df.isnull().sum().sum()}")
    print("\\nTool label distribution:")
    print(f"  {'Tool':<25} {'SELECT':>8} {'DESELECT':>10} {'%SELECT':>9}")
    print("-" * 55)
    for t in tool_list:
        pos = (df[t] == 1).sum()
        neg = (df[t] == 0).sum()
        pct = 100.0 * pos / len(df) if len(df) > 0 else 0
        print(f"  {t:<25} {pos:>8} {neg:>10} {pct:>8.1f}%")
    n_tools_per_prompt = df[tool_list].sum(axis=1)
    print(f"\\nAvg tools per prompt : {n_tools_per_prompt.mean():.2f}")
    print(f"Max tools per prompt : {n_tools_per_prompt.max()}")
    print(f"Multi-tool prompts   : {(n_tools_per_prompt > 1).sum()}")
    print("=" * 65 + "\\n")
""")

# =============================================================================
write("src/data/data_quality.py", """\
\"\"\"
src/data/data_quality.py
Data quality checks before training.
\"\"\"
from __future__ import annotations
import logging
from typing import List, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


def run_data_quality_checks(
    df: pd.DataFrame,
    tool_list: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    \"\"\"
    Run all data quality checks.
    Returns cleaned DataFrame and list of issues found.
    \"\"\"
    issues = []
    original_len = len(df)

    # Check 1: Empty prompts
    empty = df["prompt"].str.strip().eq("").sum()
    if empty > 0:
        issues.append(f"Found {empty} empty prompts — removed")
        df = df[df["prompt"].str.strip() != ""].copy()

    # Check 2: Duplicate prompts (keep max label per tool)
    dupes = df.duplicated(subset=["prompt"]).sum()
    if dupes > 0:
        issues.append(f"Found {dupes} duplicate prompts — merged with OR logic")
        df = df.groupby("prompt", as_index=False)[tool_list].max()

    # Check 3: Rows with no tools selected (all zeros)
    no_tool = (df[tool_list].sum(axis=1) == 0).sum()
    if no_tool > 0:
        issues.append(f"Found {no_tool} prompts with no tool selected — kept (valid negative examples)")

    # Check 4: Tools with no positive examples
    for t in tool_list:
        pos = (df[t] == 1).sum()
        if pos == 0:
            issues.append(f"Tool '{t}' has ZERO positive examples — model cannot learn it")
        elif pos < 3:
            issues.append(f"Tool '{t}' has only {pos} positive examples — may underperform")

    # Check 5: Tools with no negative examples
    for t in tool_list:
        neg = (df[t] == 0).sum()
        if neg == 0:
            issues.append(f"Tool '{t}' has ZERO negative examples — model cannot learn it")

    # Check 6: Label validity
    for t in tool_list:
        invalid = (~df[t].isin([0, 1])).sum()
        if invalid > 0:
            issues.append(f"Tool '{t}' has {invalid} invalid labels — clipped to 0/1")
            df[t] = df[t].clip(0, 1)

    removed = original_len - len(df)
    if removed > 0:
        issues.append(f"Total rows removed: {removed}")

    print("\\n" + "=" * 65)
    print("DATA QUALITY REPORT")
    print("=" * 65)
    if not issues:
        print("  No issues found.")
    for issue in issues:
        print(f"  [CHECK] {issue}")
    print(f"\\n  Final dataset: {len(df)} prompts")
    print("=" * 65 + "\\n")

    return df, issues
""")

# =============================================================================
write("src/training/__init__.py", "")

write("src/training/trainer.py", """\
\"\"\"
src/training/trainer.py
Production training pipeline.
Trains Logistic Regression and Linear SVM.
Compares them. Selects the best.
\"\"\"
from __future__ import annotations
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, accuracy_score,
)
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def tune_thresholds(
    selector,
    X_val,
    y_val_df: pd.DataFrame,
    tool_list: List[str],
) -> Dict[str, float]:
    \"\"\"Find best threshold per tool using validation set.\"\"\"
    candidates = [round(t, 2) for t in np.arange(0.10, 0.95, 0.05)]
    probas = selector.predict_proba(X_val)
    best = {}
    logger.info("Tuning thresholds on validation set")
    for tool in tool_list:
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
        # Safety: never let threshold go below 0.30 to prevent false positives
        bt = max(bt, 0.30)
        logger.info("  %-25s threshold=%.2f val_F1=%.4f", tool, bt, bf)
        best[tool] = bt
    return best


def evaluate_selector(
    selector,
    X: np.ndarray,
    y_df: pd.DataFrame,
    thresholds: Dict[str, float],
    tool_list: List[str],
    split_name: str = "test",
    verbose: bool = True,
) -> Dict:
    \"\"\"Full multilabel evaluation.\"\"\"
    probas = selector.predict_proba(X)
    all_yt, all_yp = [], []
    per_tool = {}

    if verbose:
        print(f"\\n{'='*70}")
        print(f"EVALUATION — {split_name.upper()}")
        print(f"{'='*70}")
        print(f"  {'Tool':<25} {'T':>5} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7} {'TP':>5} {'FP':>5} {'FN':>5}")
        print("-" * 70)

    for tool in tool_list:
        y_true = y_df[tool].values
        proba = probas[tool]
        t = thresholds.get(tool, 0.50)
        y_pred = (proba >= t).astype(int)

        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_true, proba) if y_true.sum() > 0 and y_true.sum() < len(y_true) else float("nan")
        except Exception:
            auc = float("nan")

        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())

        if verbose:
            auc_str = f"{auc:.4f}" if not np.isnan(auc) else "  N/A "
            print(f"  {tool:<25} {t:>5.2f} {prec:>7.4f} {rec:>7.4f} {f1:>7.4f} {auc_str:>7} {tp:>5} {fp:>5} {fn:>5}")

        per_tool[tool] = {
            "threshold": t, "precision": round(prec, 4),
            "recall": round(rec, 4), "f1": round(f1, 4),
            "auc": round(auc, 4) if not np.isnan(auc) else None,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        }
        all_yt.append(y_true)
        all_yp.append(y_pred)

    Yt = np.stack(all_yt, axis=1)
    Yp = np.stack(all_yp, axis=1)

    micro_f1 = f1_score(Yt, Yp, average="micro", zero_division=0)
    macro_f1 = f1_score(Yt, Yp, average="macro", zero_division=0)
    weighted_f1 = f1_score(Yt, Yp, average="weighted", zero_division=0)
    micro_prec = precision_score(Yt, Yp, average="micro", zero_division=0)
    micro_rec = recall_score(Yt, Yp, average="micro", zero_division=0)
    exact = float((Yt == Yp).all(axis=1).mean())

    sm, rm = Yp == 1, Yp == 0
    sel_acc = float((Yt[sm] == 1).mean()) if sm.sum() > 0 else float("nan")
    rej_acc = float((Yt[rm] == 0).mean()) if rm.sum() > 0 else float("nan")

    total_possible = Yt.shape[0] * Yt.shape[1]
    total_selected = Yp.sum()
    tool_reduction = 1.0 - (total_selected / total_possible) if total_possible > 0 else 0.0
    avg_selected = Yp.sum(axis=1).mean()

    if verbose:
        print(f"{'─'*70}")
        print(f"  Micro  F1        : {micro_f1:.4f}")
        print(f"  Macro  F1        : {macro_f1:.4f}")
        print(f"  Weighted F1      : {weighted_f1:.4f}")
        print(f"  Micro  Precision : {micro_prec:.4f}")
        print(f"  Micro  Recall    : {micro_rec:.4f}")
        print(f"  Exact Match      : {exact:.4f}")
        print(f"  Selection Acc    : {sel_acc:.4f}  (when we SELECT, are we right?)")
        print(f"  Rejection Acc    : {rej_acc:.4f}  (when we DESELECT, are we right?)")
        print(f"  Tool Reduction   : {tool_reduction:.1%}  (fewer tools = less cost)")
        print(f"  Avg Tools/Prompt : {avg_selected:.2f}")
        print("=" * 70 + "\\n")

    return {
        "micro_f1": round(micro_f1, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "micro_precision": round(micro_prec, 4),
        "micro_recall": round(micro_rec, 4),
        "exact_match": round(exact, 4),
        "tool_selection_accuracy": round(sel_acc, 4) if not np.isnan(sel_acc) else None,
        "tool_rejection_accuracy": round(rej_acc, 4) if not np.isnan(rej_acc) else None,
        "tool_reduction": round(tool_reduction, 4),
        "avg_tools_per_prompt": round(float(avg_selected), 4),
        "per_tool": per_tool,
    }


def run_experiment(
    experiment_id: str,
    algorithm: str,
    X_train, X_val, X_test,
    y_train_df, y_val_df, y_test_df,
    tool_list: List[str],
    lr_config: Dict,
    svm_config: Dict,
    experiments_dir: Path,
) -> Tuple[Dict, Dict, object]:
    \"\"\"Run one complete experiment. Returns metrics, thresholds, selector.\"\"\"
    from src.models.model_classes import MultiLabelToolSelector

    print(f"\\n{'#'*70}")
    print(f"# EXPERIMENT: {experiment_id}  |  Algorithm: {algorithm}")
    print(f"{'#'*70}")

    t0 = time.time()
    selector = MultiLabelToolSelector(
        tools=tool_list,
        algorithm=algorithm,
        lr_config=lr_config,
        svm_config=svm_config,
    )
    selector.fit(X_train, y_train_df)
    train_time = time.time() - t0

    thresholds = tune_thresholds(selector, X_val, y_val_df, tool_list)

    t1 = time.time()
    metrics = evaluate_selector(
        selector, X_test, y_test_df, thresholds, tool_list,
        split_name=f"{experiment_id} ({algorithm})",
    )
    infer_time = (time.time() - t1) / len(y_test_df) * 1000

    metrics["training_time_seconds"] = round(train_time, 3)
    metrics["inference_latency_ms_per_prompt"] = round(infer_time, 3)
    metrics["algorithm"] = algorithm
    metrics["experiment_id"] = experiment_id

    # Save experiment record
    experiments_dir.mkdir(parents=True, exist_ok=True)
    exp_file = experiments_dir / f"{experiment_id}.json"
    with open(exp_file, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Experiment saved: {exp_file}")

    return metrics, thresholds, selector
""")

# =============================================================================
write("src/inference/__init__.py", "")

write("src/inference/predict.py", """\
\"\"\"
src/inference/predict.py
Production inference. Model loaded ONCE. Never reloaded per prediction.

Usage:
    from src.inference.predict import select_tools
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
        \"\"\"Human-readable explanation of the routing decision.\"\"\"
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
        return "\\n".join(lines)


class ToolSelectorModel:
    \"\"\"Production inference model. Loads once. Never reloads.\"\"\"

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
                f"Model not found: {self.model_path}\\n"
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
    \"\"\"Primary production entry point.\"\"\"
    return _get_model().predict(prompt, request_id=request_id).to_dict()


def explain_selection(prompt: str) -> str:
    \"\"\"Return human-readable explanation of the routing decision.\"\"\"
    result = _get_model().predict(prompt)
    return result.explain()
""")

# =============================================================================
write("src/routing/__init__.py", "")

write("src/routing/router.py", """\
\"\"\"
src/routing/router.py
Tool routing layer. Completely separate from ML logic.

ML answers:  Which tools are relevant?
Router answers: Which tools should actually execute?
\"\"\"
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
        return "\\n".join(lines)


class ToolRouter:
    \"\"\"
    Applies business routing policies on top of ML predictions.

    Policies (all optional):
        mandatory_tools    — always execute regardless of ML score
        banned_tools       — never execute regardless of ML score
        tool_dependencies  — if A selected, also select B
    \"\"\"

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
""")

# =============================================================================
# MAIN TRAINING SCRIPT
# =============================================================================
write("src/training/train.py", """\
\"\"\"
src/training/train.py
A-to-Z production training pipeline.

Phases:
    1. Load and validate dataset
    2. Data quality check
    3. Train/Val/Test split
    4. Feature engineering (fit on train only)
    5. Train Logistic Regression
    6. Train Linear SVM
    7. Compare models objectively
    8. Select best model
    9. Save artifact + metadata

Usage:
    python src/training/train.py
    python src/training/train.py --source manager
    python src/training/train.py --algorithm logistic_regression
    python src/training/train.py --algorithm linear_svm
    python src/training/train.py --algorithm compare  (trains both, picks best)
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
from sklearn.model_selection import train_test_split

# Allow running from repo root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.preprocessing import load_and_validate_dataset, describe_dataset
from src.data.data_quality import run_data_quality_checks
from src.features.feature_engineering import PromptFeatureTransformer
from src.models.model_classes import MultiLabelToolSelector
from src.training.trainer import tune_thresholds, evaluate_selector, run_experiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("train")


def load_config() -> Dict:
    config_path = REPO_ROOT / "configs" / "config.yaml"
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def train(source: str = None, algorithm: str = "compare") -> None:
    t_start = time.time()
    run_id = str(uuid.uuid4())[:8]
    config = load_config()

    logger.info("=" * 60)
    logger.info("Training run: %s | Algorithm: %s", run_id, algorithm)
    logger.info("=" * 60)

    # ── Load tool catalog ─────────────────────────────────────────────────────
    catalog_path = REPO_ROOT / config["data"]["tool_catalog_path"]
    if not catalog_path.exists():
        logger.warning("Tool catalog not found — creating from config")
        _create_catalog(config, catalog_path)

    with open(catalog_path) as f:
        raw_catalog = json.load(f)
    tool_list = [k for k in raw_catalog if not k.startswith("__")]
    logger.info("Tools: %d → %s", len(tool_list), tool_list)

    # ── Load dataset ──────────────────────────────────────────────────────────
    effective_source = source or config["data"]["training_source"]
    if effective_source == "synthetic":
        data_path = REPO_ROOT / config["data"]["synthetic_data_path"]
    elif effective_source == "manager":
        data_path = REPO_ROOT / config["data"]["manager_data_path"]
    else:
        raise ValueError(f"Unknown source: {effective_source}")

    df = load_and_validate_dataset(data_path, tool_list, effective_source)
    describe_dataset(df, tool_list)

    # ── Data quality ──────────────────────────────────────────────────────────
    df, issues = run_data_quality_checks(df, tool_list)

    # ── Split ─────────────────────────────────────────────────────────────────
    prompts = df["prompt"].tolist()
    y_df = df[tool_list]
    test_size = config["split"]["test_size"]
    val_size = config["split"]["validation_size"]
    seed = config["split"]["random_state"]

    p_tv, p_te, y_tv, y_te = train_test_split(
        prompts, y_df, test_size=test_size, random_state=seed)
    vf = val_size / (1.0 - test_size)
    p_tr, p_va, y_tr, y_va = train_test_split(
        p_tv, y_tv, test_size=vf, random_state=seed)

    y_tr_df = pd.DataFrame(y_tr, columns=tool_list)
    y_va_df = pd.DataFrame(y_va, columns=tool_list)
    y_te_df = pd.DataFrame(y_te, columns=tool_list)

    logger.info(
        "Split: train=%d  val=%d  test=%d",
        len(p_tr), len(p_va), len(p_te),
    )

    # ── Feature engineering ───────────────────────────────────────────────────
    logger.info("Fitting feature transformer on TRAINING data only")
    transformer = PromptFeatureTransformer(
        tfidf_config=config["features"]["tfidf"],
        keyword_groups=config["features"]["keyword_groups"],
    )
    X_tr = transformer.fit_transform(p_tr)
    X_va = transformer.transform(p_va)
    X_te = transformer.transform(p_te)
    logger.info("Feature shape: train=%s val=%s test=%s",
                X_tr.shape, X_va.shape, X_te.shape)

    experiments_dir = REPO_ROOT / config["experiments"]["output_dir"]

    # ── Train models ──────────────────────────────────────────────────────────
    lr_config = config["model"]["logistic_regression"]
    svm_config = config["model"]["linear_svm"]

    results = {}

    if algorithm in ("logistic_regression", "compare"):
        lr_metrics, lr_thresholds, lr_selector = run_experiment(
            experiment_id=f"{run_id}_LR",
            algorithm="logistic_regression",
            X_train=X_tr, X_val=X_va, X_test=X_te,
            y_train_df=y_tr_df, y_val_df=y_va_df, y_test_df=y_te_df,
            tool_list=tool_list,
            lr_config=lr_config, svm_config=svm_config,
            experiments_dir=experiments_dir,
        )
        results["logistic_regression"] = {
            "metrics": lr_metrics,
            "thresholds": lr_thresholds,
            "selector": lr_selector,
        }

    if algorithm in ("linear_svm", "compare"):
        svm_metrics, svm_thresholds, svm_selector = run_experiment(
            experiment_id=f"{run_id}_SVM",
            algorithm="linear_svm",
            X_train=X_tr, X_val=X_va, X_test=X_te,
            y_train_df=y_tr_df, y_val_df=y_va_df, y_test_df=y_te_df,
            tool_list=tool_list,
            lr_config=lr_config, svm_config=svm_config,
            experiments_dir=experiments_dir,
        )
        results["linear_svm"] = {
            "metrics": svm_metrics,
            "thresholds": svm_thresholds,
            "selector": svm_selector,
        }

    # ── Select best model ─────────────────────────────────────────────────────
    if len(results) == 1:
        best_algo = list(results.keys())[0]
    else:
        # Primary: Micro F1. Tiebreaker: Macro F1.
        best_algo = max(
            results,
            key=lambda a: (
                results[a]["metrics"]["micro_f1"],
                results[a]["metrics"]["macro_f1"],
            ),
        )

    best = results[best_algo]
    best_metrics = best["metrics"]
    best_thresholds = best["thresholds"]
    best_selector = best["selector"]

    # ── Print comparison ──────────────────────────────────────────────────────
    if len(results) > 1:
        print("\\n" + "=" * 70)
        print("MODEL COMPARISON")
        print("=" * 70)
        print(f"  {'Algorithm':<25} {'Micro F1':>10} {'Macro F1':>10} {'Precision':>10} {'Recall':>10}")
        print("-" * 70)
        for algo, r in results.items():
            m = r["metrics"]
            marker = " ← SELECTED" if algo == best_algo else ""
            print(f"  {algo:<25} {m['micro_f1']:>10.4f} {m['macro_f1']:>10.4f} "
                  f"{m['micro_precision']:>10.4f} {m['micro_recall']:>10.4f}{marker}")
        print("=" * 70)
        print(f"\\nSelected: {best_algo}")
        print(f"Reason: Highest Micro F1 = {best_metrics['micro_f1']:.4f}")
        print()

    # ── Save artifact ─────────────────────────────────────────────────────────
    output_dir = REPO_ROOT / config["model"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = output_dir / config["model"]["artifact_name"]
    artifact = {
        "transformer": transformer,
        "selector": best_selector,
        "thresholds": best_thresholds,
        "tools": tool_list,
    }
    joblib.dump(artifact, artifact_path)
    logger.info("Model saved: %s", artifact_path)

    # ── Save metadata ─────────────────────────────────────────────────────────
    meta = {
        "run_id": run_id,
        "model_version": config["model"]["version"],
        "algorithm": best_algo,
        "training_source": effective_source,
        "tools": tool_list,
        "num_tools": len(tool_list),
        "thresholds": best_thresholds,
        "train_size": len(p_tr),
        "val_size": len(p_va),
        "test_size": len(p_te),
        "feature_count": transformer.get_feature_count(),
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_training_seconds": round(time.time() - t_start, 2),
        "data_quality_issues": issues,
        "evaluation_metrics": best_metrics,
        "all_experiment_results": {
            algo: r["metrics"] for algo, r in results.items()
        },
    }
    meta_path = output_dir / config["model"]["metadata_name"]
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    logger.info("Metadata saved: %s", meta_path)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Algorithm     : {best_algo}")
    print(f"  Tools         : {len(tool_list)}")
    print(f"  Train prompts : {len(p_tr)}")
    print(f"  Micro F1      : {best_metrics['micro_f1']:.4f}")
    print(f"  Macro F1      : {best_metrics['macro_f1']:.4f}")
    print(f"  Exact Match   : {best_metrics['exact_match']:.4f}")
    print(f"  Tool Reduction: {best_metrics['tool_reduction']:.1%}")
    print(f"  Model saved   : {artifact_path}")
    print("=" * 70 + "\\n")


def _create_catalog(config: Dict, output_path: Path) -> None:
    \"\"\"Create tool catalog from config keyword groups.\"\"\"
    from datetime import datetime, timezone
    catalog = {}
    for tool, keywords in config["features"]["keyword_groups"].items():
        catalog[tool] = {
            "description": f"Tool: {tool}",
            "category": "auto",
            "keywords": keywords,
        }
    catalog["__meta__"] = {
        "version": config["data"]["tool_catalog_version"],
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        import json
        json.dump(catalog, f, indent=2)
    logger.info("Created tool catalog: %s", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML tool selector")
    parser.add_argument("--source", choices=["synthetic", "manager"], default=None)
    parser.add_argument(
        "--algorithm",
        choices=["logistic_regression", "linear_svm", "compare"],
        default="compare",
        help="compare trains both and picks best",
    )
    args = parser.parse_args()
    train(source=args.source, algorithm=args.algorithm)
""")

# =============================================================================
# EVALUATE SCRIPT
# =============================================================================
write("src/evaluation/evaluate.py", """\
\"\"\"
src/evaluation/evaluate.py
Standalone evaluation on any dataset.
\"\"\"
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

import joblib

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.preprocessing import load_and_validate_dataset, describe_dataset
from src.training.trainer import evaluate_selector
from src.models.model_classes import MultiLabelToolSelector
from src.features.feature_engineering import PromptFeatureTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("evaluate")


def evaluate(data_path: str = None) -> None:
    import yaml
    config_path = REPO_ROOT / "configs" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    artifact_path = REPO_ROOT / config["model"]["output_dir"] / config["model"]["artifact_name"]
    meta_path = REPO_ROOT / config["model"]["output_dir"] / config["model"]["metadata_name"]

    if not artifact_path.exists():
        logger.error("No model at: %s", artifact_path)
        logger.error("Run: python src/training/train.py")
        sys.exit(1)

    artifact = joblib.load(artifact_path)
    transformer = artifact["transformer"]
    selector = artifact["selector"]
    thresholds = artifact["thresholds"]
    tools = artifact["tools"]

    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"\\nModel v{meta.get('model_version')} | {meta.get('algorithm')} | {meta.get('training_timestamp')}")

    eval_path = Path(data_path) if data_path else REPO_ROOT / config["data"]["synthetic_data_path"]
    df = load_and_validate_dataset(eval_path, tools)
    describe_dataset(df, tools)

    X = transformer.transform(df["prompt"].tolist())
    evaluate_selector(selector, X, df[tools], thresholds, tools, "full evaluation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None)
    args = parser.parse_args()
    evaluate(data_path=args.data)
""")

write("src/evaluation/__init__.py", "")

# =============================================================================
# PREDICT CLI
# =============================================================================
write("predict.py", """\
\"\"\"
predict.py
CLI entry point for production inference.

Usage:
    python predict.py --prompt "Find all functions calling authenticate_user()"
    python predict.py --prompt "..." --json
    python predict.py --prompt "..." --explain
\"\"\"
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.inference.predict import select_tools, explain_selection, ModelNotFoundError

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML Tool Selector")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--explain", action="store_true", help="Explain decision")
    args = parser.parse_args()

    try:
        if args.explain:
            print(explain_selection(args.prompt))
        elif args.json:
            result = select_tools(args.prompt)
            print(json.dumps(result, indent=2))
        else:
            result = select_tools(args.prompt)
            print(f"\\nPrompt  : {result['prompt']}")
            print(f"Model   : v{result['model_version']} ({result['algorithm']})")
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
# DATA GENERATOR
# =============================================================================
write("src/data/data_generator.py", """\
\"\"\"Synthetic data generator. Development only.\"\"\"
from __future__ import annotations
import json
import logging
import sys
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_training_data() -> pd.DataFrame:
    print("\\n[WARNING] Generating SYNTHETIC data — development only\\n")
    import yaml
    with open(REPO_ROOT / "configs" / "config.yaml") as f:
        config = yaml.safe_load(f)

    tools = list(config["features"]["keyword_groups"].keys())
    prompts_path = REPO_ROOT / config["data"]["raw_dir"] / "prompts.json"

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
    out = REPO_ROOT / config["data"]["synthetic_data_path"]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info("Generated %d rows -> %s", len(df), out)
    return df


if __name__ == "__main__":
    generate_training_data()
""")

# =============================================================================
# TESTS
# =============================================================================
write("tests/__init__.py", "")

write("tests/test_features.py", """\
\"\"\"Tests for feature engineering.\"\"\"
import sys
from pathlib import Path
import numpy as np
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

from src.features.feature_engineering import (
    PromptFeatureTransformer, KeywordFeatureExtractor,
    StructuralFeatureExtractor, normalize_prompt,
)

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


class TestNormalize:
    def test_lowercase(self):
        assert normalize_prompt("FIND") == "find"

    def test_strips(self):
        assert normalize_prompt("  find  ") == "find"

    def test_none(self):
        assert isinstance(normalize_prompt(None), str)

    def test_empty(self):
        assert normalize_prompt("") == ""


class TestKeywordExtractor:
    def setup_method(self):
        self.ext = KeywordFeatureExtractor(cfg["features"]["keyword_groups"])
        self.ext.fit(TRAIN_PROMPTS)

    def test_shape(self):
        r = self.ext.transform(["find all", "read file"])
        assert r.shape == (2, len(cfg["features"]["keyword_groups"]))

    def test_binary(self):
        r = self.ext.transform(["find something"])
        assert set(r.flatten()).issubset({0.0, 1.0})

    def test_grep_detected(self):
        r = self.ext.transform(["find all occurrences"])
        idx = list(cfg["features"]["keyword_groups"].keys()).index("grep_search")
        assert r[0, idx] == 1.0


class TestStructuralExtractor:
    def setup_method(self):
        self.ext = StructuralFeatureExtractor()
        self.ext.fit([])

    def test_shape(self):
        r = self.ext.transform(["find authenticate_user()", "list files"])
        assert r.shape == (2, 8)

    def test_parens(self):
        r = self.ext.transform(["find func()"])
        assert r[0, 2] == 1.0

    def test_underscore(self):
        r = self.ext.transform(["find my_func"])
        assert r[0, 4] == 1.0

    def test_empty(self):
        r = self.ext.transform([""])
        assert r.shape == (1, 8)


class TestTransformer:
    def setup_method(self):
        self.t = PromptFeatureTransformer(
            cfg["features"]["tfidf"],
            cfg["features"]["keyword_groups"],
        )
        self.t.fit(TRAIN_PROMPTS)

    def test_sparse(self):
        assert sp.issparse(self.t.fit_transform(TRAIN_PROMPTS))

    def test_consistent_shape(self):
        X1 = self.t.fit_transform(TRAIN_PROMPTS)
        X2 = self.t.transform(["find something"])
        assert X1.shape[1] == X2.shape[1]

    def test_no_nan(self):
        X = self.t.transform(["find all functions"])
        assert not np.any(np.isnan(X.toarray()))

    def test_empty_no_crash(self):
        assert self.t.transform([""]).shape[0] == 1
""")

write("tests/test_predict.py", """\
\"\"\"Tests for production inference.\"\"\"
import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.predict import ModelNotFoundError, ToolSelectorModel, select_tools

MODEL_PATH = Path("models/tool_selector_pipeline.joblib")
META_PATH = Path("models/metadata.json")
MODEL_TRAINED = MODEL_PATH.exists()

SKIP = pytest.mark.skipif(not MODEL_TRAINED, reason="Run train.py first")


class TestModelNotFound:
    def test_raises(self, tmp_path):
        with pytest.raises(ModelNotFoundError):
            ToolSelectorModel(tmp_path / "none.joblib", tmp_path / "none.json")


@SKIP
class TestPredictions:
    def setup_method(self):
        self.model = ToolSelectorModel(MODEL_PATH, META_PATH)

    def test_returns_result(self):
        r = self.model.predict("Find all functions calling authenticate_user()")
        assert r is not None

    def test_all_tools_accounted(self):
        r = self.model.predict("Find all functions calling authenticate_user()")
        returned = {t["tool"] for t in r.selected_tools} | {t["tool"] for t in r.rejected_tools}
        assert returned == set(self.model.tools)

    def test_scores_valid(self):
        r = self.model.predict("Find all functions calling authenticate_user()")
        for t in r.selected_tools + r.rejected_tools:
            assert -1.0 <= t["score"] <= 1.0

    def test_select_tools_dict(self):
        r = select_tools("find something")
        assert isinstance(r, dict)
        assert "selected_tools" in r

    def test_empty_fallback(self):
        r = self.model.predict("")
        assert r.fallback_used is True

    def test_none_fallback(self):
        r = self.model.predict(None)
        assert r.fallback_used is True

    def test_no_crash_unseen(self):
        assert self.model.predict("xyzzy frob blorb") is not None


@SKIP
class TestThreshold:
    def setup_method(self):
        self.model = ToolSelectorModel(MODEL_PATH, META_PATH)

    def test_select_above_threshold(self):
        import joblib
        art = joblib.load(MODEL_PATH)
        thresholds = art["thresholds"]
        r = self.model.predict("Find all functions calling authenticate_user()")
        for t in r.selected_tools:
            thresh = thresholds.get(t["tool"], 0.5)
            assert t["score"] >= thresh

    def test_reject_below_threshold(self):
        import joblib
        art = joblib.load(MODEL_PATH)
        thresholds = art["thresholds"]
        r = self.model.predict("Find all functions calling authenticate_user()")
        for t in r.rejected_tools:
            thresh = thresholds.get(t["tool"], 0.5)
            assert t["score"] < thresh
""")

write("tests/test_router.py", """\
\"\"\"Tests for tool router.\"\"\"
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

from src.routing.router import ToolRouter, RoutingDecision

ALL_TOOLS = list(cfg["features"]["keyword_groups"].keys())
OTHER = [t for t in ALL_TOOLS if t != "grep_search"]


def ml_result(selected, rejected):
    return {
        "request_id": "test-001",
        "model_version": "2.0.0",
        "algorithm": "logistic_regression",
        "selected_tools": [{"tool": t, "score": 0.9} for t in selected],
        "rejected_tools": [{"tool": t, "score": 0.1} for t in rejected],
        "fallback_used": False,
    }


class TestBasic:
    def test_returns_decision(self):
        r = ToolRouter(ALL_TOOLS).route(ml_result(["grep_search"], OTHER))
        assert isinstance(r, RoutingDecision)

    def test_selected_in_execute(self):
        r = ToolRouter(ALL_TOOLS).route(ml_result(["grep_search"], OTHER))
        assert "grep_search" in r.execute

    def test_rejected_in_skip(self):
        r = ToolRouter(ALL_TOOLS).route(ml_result(["grep_search"], OTHER))
        assert "codebase_search" in r.skip

    def test_all_accounted(self):
        r = ToolRouter(ALL_TOOLS).route(ml_result(["grep_search"], OTHER))
        assert set(r.execute) | set(r.skip) == set(ALL_TOOLS)


class TestPolicies:
    def test_mandatory(self):
        router = ToolRouter(ALL_TOOLS, mandatory_tools=["grep_search"])
        r = router.route(ml_result([], ALL_TOOLS))
        assert "grep_search" in r.execute

    def test_banned(self):
        router = ToolRouter(ALL_TOOLS, banned_tools=["web_search"])
        r = router.route(ml_result(ALL_TOOLS, []))
        assert "web_search" not in r.execute

    def test_dependency(self):
        router = ToolRouter(ALL_TOOLS, tool_dependencies={"grep_search": ["codebase_search"]})
        non_gs = [t for t in ALL_TOOLS if t != "grep_search"]
        r = router.route(ml_result(["grep_search"], non_gs))
        assert "codebase_search" in r.execute
""")

# =============================================================================
# REQUIREMENTS
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
# README
# ======================================================================
