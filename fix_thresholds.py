"""
fix_thresholds.py
Fixes the low thresholds caused by small validation set.
Also adds more training data for better model quality.
Run: python fix_thresholds.py
"""

import json
import shutil
from pathlib import Path

print("\nStep 1: Adding more training prompts for better coverage...")

# More diverse prompts to improve model quality
extra_prompts = [
    # grep_search only
    {"prompt": "Find all occurrences of the word password in the codebase", "relevant_tools": ["grep_search"]},
    {"prompt": "Search for all print statements in the project", "relevant_tools": ["grep_search"]},
    {"prompt": "Find all files containing the word deprecated", "relevant_tools": ["grep_search"]},
    {"prompt": "Locate all uses of os.path.join", "relevant_tools": ["grep_search"]},
    {"prompt": "Find all lines that import numpy", "relevant_tools": ["grep_search"]},
    {"prompt": "Search for the string connection_string in all files", "relevant_tools": ["grep_search"]},

    # codebase_search only
    {"prompt": "Find the implementation of the payment processor", "relevant_tools": ["codebase_search"]},
    {"prompt": "Where is the database connection initialized?", "relevant_tools": ["codebase_search"]},
    {"prompt": "Find the class that handles user authentication", "relevant_tools": ["codebase_search"]},
    {"prompt": "Locate the function that sends emails", "relevant_tools": ["codebase_search"]},
    {"prompt": "Find the method responsible for token validation", "relevant_tools": ["codebase_search"]},
    {"prompt": "Where is the logging configuration set up?", "relevant_tools": ["codebase_search"]},

    # grep_search + codebase_search
    {"prompt": "Find all functions that call send_email()", "relevant_tools": ["grep_search", "codebase_search"]},
    {"prompt": "Find all callers of validate_token() function", "relevant_tools": ["grep_search", "codebase_search"]},
    {"prompt": "Search for all usages of the UserRepository class", "relevant_tools": ["grep_search", "codebase_search"]},
    {"prompt": "Find all places where DatabaseError is caught", "relevant_tools": ["grep_search", "codebase_search"]},
    {"prompt": "Locate all calls to the render_template function", "relevant_tools": ["grep_search", "codebase_search"]},

    # read_file only
    {"prompt": "Show me the content of the .env file", "relevant_tools": ["read_file"]},
    {"prompt": "What does the Dockerfile contain?", "relevant_tools": ["read_file"]},
    {"prompt": "Display the contents of setup.py", "relevant_tools": ["read_file"]},
    {"prompt": "Open the pyproject.toml file", "relevant_tools": ["read_file"]},
    {"prompt": "Read the contents of the main.py file", "relevant_tools": ["read_file"]},
    {"prompt": "Show me what is inside the constants.py file", "relevant_tools": ["read_file"]},
    {"prompt": "View the migration script", "relevant_tools": ["read_file"]},

    # list_dir only
    {"prompt": "What files exist in the tests folder?", "relevant_tools": ["list_dir"]},
    {"prompt": "Show me the contents of the models directory", "relevant_tools": ["list_dir"]},
    {"prompt": "List all Python files in the project", "relevant_tools": ["list_dir"]},
    {"prompt": "What is in the scripts folder?", "relevant_tools": ["list_dir"]},
    {"prompt": "Show me the project structure", "relevant_tools": ["list_dir"]},
    {"prompt": "List the files in the data directory", "relevant_tools": ["list_dir"]},
    {"prompt": "What folders are in the root directory?", "relevant_tools": ["list_dir"]},

    # run_terminal_cmd only
    {"prompt": "Run the database migrations", "relevant_tools": ["run_terminal_cmd"]},
    {"prompt": "Execute the setup script", "relevant_tools": ["run_terminal_cmd"]},
    {"prompt": "Run pytest with coverage", "relevant_tools": ["run_terminal_cmd"]},
    {"prompt": "Start the development server", "relevant_tools": ["run_terminal_cmd"]},
    {"prompt": "Run the formatter on all Python files", "relevant_tools": ["run_terminal_cmd"]},
    {"prompt": "Execute the deployment pipeline", "relevant_tools": ["run_terminal_cmd"]},
    {"prompt": "Run mypy type checking", "relevant_tools": ["run_terminal_cmd"]},
    {"prompt": "Install all project dependencies", "relevant_tools": ["run_terminal_cmd"]},

    # edit_file only
    {"prompt": "Create a new file called constants.py", "relevant_tools": ["edit_file"]},
    {"prompt": "Add error handling to the login function", "relevant_tools": ["edit_file"]},
    {"prompt": "Update the version number in setup.py", "relevant_tools": ["edit_file"]},
    {"prompt": "Write a new test for the payment module", "relevant_tools": ["edit_file"]},
    {"prompt": "Delete the old migration file", "relevant_tools": ["edit_file"]},
    {"prompt": "Rename the config file to settings.py", "relevant_tools": ["edit_file"]},
    {"prompt": "Add a docstring to the UserService class", "relevant_tools": ["edit_file"]},

    # web_search only
    {"prompt": "What is the difference between OAuth and JWT?", "relevant_tools": ["web_search"]},
    {"prompt": "How do I configure CORS in Flask?", "relevant_tools": ["web_search"]},
    {"prompt": "Find the documentation for the boto3 library", "relevant_tools": ["web_search"]},
    {"prompt": "What are the best practices for Python logging?", "relevant_tools": ["web_search"]},
    {"prompt": "How to deploy a FastAPI application to AWS?", "relevant_tools": ["web_search"]},
    {"prompt": "What is the latest version of scikit-learn?", "relevant_tools": ["web_search"]},
    {"prompt": "How do I use async await in Python?", "relevant_tools": ["web_search"]},

    # Multi-tool combinations
    {"prompt": "Find the send_notification function and update it", "relevant_tools": ["codebase_search", "edit_file"]},
    {"prompt": "Search for all TODO comments and fix them", "relevant_tools": ["grep_search", "edit_file"]},
    {"prompt": "Find the test files and run them", "relevant_tools": ["list_dir", "run_terminal_cmd"]},
    {"prompt": "Read the error log and find the exception location", "relevant_tools": ["read_file", "grep_search"]},
    {"prompt": "Find the database models and add a new field", "relevant_tools": ["codebase_search", "edit_file"]},
    {"prompt": "Search for all API endpoints and document them", "relevant_tools": ["grep_search", "codebase_search", "edit_file"]},
    {"prompt": "Find the auth middleware and run the auth tests", "relevant_tools": ["codebase_search", "run_terminal_cmd"]},
    {"prompt": "List all config files and read the main one", "relevant_tools": ["list_dir", "read_file"]},
    {"prompt": "Find all functions that use deprecated methods and update them", "relevant_tools": ["grep_search", "codebase_search", "edit_file"]},
    {"prompt": "Create a new utility module and run the tests to verify", "relevant_tools": ["edit_file", "run_terminal_cmd"]},
    {"prompt": "Search for slow database queries and optimize them", "relevant_tools": ["grep_search", "codebase_search", "edit_file"]},
    {"prompt": "Find the UserController class definition", "relevant_tools": ["codebase_search"]},
    {"prompt": "What functions are defined in utils.py?", "relevant_tools": ["read_file"]},
    {"prompt": "Find all exception handlers in the codebase", "relevant_tools": ["grep_search", "codebase_search"]},
    {"prompt": "Create a Docker compose file for the project", "relevant_tools": ["edit_file"]},
    {"prompt": "How to use Redis for caching in Python?", "relevant_tools": ["web_search"]},
    {"prompt": "Run the security audit on the project", "relevant_tools": ["run_terminal_cmd"]},
    {"prompt": "List all directories in the src folder", "relevant_tools": ["list_dir"]},
    {"prompt": "Show me the requirements.txt file", "relevant_tools": ["read_file"]},
]

# Load existing prompts
prompts_path = Path("data/raw/prompts.json")
with open(prompts_path) as f:
    existing = json.load(f)

# Merge - avoid duplicates
existing_texts = {p["prompt"] for p in existing}
added = 0
for p in extra_prompts:
    if p["prompt"] not in existing_texts:
        existing.append(p)
        existing_texts.add(p["prompt"])
        added += 1

with open(prompts_path, "w") as f:
    json.dump(existing, f, indent=2)

print(f"  Added {added} new prompts. Total: {len(existing)} prompts")

print("\nStep 2: Regenerating training data...")
import subprocess
result = subprocess.run(["python", "src/data_generator.py"], capture_output=True, text=True)
print(result.stdout[-500:] if result.stdout else "")
if result.returncode != 0:
    print("ERROR:", result.stderr[-300:])

print("\nStep 3: Deleting old model...")
shutil.rmtree("models", ignore_errors=True)
Path("models").mkdir(exist_ok=True)
print("  Done")

print("\nStep 4: Retraining with more data and better thresholds...")
result = subprocess.run(["python", "src/train.py"], capture_output=False, text=True)
if result.returncode != 0:
    print("Training failed")
else:
    print("Training complete")

print("\nStep 5: Overriding thresholds with safer values...")
meta_path = Path("models/metadata.json")
with open(meta_path) as f:
    meta = json.load(f)

# Override with safer thresholds - prevents false positives
safer_thresholds = {
    "grep_search":      0.50,
    "codebase_search":  0.50,
    "read_file":        0.50,
    "list_dir":         0.50,
    "run_terminal_cmd": 0.50,
    "edit_file":        0.50,
    "web_search":       0.50,
}

meta["thresholds"] = safer_thresholds
meta["threshold_override"] = "manually set to 0.50 after retraining with expanded dataset"

with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)

# Also update the joblib artifact thresholds
import joblib
artifact_path = Path("models/tool_selector_pipeline.joblib")
artifact = joblib.load(artifact_path)
artifact["thresholds"] = safer_thresholds
joblib.dump(artifact, artifact_path)

print("  Thresholds set to 0.50 for all tools")

print("\nStep 6: Verifying fix...")
result = subprocess.run(
    ["python", "src/predict.py", "--prompt",
     "Find all functions that call authenticate_user()"],
    capture_output=True, text=True
)
print(result.stdout)

print("\nDone! Run these to verify:")
print('  python src/predict.py --prompt "Find all functions that call authenticate_user()"')
print('  python src/predict.py --prompt "What files are in the src directory?"')
print('  python src/predict.py --prompt "Refactor UserService and run the tests"')
print("  python -m pytest tests/ -v")