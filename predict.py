"""
predict.py
CLI entry point for production inference.

Usage:
    python predict.py --prompt "Find all functions calling authenticate_user()"
    python predict.py --prompt "..." --json
    python predict.py --prompt "..." --explain
"""
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
            print(f"\nPrompt  : {result['prompt']}")
            print(f"Model   : v{result['model_version']} ({result['algorithm']})")
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
