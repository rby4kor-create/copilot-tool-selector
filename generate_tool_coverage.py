"""
generate_tool_coverage.py
Systematically generates enough training prompts for every tool that has
thin coverage (<8 examples), using the tool's own real metadata
(description, capabilities, category, operation_type) as the source --
not invented facts about what the tool does, just varied natural-language
phrasings of what's already documented.

This is synthetic data generation, consistent with this repo's own
config (training_source: synthetic) -- it augments, it doesn't replace,
the existing hand-authored/ported prompts.
"""
import json
import random
from collections import Counter

random.seed(42)

READ_TEMPLATES = [
    "{desc}",
    "Can you {desc_lc}?",
    "Show me {noun}",
    "What is {noun}?",
    "I need to see {noun}",
    "Look up {noun} for this repo",
    "Check {noun}",
    "Display {noun}",
]
LIST_TEMPLATES = [
    "{desc}",
    "List {noun}",
    "Show all {noun}",
    "What {noun} exist?",
    "Give me every {noun_singular}",
]
CREATE_TEMPLATES = [
    "{desc}",
    "Create {noun}",
    "Add {noun}",
    "Set up {noun}",
    "I want to add {noun}",
]
UPDATE_TEMPLATES = [
    "{desc}",
    "Update {noun}",
    "Change {noun}",
    "Modify {noun}",
    "Edit {noun}",
]
DELETE_TEMPLATES = [
    "{desc}",
    "Delete {noun}",
    "Remove {noun}",
    "Get rid of {noun}",
]
RUN_TEMPLATES = [
    "{desc}",
    "Run {noun}",
    "Trigger {noun}",
    "Execute {noun}",
    "Kick off {noun}",
]
SEARCH_TEMPLATES = [
    "{desc}",
    "Search for {noun}",
    "Find {noun}",
    "Look for {noun}",
]
WRITE_GENERIC_TEMPLATES = [
    "{desc}",
    "Write {noun}",
    "Add {noun}",
    "Post {noun}",
    "Submit {noun}",
]

GENERIC_WORDS = {"read", "get", "list", "operation", "single item lookup", "write",
                  "bulk listing", "create", "update", "delete"}


def build_noun_phrase(tool, variant=0):
    caps = [c for c in tool.get("capabilities", []) if c.lower() not in GENERIC_WORDS and len(c) > 2]
    if not caps:
        return tool["category"].replace("_", " ")
    random.Random(variant * 97 + len(caps)).shuffle(caps)
    n = 1 + (variant % min(3, len(caps)))
    return " ".join(caps[:n])


def pick_templates(tool_name, op_type, category):
    name = tool_name.lower()
    if name.startswith("list") or category in ("repository_stars",) and "list" in name:
        return LIST_TEMPLATES
    if name.startswith("create") or name.startswith("fork"):
        return CREATE_TEMPLATES
    if name.startswith("update") or name.startswith("edit"):
        return UPDATE_TEMPLATES
    if name.startswith("delete") or name.startswith("dismiss") or name.startswith("unstar"):
        return DELETE_TEMPLATES
    if name.startswith("search"):
        return SEARCH_TEMPLATES
    if "trigger" in name or name.startswith("run") or name.startswith("merge"):
        return RUN_TEMPLATES
    if name.endswith("_write") or name.startswith("add") or name.startswith("push") or name.startswith("star"):
        return WRITE_GENERIC_TEMPLATES
    if op_type == "read":
        return READ_TEMPLATES
    return WRITE_GENERIC_TEMPLATES


DESC_FALLBACK_TEMPLATES = [
    "{desc}.",
    "Could you {desc_lc}?",
    "I'd like to {desc_lc}.",
    "Please {desc_lc}.",
    "Can you help me {desc_lc}?",
    "I need you to {desc_lc}.",
    "Would you {desc_lc}?",
    "Go ahead and {desc_lc}.",
    "For this repo, {desc_lc}.",
    "Right now I need to {desc_lc}.",
]


def generate_for_tool(tool_name, tool, n_needed):
    desc = tool["description"].rstrip(".")
    desc_lc = desc[0].lower() + desc[1:] if desc else desc

    templates = pick_templates(tool_name, tool.get("operation_type", "read"), tool.get("category", ""))

    out = []
    local_seen = set()
    attempt = 0
    while len(out) < n_needed and attempt < len(templates) * 6:
        t = templates[attempt % len(templates)]
        noun = build_noun_phrase(tool, variant=attempt)
        noun_singular = noun.split()[0] if noun else tool["category"]
        text = t.format(desc=desc, desc_lc=desc_lc, noun=noun, noun_singular=noun_singular)
        text = text[0].upper() + text[1:]
        if not text.endswith(("?", ".")):
            text += "."
        if text not in local_seen:
            local_seen.add(text)
            out.append(text)
        attempt += 1

    # Fallback layer: description-based phrasings, unique per tool
    # regardless of how few distinctive capability words it has.
    fi = 0
    while len(out) < n_needed and fi < len(DESC_FALLBACK_TEMPLATES):
        t = DESC_FALLBACK_TEMPLATES[fi]
        text = t.format(desc=desc, desc_lc=desc_lc)
        text = text[0].upper() + text[1:]
        if text not in local_seen:
            local_seen.add(text)
            out.append(text)
        fi += 1

    return out


def main():
    prompts = json.load(open("data/raw/prompts.json"))
    catalog = json.load(open("data/processed/tool_catalog.json"))
    catalog.pop("__meta__", None)

    counts = Counter()
    for p in prompts:
        for t in p["relevant_tools"]:
            counts[t] += 1

    TARGET = 10
    new_prompts = []
    seen_text = set(p["prompt"] for p in prompts)

    for tool_name, tool in catalog.items():
        have = counts.get(tool_name, 0)
        if have >= TARGET:
            continue
        need = TARGET - have
        generated = generate_for_tool(tool_name, tool, need + 10)  # generous margin to survive dedup
        added = 0
        for text in generated:
            if text in seen_text:
                continue
            seen_text.add(text)
            new_prompts.append({"prompt": text, "relevant_tools": [tool_name]})
            added += 1
            if added >= need:
                break

    print(f"Generated {len(new_prompts)} new prompts across {sum(1 for t in catalog if counts.get(t,0) < TARGET)} thin tools")

    merged = prompts + new_prompts
    json.dump(merged, open("data/raw/prompts.json", "w"), indent=2)
    print(f"Total prompts now: {len(merged)}")


if __name__ == "__main__":
    main()
