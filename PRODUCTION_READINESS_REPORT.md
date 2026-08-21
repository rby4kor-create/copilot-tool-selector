# Production Readiness Report — 96-Tool Model

## What changed, and the real, measured before/after

| Metric | Before (naive merge, 409 prompts) | After (coverage fix, 1,039 prompts) |
|---|---|---|
| Tools | 96 | 96 |
| Training prompts | 409 | 1,039 |
| Tools with F1 = 0 (no working signal) | 66 of 96 | **33 of 96** |
| Tools with real signal (F1 > 0) | 30 of 96 | **63 of 96** |
| Tools never reaching test set | 49 of 96 | **10 of 96** |
| Exact-match accuracy | 19.5% | **56.25%** |
| Micro F1 | 0.4484 | **0.6891** |
| Tool reduction | 98.2% | **99.3%** |
| Training crash on `--algorithm compare` | Yes (SVM: <3 examples/class) | **No — fixed by having enough data** |
| Best algorithm | logistic_regression (SVM crashed) | **linear_svm** (auto-selected, beat LR) |

## What was actually done
1. **Diagnosed and quarantined 12 stale duplicate implementation files** (`legacy_unused/`) — two parallel, divergent pipelines existed; verified the canonical nested architecture (`src/catalog/`, `src/training/`, `src/inference/`, etc.) still works correctly after removing the stale ones.
2. **Merged the 89 real GitHub tools into the existing 7-tool catalog** — 96 tools total, zero name collisions, real tool metadata (no invented tools).
3. **Recovered 72 already-written, never-used training prompts** sitting dead in an unrun `fix_thresholds.py` script.
4. **Built `generate_tool_coverage.py`** — a template-based, metadata-driven prompt generator (uses each tool's real description/capabilities, not invented facts) to bring every tool up to at least 10 training examples. Found and fixed a real bug in the generator itself (internal duplicate cycling prevented it from reaching its own fallback templates) before trusting its output.
5. **Verified a suspected threshold-persistence bug does NOT reproduce** in the current canonical pipeline (the flat 0.5 thresholds seen in the original artifact came from a stale run, not a live code bug).
6. **Retrained end-to-end**, confirmed the SVM crash is resolved now that every tool has sufficient per-class examples.

## Honest remaining gaps — NOT fully production-ready yet
- **33 of 96 tools still have zero measurable signal.** These need real usage examples, not just more templated ones — templated phrasing has a ceiling.
- **New cross-catalog confusion found**: `list_dir` (an IDE tool) now incorrectly fires alongside correct GitHub tools on prompts like "Show me PR #142" and "List all open issues" — the two catalogs' vocabularies overlap in ways neither original dataset anticipated. This needs targeted adversarial examples (same pattern as the GitHub-only project's Phase 12-14 work), not yet done here.
- **10 tools still never appear in the test set** — too few examples to even measure them reliably.
- 1,039 prompts is templated-heavy (287 of them are synthetic, generated, not organically written) — real user prompt logs, once available, should replace/augment these for a genuinely production-grade dataset. The config's `manager_data_path` (a real-usage data source) is still unfilled.

## Recommended before actual production traffic
1. Run the same shadow-mode logging approach used on the GitHub-only project against real usage for 1-2 weeks.
2. Target the 33 remaining zero-signal tools with real (not just templated) examples.
3. Add adversarial pairs for the newly-found `list_dir` cross-catalog confusion.
4. Re-evaluate before considering this "production-ready" in the full sense — this round closed the biggest gap (data starvation) but did not solve every open issue.
