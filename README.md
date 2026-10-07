# agentic_verification
A framework that classifies autonomous AI agents by the Chomsky hierarchy, reduces unverifiable agents into verifiable ones through architectural interventions, and backs the whole thing with an epistemically self-aware knowledge base that knows what it does not know

## Contents

| Path | What it is |
|---|---|
| `horizon-probe/` | Boundary analysis of where models diverge on the same question, with "legal" and "banned" as two separate axes. It has a 38-statement item bank built from the Gemma 4 Developer Agent forum and notebooks, and a recursive root question. Total surprise is split into within-model noise and between-model divergence (Bayes factors, per-model surprise). A stacked-RBM deep belief network and repo-local SWI-Prolog rules compute horizons, size pivots and universal definitions. Includes a CPU smoke run on Gemma 4 E4B and E2B. |
| `horizon-probe/data/kaggle_public_extract.{json,md}` | Rules, syntax, errors, task and test facts, allow/ban vocabulary and disputed questions extracted from the competition's public forum threads and notebooks, with sources |
| `kaggriculture/` | Two definitions of "legal", "wants", "shared borders" and "mutual success" for neighboring autonomous farmers, the $80 border underwriting as a Nash bargaining split, and an interactive farm-grid page |
