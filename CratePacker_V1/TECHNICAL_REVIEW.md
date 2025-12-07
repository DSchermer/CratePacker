# Technical Review Checklist

This document helps engineering reviewers evaluate the Crate Packing Planner before approving the solution for deployment or hand-off.

## 1. Architecture Overview

- Streamlit front-end (`app.py`) orchestrates data intake, packing logic, visualization, and downloads.
- Packing engine uses `rectpack` (SkylineMWFL algorithm) to place buffered item footprints in crate/box catalogs.
- Catalogs stored as YAML (`crates_catalog.yaml`, `boxes_catalog.yaml`) with dimensions, costs, and priorities.
- Cost optimizer enumerates permutations of allowed crate types (up to six) to minimize total cost.
- Visuals rendered via Matplotlib and exported as PNG for each packed container.

## 2. Data Flow

1. User chooses catalog, overrides, and buffer settings in Streamlit sidebar.
2. Items table populated from uploaded CSV or inline edits.
3. `pack_items` expands counts and sorts items by footprint.
4. `run_packing_plan` iterates crate permutations, packs each scenario with `rectpack`, repacks for cost optimization, and records metrics.
5. Results shown live in Streamlit and saved under `output/pack_<timestamp>/`.

## 3. Key Modules & Functions (see `app.py`)

- `load_catalog` / `load_default_items`: cached IO helpers.
- `sanitise_items`, `validate_and_expand_items`: user data cleaning.
- `pack_items`: main packing loop.
- `run_packing_plan`: orchestrates optimization, generates `plan_results` structure.
- `render_bin_image`: Matplotlib diagram generator using buffered vs base rectangles.
- `persist_outputs`: writes JSON, CSV, PNG, and ZIP artifacts.

## 4. Algorithmic Considerations

- Packing Mode: `PackingMode.Offline` with `BinManager` to allow multiple bins of each crate type.
- Sorting heuristic prioritizes largest buffered dimension, then area, then label.
- Cost search enumerates permutations for up to six crate types; scale the limit carefully if catalog grows.
- Repacking: after initial pack, attempts to reassign each bin to cheaper crates without violating fit constraints.

## 5. Performance Notes

- Suitable for dozens of items; runtime increases factorially with allowed crate permutations. Document maximum expected load.
- Streamlit session caches to prevent repeated catalog reads.
- PNG generation uses Matplotlib Agg backend; ensure headless environments have fonts available.

## 6. Dependency Review

See `requirements.txt` (Streamlit, rectpack, pandas, numpy, matplotlib, pyyaml, pillow).
- Verify license compatibility for deployment.
- Confirm pinned minimum versions align with target environment.

## 7. Configuration & Customization

- Catalogs editable via YAML; changing cost/priority takes effect on next pack run.
- App defaults to inches; units label adjustable in sidebar.
- Buffer percentage configured per session.

## 8. Output Artifacts

- `summary.json`: structured summary of packing plan.
- `crates.csv` or `boxes.csv`: selected container mix.
- PNG renders for each container with annotations.
- `session.zip`: combined export.

## 9. Quality & Testing

- Manual test plan: smoke test with `examples/items_small.csv`, verify pack output and downloads.
- Suggested automated coverage: unit tests for item sanitization, packing heuristics, cost optimizer permutations.
- Linting/formatting: no current tooling; consider `black`, `ruff`, or `flake8` for consistency.

## 10. Security & Privacy

- Application stores all data locally; no network calls beyond Streamlit.
- Ensure shared machines clear `output/` directory if sensitive item data is used.

## 11. Deployment Considerations

- Designed for local execution. For shared deployment, plan for a managed Streamlit hosting environment or internal server.
- Document how to refresh catalogs and redeploy when crate dimensions change.

## 12. Open Questions

- Do we need automated regression tests for new catalog entries?
- Should cost overrides persist across sessions (currently per session only)?
- Are there requirements for metric (cm/mm) support?

Reviewers should sign off once they confirm each section aligns with organizational standards and risk tolerance.