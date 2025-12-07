Build a Streamlit app that uses rectpack for 2D crate packing with standardized crate types and variable assemblies, applying a uniform buffer, and exporting per-crate visualizations that clearly show the crate outline and buffered rectangles. the project should be well organized.

Key choices
- Library: rectpack (guillotine/skyline packers) for 2D rectangle packing with rotation support. Use a deterministic, stable packing variant (e.g., Skyline or Guillotine with consistent heuristics). 
- App: Streamlit single-file for minimal setup; no separate backend. 
- Goal: Minimize number of crates used; within that, maximize fill (via packing order and crate selection). Exact lexicographic optimality is not guaranteed by rectpack; this is an engineering tradeoff for simplicity and speed.

Assumptions
- 2D only; no stacking; allow 90° rotations of items.
- Uniform buffer: inflate each assembly by 2×buffer on both axes before packing to enforce inter-item and wall clearance.
- Crate catalog is standardized and versioned (crates_catalog.yaml or embedded JSON).
- Assemblies list can be arbitrarily long; typical runs are a few dozen items.

Crate selection and priority
- allowed_crate_ids: user-selected subset of the catalog for the current job.
- Prefer lower numeric priority (1 is better than 5) when choosing which crate type to instantiate first; if no priority or equal, follow allowed_crate_ids order; then lexical crate id. Use this order when opening new bins.

Packing strategy with rectpack
- For each allowed crate type, create a packer with:
  - bin size = crate length × width
  - rotation allowed for items
  - a deterministic algorithm (e.g., Skyline-MW or Guillotine-CF with fixed heuristics)
- Expand items by buffer (L’ = L + 2×buffer, W’ = W + 2×buffer) and add to a unified queue sorted by descending max(L’,W’) then area to reduce fragmentation.
- Iteratively pack items:
  - Try packing into existing bins ordered by preference (priority → allowed order → id).
  - If an item does not fit any existing bin, open a new bin of the best-preference crate type that can fit the item and retry.
- Continue until all items are placed; report crate usage distribution.

Visualizations
- For each crate/bin, render a labeled top-down drawing showing:
  - Crate outline as a rectangle border.
  - Each buffered item as a filled rectangle with label and orientation annotation.
  - Optional hatching or distinct stroke color to visually emphasize buffer is included in rectangle size.
- Output both on-screen (Streamlit) and as saved files in an output directory (PNG/SVG).

Streamlit UX
- Sidebar:
  - Buffer (≥0), Units (string), Allowed Crate IDs (multiselect from catalog).
  - CSV uploader (columns: label,length,width,count).
  - Toggle “Auto-aggregate identical rows” (by trimmed label, length, width; case-sensitive labels by default).
- Main:
  - Editable items table with validation.
  - Preview panel: unique rows, total count, total buffered area, catalog version.
  - “Pack” button; show status, bin counts, per-crate-type usage and utilization.
  - Results tabs: Summary, Visuals (per-crate), Tables (placements), Downloads (summary.json, placements_{crate}.csv, images zip).

Data contracts
- Catalog (crates_catalog.yaml):
  {version: string, crates: [{id: string, length: number>0, width: number>0, priority?: int}]}
- Items CSV: label,length,width,count
- Outputs:
  - summary.json with crate counts, utilization by crate, overall utilization, catalog version.
  - placements_{crateId}_{index}.csv with: item_id,label,x,y,length,width,rotated.
  - images/{crateId}_{index}.png (or .svg).

Implementation notes
- Buffer handling: change item dims to L’=L+2×buffer, W’=W+2×buffer, so drawings inherently show buffer; annotate chart legend “dimensions include buffer.”
- Coordinate system: use rectpack’s placements directly; origin at crate bottom-left; draw rectangles using matplotlib (or svgwrite) with consistent scaling and axis equal.
- Rotation: enable in rectpack; reflect in output as rotated=true/false.
- Ordering: to reduce bin count, sort items by decreasing max side then area; iterate crate types in preference order when opening new bins.
- Determinism: use a fixed item order, fixed crate opening order, and a rectpack algorithm with stable heuristics to yield consistent results for the same inputs.

Project layout
- app.py (Streamlit UI + rectpack packing + visualization)
- crates_catalog.yaml (standardized crates, with version)
- examples/items_small.csv
- requirements.txt (streamlit, rectpack, pandas, numpy, matplotlib, pyyaml, pillow or svgwrite)
- README.md with run instructions (pip install -r requirements.txt; streamlit run app.py)

Acceptance criteria
- Single-command run, no extra services.
- All items placed respecting buffer and crate walls.
- Minimizes crate count in practice; uses priority and allowed order to choose crate types when opening bins.
- Per-crate visualization clearly shows crate outline and buffered rectangles with labels/orientation.
- Repeatable outputs given identical inputs (deterministic algorithm and ordering).
