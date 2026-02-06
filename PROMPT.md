# Crate Packing Planner — Project Prompt

Build a browser-based tool that helps logistics teams choose the cheapest combination of standardized shipping containers to pack a list of rectangular items, and visualize the resulting layouts. No backend server; the app runs locally in a single process.

Use **Streamlit** for the UI, **rectpack** for 2D rectangle bin-packing, **matplotlib** for rendering layout diagrams, and **pandas** for tabular data handling. Container catalogs are defined in **YAML** files.

---

## Core Problem

Given:
- A list of rectangular items, each with a label, length, width, and quantity.
- A catalog of available container types, each with an ID, interior dimensions, an optional unit cost, and an optional priority (lower = preferred).
- A uniform buffer value that represents required clearance around every item (applied as +2x buffer to both length and width before packing).

Find the combination of containers that:
1. Fits every item (with buffer applied), allowing 90-degree rotation.
2. Minimizes total container cost.
3. Among equal-cost solutions, uses the fewest containers.
4. Among ties, maximizes space utilization.

Then render a color-coded top-down diagram of each packed container showing item placement.

---

## Optimization Approach

The solver should work in two phases:

### Phase 1 — Permutation Search

Because the order in which container types are offered to a greedy packer changes the outcome, enumerate all orderings (permutations) of the selected container types. Cap this at six types (720 permutations) to keep runtime practical; refuse with a clear message if exceeded.

For each permutation, greedily pack items by walking the container sequence in order:
- For the current container type, filter to items whose buffered dimensions fit (considering rotation).
- Hand those items to rectpack with enough bins of that size to accommodate them all. Use an offline, deterministic packing algorithm (e.g., Skyline with Best Fit First bin selection) with rotation enabled.
- Remove successfully placed items from the pool and move to the next container type.
- If any items remain unplaced after exhausting all types, discard this permutation.

Score each successful permutation as a tuple: (total cost, container count, -utilization). The lowest-scoring permutation wins.

Pre-sort items before packing by descending largest dimension, then descending area, then label alphabetically. This reduces fragmentation and makes results deterministic.

### Phase 2 — Per-Bin Cost Reduction

After selecting the best permutation, iterate over each packed bin individually and attempt to repack its contents into every cheaper (or same-cost, smaller-area) container type. Accept the cheapest container that fits all of the bin's items in a single unit. This is a local optimization — it does not move items between bins.

---

## Container Catalogs

Define catalogs as YAML files with a version string and a list of entries. Each entry has an `id`, `length`, `width`, and optional `cost` and `priority` fields. Ship at least two catalog files (e.g., one for large crates, one for small boxes) and let the user switch between them.

When building the working list of container types, sort by priority (ascending), then by the user's selection order, then by ID. Allow the user to override any container's unit cost for the current session without modifying the YAML.

---

## User Interface

### Sidebar
- Radio button to switch between container catalogs.
- Numeric input for the buffer value.
- Option to restrict packing to a subset of the catalog's container types.
- Per-container cost override control.
- CSV upload for the item list (columns: label, length, width, count).
- Toggle to auto-aggregate rows with identical label and dimensions by summing counts.

### Main Area
- An editable data table for items with inline validation (positive dimensions, non-empty labels, integer counts).
- Preview metrics: unique row count, total item count, total buffered area, number of allowed container types.
- A "Pack" button that runs the solver.

### Results (shown after packing)
Organize into tabs:
- **Summary** — total containers used, overall utilization percentage, total cost, and a per-container-type breakdown table (count, average utilization, unit cost, subtotal).
- **Visuals** — one labeled diagram per packed container. Each diagram draws the container outline, each item as a colored rectangle at its placed position (outer rectangle = buffered size, inner rectangle = base size), with a label and rotation indicator. Include utilization in the title.
- **Tables** — per-container placement tables listing each item's ID, label, x/y position, placed dimensions, and whether it was rotated.
- **Downloads** — summary as JSON, all layout PNGs in a zip, and a full bundle zip containing everything.

---

## Output Artifacts

On each packing run, create a timestamped directory containing:
- `summary.json` — catalog version, buffer, units, total items, container counts, utilization metrics, cost breakdown, and the winning container ordering.
- Per-container PNG images in an `images/` subdirectory.
- A zip of the images folder.
- A session zip bundling all of the above.

---

## Behavior Details

- Items that cannot fit in any selected container type (after buffer) should be reported by name before packing begins.
- If no permutation can place all items, report which items could not be placed.
- All numeric display should suppress unnecessary decimals (e.g., show `12` not `12.0`).
- Rotation detection: compare placed dimensions to the item's buffered dimensions; if they don't match, the item was rotated.
- The app must be fully deterministic — identical inputs always produce identical outputs.

---

## Project Structure

- Single application file for all logic and UI.
- One YAML catalog file per container family.
- A `requirements.txt` listing: streamlit, rectpack, pandas, numpy, matplotlib, pyyaml, pillow.
- A README with setup and launch instructions.
- An `examples/` directory with a sample items CSV.
