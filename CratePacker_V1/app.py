"""Streamlit app for 2D crate packing with rectpack."""
from __future__ import annotations

import copy
import io
import itertools
import json
import math
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import streamlit as st
import yaml
import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from rectpack import PackingBin, PackingMode, newPacker
import rectpack

# Paths relative to this file
BASE_DIR = Path(__file__).parent
DEFAULT_ITEMS_PATH = BASE_DIR / "examples" / "items_small.csv"
OUTPUT_ROOT = BASE_DIR / "output"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

DEFAULT_UNITS = "in"

CATALOG_SOURCES = {
    "Crates": {
        "path": BASE_DIR / "crates_catalog.yaml",
        "list_key": "crates",
        "singular": "crate",
        "plural": "crates",
    },
    "Boxes": {
        "path": BASE_DIR / "boxes_catalog.yaml",
        "list_key": "boxes",
        "singular": "box",
        "plural": "boxes",
    },
}

# Streamlit page setup
st.set_page_config(
    page_title="Crate Packing Planner",
    layout="wide",
)

st.title("Crate Packing Planner")
st.caption(
    "Buffered dimensions are used for packing and visualizations; drawings include the buffer margin. "
    "All dimensions are in inches."
)


@dataclass
class CrateType:
    crate_id: str
    length: float
    width: float
    priority: int
    cost: float | None = None


@dataclass
class PackedItem:
    item_id: str
    label: str
    base_length: float
    base_width: float
    buffered_length: float
    buffered_width: float


@st.cache_data(show_spinner=False)
def load_catalog(path_str: str, list_key: str, cache_buster: float) -> Dict:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Catalog not found at {path}")
    _ = cache_buster  # ensure cache invalidates when file timestamp changes
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    entries = data.get(list_key, [])
    if not entries:
        raise ValueError(f"Catalog must define at least one entry under '{list_key}'")
    return data


@st.cache_data(show_spinner=False)
def load_default_items(path_str: str, cache_buster: float) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame(columns=["label", "length", "width", "count"])
    _ = cache_buster
    df = pd.read_csv(path)
    expected_cols = ["label", "length", "width", "count"]
    missing = [col for col in expected_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Sample CSV missing columns: {', '.join(missing)}")
    return df[expected_cols]


def normalise_label(value: str) -> str:
    return str(value).strip()


def sanitise_items(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    records: List[Dict] = []
    issues: List[str] = []
    if df is None or df.empty:
        return pd.DataFrame(columns=["label", "length", "width", "count"]), []

    for idx, row in df.iterrows():
        label = normalise_label(row.get("label", ""))
        if not label:
            issues.append(f"Row {idx + 1}: label is required")
            continue

        try:
            length = float(row.get("length"))
        except (TypeError, ValueError):
            issues.append(f"Row {idx + 1}: length must be a number")
            continue

        try:
            width = float(row.get("width"))
        except (TypeError, ValueError):
            issues.append(f"Row {idx + 1}: width must be a number")
            continue

        try:
            count_raw = row.get("count", 1)
            count = int(float(count_raw))
        except (TypeError, ValueError):
            issues.append(f"Row {idx + 1}: count must be an integer")
            continue

        if length <= 0 or width <= 0:
            issues.append(f"Row {idx + 1}: length and width must be > 0")
            continue
        if count <= 0:
            issues.append(f"Row {idx + 1}: count must be > 0")
            continue

        records.append({
            "label": label,
            "length": length,
            "width": width,
            "count": count,
        })

    clean_df = pd.DataFrame(records, columns=["label", "length", "width", "count"])
    return clean_df, issues


def aggregate_items(df: pd.DataFrame, enabled: bool) -> pd.DataFrame:
    if not enabled or df.empty:
        return df.copy()
    agg_df = (
        df.groupby(["label", "length", "width"], as_index=False, sort=False)["count"].sum()
    )
    return agg_df


def normalize_numeric_display(value):
    if pd.isna(value):
        return value
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    if math.isclose(numeric, round(numeric), rel_tol=1e-12, abs_tol=1e-9):
        return int(round(numeric))
    return numeric


def compute_preview_metrics(df: pd.DataFrame, buffer: float) -> Dict:
    if df.empty:
        return {
            "unique_rows": 0,
            "total_count": 0,
            "total_buffered_area": 0.0,
        }
    total_count = int(df["count"].sum())
    buffered_area = 0.0
    for _, row in df.iterrows():
        buffered_length = row["length"] + 2 * buffer
        buffered_width = row["width"] + 2 * buffer
        buffered_area += buffered_length * buffered_width * row["count"]
    return {
        "unique_rows": len(df),
        "total_count": total_count,
        "total_buffered_area": buffered_area,
    }


def build_crate_objects(
    catalog_entries: Iterable[Dict],
    allowed_ids: Iterable[str],
    cost_overrides: Dict[str, float] | None = None,
) -> List[CrateType]:
    crates: List[CrateType] = []
    allowed_index = {crate_id: idx for idx, crate_id in enumerate(allowed_ids)}
    for crate in catalog_entries:
        crate_id = crate.get("id")
        if crate_id not in allowed_index:
            continue
        priority_val = crate.get("priority")
        if priority_val is None:
            priority = 10_000_000
        else:
            priority = int(priority_val)
        cost_val = crate.get("cost")
        if cost_overrides and crate_id in cost_overrides:
            cost = float(cost_overrides[crate_id])
        else:
            cost = float(cost_val) if cost_val is not None else None
        crates.append(
            CrateType(
                crate_id=crate_id,
                length=float(crate.get("length")),
                width=float(crate.get("width")),
                priority=priority,
                cost=cost,
            )
        )
    crates.sort(key=lambda c: (c.priority, allowed_index[c.crate_id], c.crate_id))
    return crates


def expand_items(df: pd.DataFrame, buffer: float) -> Tuple[List[PackedItem], Dict[str, PackedItem]]:
    expanded: List[PackedItem] = []
    lookup: Dict[str, PackedItem] = {}
    serial = 0
    for _, row in df.iterrows():
        for n in range(int(row["count"])):
            item_id = f"item_{serial:05d}"
            serial += 1
            base_length = float(row["length"])
            base_width = float(row["width"])
            buffered_length = base_length + 2 * buffer
            buffered_width = base_width + 2 * buffer
            item = PackedItem(
                item_id=item_id,
                label=row["label"],
                base_length=base_length,
                base_width=base_width,
                buffered_length=buffered_length,
                buffered_width=buffered_width,
            )
            expanded.append(item)
            lookup[item_id] = item
    expanded.sort(
        key=lambda item: (
            -max(item.buffered_length, item.buffered_width),
            -(item.buffered_length * item.buffered_width),
            item.label,
        )
    )
    return expanded, lookup


def ensure_items_fit(items: List[PackedItem], crates: List[CrateType]) -> List[str]:
    oversized: List[str] = []
    for item in items:
        fits = any(
            (item.buffered_length <= crate.length and item.buffered_width <= crate.width)
            or (item.buffered_length <= crate.width and item.buffered_width <= crate.length)
            for crate in crates
        )
        if not fits:
            oversized.append(f"{item.label} ({item.base_length}×{item.base_width})")
    return oversized


def render_bin_image(bin_data: Dict, units: str, output_path: Path) -> bytes:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlim(0, bin_data["crate_length"])
    ax.set_ylim(0, bin_data["crate_width"])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(f"Length ({units})")
    ax.set_ylabel(f"Width ({units})")
    ax.set_title(f"{bin_data['crate_id']} #{bin_data['crate_index']} | Utilization {bin_data['utilization'] * 100:.1f}%")

    color_count = max(1, len(bin_data["items"]))
    colors = matplotlib.colormaps.get_cmap("tab20").resampled(color_count)
    for idx, item in enumerate(bin_data["items"]):
        facecolor = colors(idx)

        outer_rect = Rectangle(
            (item["x"], item["y"]),
            item["placed_length"],
            item["placed_width"],
            linewidth=0.6,
            edgecolor="black",
            facecolor=facecolor,
            alpha=0.35,
        )
        ax.add_patch(outer_rect)

        if item["rotated"]:
            base_length = item["base_width"]
            base_width = item["base_length"]
        else:
            base_length = item["base_length"]
            base_width = item["base_width"]

        inset_x = item["x"] + max(0.0, (item["placed_length"] - base_length) / 2)
        inset_y = item["y"] + max(0.0, (item["placed_width"] - base_width) / 2)

        inner_rect = Rectangle(
            (inset_x, inset_y),
            max(base_length, 0.0),
            max(base_width, 0.0),
            linewidth=0.9,
            edgecolor="black",
            facecolor=facecolor,
            alpha=0.85,
        )
        ax.add_patch(inner_rect)

        label_text = item["label"]
        if item["rotated"]:
            label_text += " (R)"
        ax.text(
            inset_x + max(base_length, 0.0) / 2,
            inset_y + max(base_width, 0.0) / 2,
            label_text,
            ha="center",
            va="center",
            fontsize=8,
            color="black",
            wrap=True,
        )

    outline = Rectangle((0, 0), bin_data["crate_length"], bin_data["crate_width"], fill=False, edgecolor="black", linewidth=2.2)
    ax.add_patch(outline)
    ax.grid(False)

    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    buffer.seek(0)
    output_path.write_bytes(buffer.getvalue())
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


def try_repack_bin(bin_items: List[Dict], crate: CrateType) -> Dict | None:
    if not bin_items:
        return None

    packer = newPacker(
        mode=PackingMode.Offline,
        bin_algo=PackingBin.BFF,
        pack_algo=rectpack.SkylineMwfl,
        sort_algo=rectpack.SORT_NONE,
        rotation=True,
    )
    packer.add_bin(crate.length, crate.width, count=1, bid=crate.crate_id)

    source_by_id = {item["item_id"]: item for item in bin_items}
    for item in bin_items:
        packer.add_rect(item["buffered_length"], item["buffered_width"], rid=item["item_id"])

    packer.pack()

    rects = packer.rect_list()
    if len(rects) != len(bin_items):
        return None

    bin_indexes = {bin_index for bin_index, *_ in rects}
    if len(bin_indexes) != 1:
        return None

    new_items: List[Dict] = []
    for _, x, y, placed_length, placed_width, rid in rects:
        source = source_by_id.get(rid)
        if source is None:
            return None
        rotated = not (
            math.isclose(placed_length, source["buffered_length"], rel_tol=1e-4, abs_tol=1e-4)
            and math.isclose(placed_width, source["buffered_width"], rel_tol=1e-4, abs_tol=1e-4)
        )
        new_items.append({
            "item_id": rid,
            "label": source["label"],
            "x": float(x),
            "y": float(y),
            "placed_length": float(placed_length),
            "placed_width": float(placed_width),
            "rotated": rotated,
            "base_length": source["base_length"],
            "base_width": source["base_width"],
            "buffered_length": source["buffered_length"],
            "buffered_width": source["buffered_width"],
        })

    new_items.sort(key=lambda item: item["item_id"])

    return {
        "crate_id": crate.crate_id,
        "crate_length": crate.length,
        "crate_width": crate.width,
        "items": new_items,
    }


def optimise_bins_by_cost(bin_results: List[Dict], crates: List[CrateType]) -> List[Dict]:
    if not bin_results or not crates:
        return bin_results

    crates_by_id = {crate.crate_id: crate for crate in crates}
    crates_with_cost = [crate for crate in crates if crate.cost is not None]
    crates_with_cost.sort(
        key=lambda c: (
            c.cost,
            c.length * c.width,
            c.priority,
            c.crate_id,
        )
    )

    optimised: List[Dict] = []
    for bin_data in bin_results:
        source_items = copy.deepcopy(bin_data["items"])
        original_crate = crates_by_id.get(bin_data["crate_id"])
        best_bin = copy.deepcopy(bin_data)
        best_cost = original_crate.cost if original_crate and original_crate.cost is not None else float("inf")
        best_area = (original_crate.length * original_crate.width) if original_crate else float("inf")

        for crate in crates_with_cost:
            candidate = try_repack_bin(source_items, crate)
            if candidate is None:
                continue
            candidate_cost = crate.cost if crate.cost is not None else float("inf")
            candidate_area = crate.length * crate.width
            if (
                candidate_cost < best_cost - 1e-9
                or (
                    math.isclose(candidate_cost, best_cost, rel_tol=1e-9, abs_tol=1e-9)
                    and candidate_area < best_area - 1e-6
                )
            ):
                best_bin = candidate
                best_cost = candidate_cost
                best_area = candidate_area

        optimised.append(best_bin)

    return optimised


def pack_items(
    items_df: pd.DataFrame,
    crates: List[CrateType],
    buffer: float,
    units: str,
    catalog_version: str,
    container_family: str,
    container_singular: str,
    container_plural: str,
    cost_overrides: Dict[str, float] | None = None,
) -> Dict:
    if items_df.empty:
        raise ValueError("No valid items to pack.")
    if not crates:
        raise ValueError("No crate types selected.")

    items_expanded, lookup = expand_items(items_df, buffer)
    oversized = ensure_items_fit(items_expanded, crates)
    if oversized:
        summary_counts: Dict[str, int] = {}
        for entry in oversized:
            summary_counts[entry] = summary_counts.get(entry, 0) + 1
        formatted = [
            f"{item} ×{count}" if count > 1 else item
            for item, count in sorted(summary_counts.items())
        ]
        raise ValueError(
            "The following items do not fit in any allowed crate (buffer applied): "
            + "; ".join(formatted)
        )

    total_buffered_area = sum(item.buffered_length * item.buffered_width for item in items_expanded)

    crate_count = len(crates)
    permutation_limit = 720  # 6! combinations keeps evaluation tractable
    if crate_count > 6:
        raise ValueError(
            "Too many container types selected to guarantee minimum cost. "
            "Select six or fewer container styles to evaluate all configurations."
        )

    total_permutations = math.factorial(crate_count)
    if total_permutations > permutation_limit:
        raise ValueError(
            "Too many container types selected to guarantee minimum cost. "
            "Select six or fewer container styles to evaluate all configurations."
        )

    crate_orders = list(itertools.permutations(crates, crate_count))
    crates_by_id = {crate.crate_id: crate for crate in crates}

    def pack_for_order(crate_order: Tuple[CrateType, ...]):
        remaining_items = list(items_expanded)
        bin_results_local: List[Dict] = []
        crate_sequence_local: Dict[str, int] = {}

        for crate in crate_order:
            if not remaining_items:
                break

            fitting_items = [
                item
                for item in remaining_items
                if (
                    (item.buffered_length <= crate.length and item.buffered_width <= crate.width)
                    or (item.buffered_length <= crate.width and item.buffered_width <= crate.length)
                )
            ]

            if not fitting_items:
                continue

            packer = newPacker(
                mode=PackingMode.Offline,
                bin_algo=PackingBin.BFF,
                pack_algo=rectpack.SkylineMwfl,
                sort_algo=rectpack.SORT_NONE,
                rotation=True,
            )

            bin_capacity = max(1, len(fitting_items))
            packer.add_bin(crate.length, crate.width, count=bin_capacity, bid=crate.crate_id)

            for item in fitting_items:
                packer.add_rect(item.buffered_length, item.buffered_width, rid=item.item_id)

            packer.pack()

            placed_rects = packer.rect_list()
            if not placed_rects:
                continue

            placed_ids = {rid for _, _, _, _, _, rid in placed_rects}
            if not placed_ids:
                continue

            crate_sequence_local.setdefault(crate.crate_id, 0)

            local_bins: Dict[int, Dict] = {}
            for bin_index, x, y, placed_length, placed_width, rid in placed_rects:
                bin_obj = packer[bin_index]
                local_key = bin_index
                if local_key not in local_bins:
                    crate_sequence_local[crate.crate_id] += 1
                    local_bins[local_key] = {
                        "crate_id": crate.crate_id,
                        "crate_index": crate_sequence_local[crate.crate_id],
                        "crate_length": float(bin_obj.width),
                        "crate_width": float(bin_obj.height),
                        "items": [],
                    }

                packed_item = lookup[rid]
                rotated = not (
                    math.isclose(placed_length, packed_item.buffered_length, rel_tol=1e-4, abs_tol=1e-4)
                    and math.isclose(placed_width, packed_item.buffered_width, rel_tol=1e-4, abs_tol=1e-4)
                )

                local_bins[local_key]["items"].append({
                    "item_id": rid,
                    "label": packed_item.label,
                    "x": float(x),
                    "y": float(y),
                    "placed_length": float(placed_length),
                    "placed_width": float(placed_width),
                    "rotated": rotated,
                    "base_length": packed_item.base_length,
                    "base_width": packed_item.base_width,
                    "buffered_length": packed_item.buffered_length,
                    "buffered_width": packed_item.buffered_width,
                })

            for local_key in sorted(local_bins.keys()):
                bin_results_local.append(local_bins[local_key])

            remaining_items = [item for item in remaining_items if item.item_id not in placed_ids]

        if remaining_items:
            return None

        return bin_results_local, crate_sequence_local

    def compute_solution_metrics(bin_results_local: List[Dict], crate_sequence_local: Dict[str, int]):
        total_used_area_local = 0.0
        total_crate_area_local = 0.0

        for bin_data in bin_results_local:
            crate_area = bin_data["crate_length"] * bin_data["crate_width"]
            used_area = sum(item["placed_length"] * item["placed_width"] for item in bin_data["items"])
            bin_data["crate_area"] = crate_area
            bin_data["used_area"] = used_area
            bin_data["utilization"] = used_area / crate_area if crate_area else 0.0
            total_used_area_local += used_area
            total_crate_area_local += crate_area

        crate_usage_summary_local: List[Dict] = []
        total_container_cost_local = 0.0
        has_cost_data_local = False
        for crate_id, _count in sorted(
            crate_sequence_local.items(),
            key=lambda item: (
                crates_by_id[item[0]].cost if crates_by_id[item[0]].cost is not None else float("inf"),
                crates_by_id[item[0]].priority,
                item[0],
            ),
        ):
            crate = crates_by_id[crate_id]
            bins_for_crate = [bin_data for bin_data in bin_results_local if bin_data["crate_id"] == crate_id]
            used_area = sum(bin_data["used_area"] for bin_data in bins_for_crate)
            crate_area = crate.length * crate.width
            avg_utilization = used_area / (crate_area * len(bins_for_crate)) if crate_area and bins_for_crate else 0.0
            unit_cost = crate.cost if crate.cost is not None else None
            total_cost_val = unit_cost * len(bins_for_crate) if unit_cost is not None else None
            if total_cost_val is not None:
                total_container_cost_local += total_cost_val
                has_cost_data_local = True
            crate_usage_summary_local.append({
                "crate_id": crate_id,
                "priority": None if crate.priority >= 10_000_000 else crate.priority,
                "count": len(bins_for_crate),
                "length": crate.length,
                "width": crate.width,
                "avg_utilization": avg_utilization,
                "total_used_area": used_area,
                "unit_cost": unit_cost,
                "total_cost": total_cost_val,
            })

        overall_utilization_local = (
            total_used_area_local / total_crate_area_local if total_crate_area_local else 0.0
        )

        return {
            "bin_results": bin_results_local,
            "crate_sequence": crate_sequence_local,
            "crate_usage_summary": crate_usage_summary_local,
            "total_container_cost": total_container_cost_local if has_cost_data_local else None,
            "has_cost_data": has_cost_data_local,
            "overall_utilization": overall_utilization_local,
            "total_used_area": total_used_area_local,
            "total_crate_area": total_crate_area_local,
            "total_containers": len(bin_results_local),
        }

    best_solution: Dict | None = None
    best_score: Tuple[float, int, float] | None = None
    failed_orders = 0

    for crate_order in crate_orders:
        packed = pack_for_order(crate_order)
        if packed is None:
            failed_orders += 1
            continue

        bin_results_local, crate_sequence_local = packed
        metrics_local = compute_solution_metrics(bin_results_local, crate_sequence_local)

        total_cost_value = metrics_local["total_container_cost"]
        cost_sort_value = total_cost_value if total_cost_value is not None else float("inf")
        score = (
            cost_sort_value,
            metrics_local["total_containers"],
            -metrics_local["overall_utilization"],
        )

        if best_score is None or score < best_score:
            best_score = score
            best_solution = {
                **metrics_local,
                "crate_order": [crate.crate_id for crate in crate_order],
            }

    if best_solution is None:
        if failed_orders:
            missing_labels = [
                f"{item.label} ({item.base_length}×{item.base_width})"
                for item in items_expanded
            ]
            raise RuntimeError(
                "Packing incomplete: could not place items " + ", ".join(missing_labels)
            )
        raise RuntimeError("Packing failed: no feasible container configuration found.")

    initial_bins = best_solution["bin_results"]
    bin_results = optimise_bins_by_cost(initial_bins, crates)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_ROOT / f"pack_{timestamp}"
    images_dir = run_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    crates_by_id = {crate.crate_id: crate for crate in crates}
    crate_sequence: Dict[str, int] = {}
    crate_usage_accumulator: Dict[str, Dict] = {}
    total_used_area = 0.0
    total_crate_area = 0.0
    total_container_cost = 0.0
    has_cost_data = False

    for bin_data in bin_results:
        crate = crates_by_id.get(bin_data["crate_id"])
        if crate is None:
            raise RuntimeError(f"Unknown crate ID in results: {bin_data['crate_id']}")

        crate_sequence[crate.crate_id] = crate_sequence.get(crate.crate_id, 0) + 1
        bin_data["crate_index"] = crate_sequence[crate.crate_id]
        bin_data["crate_length"] = float(crate.length)
        bin_data["crate_width"] = float(crate.width)

        crate_area = crate.length * crate.width
        used_area = sum(item["placed_length"] * item["placed_width"] for item in bin_data["items"])
        bin_data["crate_area"] = crate_area
        bin_data["used_area"] = used_area
        bin_data["utilization"] = used_area / crate_area if crate_area else 0.0

        total_used_area += used_area
        total_crate_area += crate_area

        entry = crate_usage_accumulator.setdefault(
            crate.crate_id,
            {
                "crate_id": crate.crate_id,
                "priority": None if crate.priority >= 10_000_000 else crate.priority,
                "count": 0,
                "length": float(crate.length),
                "width": float(crate.width),
                "total_used_area": 0.0,
                "unit_cost": crate.cost if crate.cost is not None else None,
                "total_cost": 0.0,
            },
        )
        entry["count"] += 1
        entry["total_used_area"] += used_area

        if crate.cost is not None:
            total_container_cost += crate.cost
            has_cost_data = True
            entry["total_cost"] += crate.cost
        else:
            entry["total_cost"] = None

    crate_usage_summary: List[Dict] = []
    for usage in crate_usage_accumulator.values():
        crate_area = usage["length"] * usage["width"]
        usage["avg_utilization"] = (
            usage["total_used_area"] / (crate_area * usage["count"])
            if crate_area and usage["count"]
            else 0.0
        )
        if usage.get("unit_cost") is None:
            usage["total_cost"] = None
        crate_usage_summary.append(usage)

    crate_usage_summary.sort(
        key=lambda entry: (
            entry.get("unit_cost", float("inf")),
            entry.get("priority", 10_000_000),
            entry["crate_id"],
        )
    )

    overall_utilization = total_used_area / total_crate_area if total_crate_area else 0.0
    total_bins_used = len(bin_results)
    total_container_cost = total_container_cost if has_cost_data else None

    summary = {
        "generated_at": timestamp,
        "catalog_version": catalog_version,
        "buffer": buffer,
        "units": units,
        "total_items": len(items_expanded),
        "total_crates_used": total_bins_used,
        "total_containers_used": total_bins_used,
        "total_buffered_area": total_buffered_area,
        "overall_utilization": overall_utilization,
        "container_family": container_family,
        "container_singular": container_singular,
        "container_plural": container_plural,
        "crate_usage": crate_usage_summary,
        "selected_container_order": best_solution.get("crate_order", []),
    }

    summary["total_container_cost"] = total_container_cost
    if cost_overrides:
        summary["cost_overrides"] = {
            crate_id: float(value)
            for crate_id, value in cost_overrides.items()
        }

    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    placements_tables: Dict[str, pd.DataFrame] = {}
    for bin_data in bin_results:
        table_name = f"{bin_data['crate_id']} #{bin_data['crate_index']:02d}"
        placement_rows = []
        for item in bin_data["items"]:
            placement_rows.append({
                "item_id": item["item_id"],
                "label": item["label"],
                "x": item["x"],
                "y": item["y"],
                "length": item["placed_length"],
                "width": item["placed_width"],
                "rotated": item["rotated"],
            })
        placement_df = pd.DataFrame(placement_rows)
        placements_tables[table_name] = placement_df

    bin_images: List[Dict] = []
    for bin_data in bin_results:
        image_name = f"{bin_data['crate_id']}_{bin_data['crate_index']:02d}.png"
        image_path = images_dir / image_name
        image_bytes = render_bin_image(bin_data, units, image_path)
        bin_images.append({
            "file_name": image_name,
            "path": image_path,
            "bytes": image_bytes,
            "bin": bin_data,
        })

    images_zip_path = run_dir / "images_folder.zip"
    images_zip_bytes = make_zip_from_images(bin_images, target_path=images_zip_path)

    return {
        "run_dir": run_dir,
        "summary": summary,
        "bin_results": bin_results,
        "placements": placements_tables,
        "images": bin_images,
        "images_zip_path": images_zip_path,
        "images_zip_bytes": images_zip_bytes,
    }


def make_zip_from_images(images: List[Dict], target_path: Path | None = None) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for image in images:
            zf.writestr(f"images/{image['file_name']}", image["bytes"])
    buffer.seek(0)
    data = buffer.getvalue()
    if target_path is not None:
        target_path.write_bytes(data)
    return data


def make_zip_for_run(run_dir: Path) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(run_dir.rglob("*")):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(run_dir).as_posix())
    buffer.seek(0)
    return buffer.getvalue()

# Load baseline items
items_file = Path(DEFAULT_ITEMS_PATH)
items_mtime = items_file.stat().st_mtime if items_file.exists() else 0.0

try:
    default_items_df = load_default_items(str(items_file), items_mtime)
except Exception as exc:
    st.warning(f"Could not load default items: {exc}")
    default_items_df = pd.DataFrame(columns=["label", "length", "width", "count"])

if "items_df" not in st.session_state:
    st.session_state["items_df"] = default_items_df.copy()

catalog_data: Dict = {}
catalog_entries: List[Dict] = []
catalog_version = "unknown"
container_options = list(CATALOG_SOURCES.keys())
container_family = container_options[0] if container_options else "Crates"
container_singular = CATALOG_SOURCES.get(container_family, {}).get("singular", "crate")
container_plural = CATALOG_SOURCES.get(container_family, {}).get("plural", "crates")
allowed_ids: List[str] = []

with st.sidebar:
    st.subheader("Setup")
    container_family = st.radio(
        "Container catalog",
        options=container_options,
        index=container_options.index(container_family) if container_family in container_options else 0,
    )
    catalog_source = CATALOG_SOURCES.get(container_family)
    if catalog_source is None:
        st.error(f"Unknown container catalog selection: {container_family}")
        st.stop()

    catalog_path = catalog_source["path"]
    catalog_mtime = catalog_path.stat().st_mtime if catalog_path.exists() else 0.0

    try:
        catalog_data = load_catalog(str(catalog_path), catalog_source["list_key"], catalog_mtime)
    except Exception as exc:
        st.error(f"Failed to load {container_family.lower()} catalog: {exc}")
        st.stop()

    catalog_entries = catalog_data.get(catalog_source["list_key"], [])
    if not catalog_entries:
        st.error(f"No entries found in {container_family.lower()} catalog.")
        st.stop()

    catalog_version = catalog_data.get("version", "unknown")
    container_singular = catalog_source["singular"]
    container_plural = catalog_source["plural"]
    allowed_options = [entry.get("id") for entry in catalog_entries if entry.get("id")]
    default_allowed = allowed_options if allowed_options else []

    st.markdown(f"**Catalog version:** `{catalog_version}`")
    buffer_value = st.number_input("Buffer", min_value=0.1, value=0.5, step=1.0)
    select_specific = st.checkbox(
        f"Select specific {container_plural}",
        value=False,
        help="Enable to limit packing to a subset of containers.",
    )
    if select_specific:
        allowed_ids = st.multiselect(
            f"Allowed {container_singular} IDs",
            options=allowed_options,
            default=default_allowed,
        )
    else:
        allowed_ids = default_allowed
    auto_aggregate = st.checkbox("Auto-aggregate identical rows", value=True)

    if "cost_overrides" not in st.session_state:
        st.session_state["cost_overrides"] = {}

    cost_options = ["(none)"] + allowed_options
    selected_cost_target = st.selectbox(
        "Adjust unit cost",
        options=cost_options,
        index=0,
        help="Choose a container to override its cost for this session.",
    )

    if selected_cost_target != "(none)":
        base_cost = next(
            (entry.get("cost") for entry in catalog_entries if entry.get("id") == selected_cost_target),
            None,
        )
        override_cost = st.session_state["cost_overrides"].get(
            selected_cost_target,
            base_cost,
        )
        if override_cost is None:
            override_cost = 0.0
        cost_value = st.number_input(
            "Unit cost",
            min_value=0.0,
            value=float(override_cost),
            step=0.5,
            key=f"cost_input_{selected_cost_target}",
        )
        if st.button("Apply cost", key=f"apply_cost_{selected_cost_target}"):
            st.session_state["cost_overrides"][selected_cost_target] = float(cost_value)
            st.success(f"Cost for {selected_cost_target} set to ${float(cost_value):.2f}")

    st.divider()
    uploaded_file = st.file_uploader("Assemblies CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            missing_cols = {"label", "length", "width", "count"} - set(uploaded_df.columns)
            if missing_cols:
                st.error(f"Uploaded CSV missing columns: {', '.join(sorted(missing_cols))}")
            else:
                st.session_state["items_df"] = uploaded_df[["label", "length", "width", "count"]]
        except Exception as exc:  # pragma: no cover - defensive against unexpected CSV issues
            st.error(f"Failed to read CSV: {exc}")

    st.download_button(
        "Download sample CSV",
        data=default_items_df.to_csv(index=False),
        file_name="items_template.csv",
        mime="text/csv",
    )

items_source_df = st.session_state["items_df"]
if items_source_df.empty:
    st.info("Use the editor below or upload a CSV to add assemblies.")

editor_df = items_source_df.copy()
editor_df = editor_df.astype({"label": "string"}, errors="ignore")
editor_df = editor_df.fillna({"count": 1})
for col in ("length", "width", "count"):
    if col in editor_df.columns:
        editor_df[col] = editor_df[col].apply(normalize_numeric_display)

edited_df = st.data_editor(
    editor_df,
    num_rows="dynamic",
    width="content",
    key="items_editor",
    column_config={
        "label": st.column_config.TextColumn("Label", required=True),
        "length": st.column_config.NumberColumn("Length", min_value=0.0),
        "width": st.column_config.NumberColumn("Width", min_value=0.0),
        "count": st.column_config.NumberColumn("Count", min_value=1, step=1),
    },
)

clean_items_df, validation_issues = sanitise_items(pd.DataFrame(edited_df))
if validation_issues:
    for issue in validation_issues:
        st.warning(issue)

prepared_items_df = aggregate_items(clean_items_df, auto_aggregate)
metrics = compute_preview_metrics(prepared_items_df, buffer_value)

preview_col1, preview_col2, preview_col3, preview_col4 = st.columns(4)
preview_col1.metric("Unique rows", metrics["unique_rows"])
preview_col2.metric("Total assemblies", metrics["total_count"])
preview_col3.metric("Total buffered area", f"{metrics['total_buffered_area']:.0f} {DEFAULT_UNITS}²")
preview_col4.metric(f"Allowed {container_plural}", len(allowed_ids))

if select_specific and not allowed_ids:
    st.error(f"Select at least one {container_singular} to enable packing.")

pack_col = st.container()
pack_button = pack_col.button("Pack", type="primary", disabled=prepared_items_df.empty or not allowed_ids)

if pack_button:
    try:
        cost_overrides_state = st.session_state.get("cost_overrides", {})
        crate_defs = build_crate_objects(
            catalog_entries,
            allowed_ids,
            cost_overrides=cost_overrides_state,
        )
        with st.spinner("Packing assemblies..."):
            result = pack_items(
                items_df=prepared_items_df,
                crates=crate_defs,
                buffer=buffer_value,
                units=DEFAULT_UNITS,
                catalog_version=catalog_version,
                container_family=container_family,
                container_singular=container_singular,
                container_plural=container_plural,
                cost_overrides=cost_overrides_state,
            )
        st.session_state["packing_result"] = result
        st.success(f"Packing complete. Outputs saved to {result['run_dir'].relative_to(BASE_DIR)}")
    except Exception as exc:  # pragma: no cover - UI feedback path
        st.error(str(exc))

result = st.session_state.get("packing_result")

if result:
    summary = result["summary"]
    container_singular_result = summary.get("container_singular", "crate")
    container_plural_result = summary.get("container_plural", "crates")

    tabs = st.tabs(["Summary", "Visuals", "Tables", "Downloads"])

    with tabs[0]:
        st.metric(
            f"{container_plural_result.capitalize()} used",
            summary.get("total_containers_used", summary.get("total_crates_used", 0)),
        )
        st.metric("Overall utilization", f"{summary['overall_utilization'] * 100:.1f}%")
        st.metric("Total buffered area", f"{summary['total_buffered_area']:.0f} {summary['units']}²")
        total_cost_value = summary.get("total_container_cost")
        if total_cost_value is not None:
            st.metric("Total container cost", f"${total_cost_value:,.2f}")
        usage_df = pd.DataFrame(summary["crate_usage"])
        if not usage_df.empty:
            usage_df = usage_df.assign(avg_utilization=lambda df: df["avg_utilization"] * 100)
            usage_df.rename(
                columns={
                    "avg_utilization": "avg_utilization_%",
                    "crate_id": f"{container_singular_result}_id",
                    "count": f"{container_singular_result}_count",
                },
                inplace=True,
            )
            st.dataframe(usage_df, width="content")
        st.write(f"Outputs stored under `{result['run_dir'].relative_to(BASE_DIR)}`")

    with tabs[1]:
        for image_info in result["images"]:
            bin_data = image_info["bin"]
            caption = (
                f"{bin_data['crate_id']} #{bin_data['crate_index']} "
                f"({container_singular_result}) | Utilization {bin_data['utilization'] * 100:.1f}%"
            )
            st.image(image_info["bytes"], caption=caption)

    with tabs[2]:
        for file_name, df in result["placements"].items():
            st.subheader(file_name)
            st.dataframe(df, width="content")

    with tabs[3]:
        summary_json = json.dumps(result["summary"], indent=2).encode("utf-8")
        st.download_button(
            "Download summary.json",
            data=summary_json,
            file_name="summary.json",
            mime="application/json",
        )
        images_zip_bytes = result.get("images_zip_bytes")
        images_zip_path = result.get("images_zip_path")
        if images_zip_bytes and images_zip_path:
            st.download_button(
                "Download PNG folder (zip)",
                data=images_zip_bytes,
                file_name=images_zip_path.name,
                mime="application/zip",
            )
            st.caption(
                f"PNG archive saved to `{images_zip_path.relative_to(BASE_DIR)}` on disk."
            )
        full_bundle = make_zip_for_run(result["run_dir"])
        st.download_button(
            "Download full bundle",
            data=full_bundle,
            file_name=f"{result['run_dir'].name}.zip",
            mime="application/zip",
        )
