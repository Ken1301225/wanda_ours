import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path


EXPERT_NAME_RE = re.compile(r"(?:^|\.)(?:layers)\.(\d+)\..*\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)$")
PROJECTION_ORDER = ("gate_proj", "up_proj", "down_proj")
DEPTH_BUCKET_ORDER = ("shallow", "mid", "deep")
TWILIGHT_CMAP = "twilight"


def build_parser():
    parser = argparse.ArgumentParser(description="Analyze sparsity patterns in pruned MoE expert weights.")
    parser.add_argument("--model", action="append", required=True, help="Path to a pruned model directory. Repeat for multiple models.")
    parser.add_argument("--output-dir", required=True, help="Directory where analysis artifacts will be written.")
    parser.add_argument("--trust-remote-code", action="store_true", default=True, help="Load models with trust_remote_code=True.")
    parser.add_argument("--max-pattern-plots", type=int, default=9, help="Maximum number of per-module sparsity pattern images per model.")
    parser.add_argument("--dpi", type=int, default=180, help="Saved figure DPI.")
    parser.add_argument("--pattern-max-side", type=int, default=512, help="Maximum rendered width or height for per-module binary masks.")
    return parser


def parse_expert_module_name(name):
    match = EXPERT_NAME_RE.search(name)
    if not match:
        return None
    layer, expert, projection = match.groups()
    return {
        "layer": int(layer),
        "expert": int(expert),
        "projection": projection,
    }


def mean(values):
    return sum(values) / len(values) if values else 0.0


def collapse_incidence(values):
    return mean([1.0 if value > 0 else 0.0 for value in values])


def aggregate_expert_rows(module_rows):
    grouped = defaultdict(list)
    for row in module_rows:
        key = (row["model_label"], row["layer"], row["expert"])
        grouped[key].append(row)

    aggregated = []
    for (model_label, layer, expert), rows in sorted(grouped.items()):
        aggregated.append(
            {
                "model_label": model_label,
                "layer": layer,
                "expert": expert,
                "projection_count": len(rows),
                "mean_sparsity": mean([row["sparsity"] for row in rows]),
                "max_sparsity": max(row["sparsity"] for row in rows),
                "mean_zero_row_ratio": mean([row["zero_row_ratio"] for row in rows]),
                "max_zero_row_ratio": max(row["zero_row_ratio"] for row in rows),
                "mean_zero_col_ratio": mean([row["zero_col_ratio"] for row in rows]),
                "max_zero_col_ratio": max(row["zero_col_ratio"] for row in rows),
            }
        )
    return aggregated


def depth_bucket_for_position(position, total_positions):
    bucket_idx = min(len(DEPTH_BUCKET_ORDER) - 1, (len(DEPTH_BUCKET_ORDER) * position) // max(1, total_positions))
    return DEPTH_BUCKET_ORDER[bucket_idx]


def aggregate_layer_projection_rows(module_rows):
    grouped = defaultdict(list)
    for row in module_rows:
        key = (row["model_label"], row["layer"], row["projection"])
        grouped[key].append(row)

    aggregated = []
    for (model_label, layer, projection), rows in sorted(grouped.items()):
        aggregated.append(
            {
                "model_label": model_label,
                "layer": layer,
                "projection": projection,
                "expert_count": len(rows),
                "mean_sparsity": mean([row["sparsity"] for row in rows]),
                "max_sparsity": max(row["sparsity"] for row in rows),
                "mean_zero_row_ratio": mean([row["zero_row_ratio"] for row in rows]),
                "max_zero_row_ratio": max(row["zero_row_ratio"] for row in rows),
                "mean_zero_col_ratio": mean([row["zero_col_ratio"] for row in rows]),
                "max_zero_col_ratio": max(row["zero_col_ratio"] for row in rows),
            }
        )
    return aggregated


def aggregate_layer_depth_projection_rows(module_rows):
    rows_by_model = defaultdict(list)
    for row in module_rows:
        rows_by_model[row["model_label"]].append(row)

    grouped = defaultdict(list)
    for model_label, rows in rows_by_model.items():
        layers = sorted({row["layer"] for row in rows})
        layer_to_bucket = {
            layer: depth_bucket_for_position(position, len(layers))
            for position, layer in enumerate(layers)
        }
        for row in rows:
            key = (model_label, layer_to_bucket[row["layer"]], row["projection"])
            grouped[key].append(row["zero_col_ratio"])

    aggregated = []
    for model_label in sorted({key[0] for key in grouped}):
        for depth_bucket in DEPTH_BUCKET_ORDER:
            for projection in PROJECTION_ORDER:
                values = grouped.get((model_label, depth_bucket, projection))
                if not values:
                    continue
                aggregated.append(
                    {
                        "model_label": model_label,
                        "depth_bucket": depth_bucket,
                        "projection": projection,
                        "module_count": len(values),
                        "mean_zero_col_ratio": mean(values),
                    }
                )
    return aggregated


def aggregate_expert_projection_rows(module_rows):
    grouped = defaultdict(list)
    for row in module_rows:
        key = (row["model_label"], row["expert"], row["projection"])
        grouped[key].append(row)

    aggregated = []
    for (model_label, expert, projection), rows in sorted(grouped.items()):
        zero_col_values = [row["zero_col_ratio"] for row in rows]
        aggregated.append(
            {
                "model_label": model_label,
                "expert": expert,
                "projection": projection,
                "layer_count": len(rows),
                "mean_zero_col_ratio": mean(zero_col_values),
                "max_zero_col_ratio": max(zero_col_values),
                "collapse_incidence": collapse_incidence(zero_col_values),
            }
        )
    return aggregated


def build_ranked_expert_projection_rows(expert_projection_rows, projection):
    ranked = [row for row in expert_projection_rows if row["projection"] == projection]
    return sorted(
        ranked,
        key=lambda row: (
            row.get("mean_zero_col_ratio", 0.0),
            row.get("max_zero_col_ratio", 0.0),
            row.get("collapse_incidence", 0.0),
            -row["expert"],
        ),
        reverse=True,
    )


def select_pattern_samples(module_rows, max_pattern_plots):
    per_projection = defaultdict(list)
    for row in module_rows:
        per_projection[row["projection"]].append(row)

    selected = []
    seen = set()
    for projection in PROJECTION_ORDER:
        rows = sorted(
            per_projection.get(projection, []),
            key=lambda row: (row["zero_col_ratio"] + row["zero_row_ratio"] + row["sparsity"]),
            reverse=True,
        )
        if rows:
            selected.append(rows[0])
            seen.add(rows[0]["module_name"])

    if len(selected) >= max_pattern_plots:
        return selected[:max_pattern_plots]

    remaining = sorted(
        [row for row in module_rows if row["module_name"] not in seen],
        key=lambda row: (row["zero_col_ratio"] + row["zero_row_ratio"] + row["sparsity"]),
        reverse=True,
    )
    for row in remaining:
        if len(selected) >= max_pattern_plots:
            break
        selected.append(row)
    return selected


def sanitize_label(name):
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return cleaned.strip("._") or "model"


def load_model(model_path, trust_remote_code=True):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    return model


def collect_module_rows(model, model_label):
    import torch.nn as nn

    module_rows = []

    for name, module in model.named_modules():
        parsed = parse_expert_module_name(name)
        if parsed is None or not isinstance(module, nn.Linear):
            continue

        weight = module.weight.detach().float().cpu()
        zero_mask = weight == 0
        zero_row_mask = zero_mask.all(dim=1)
        zero_col_mask = zero_mask.all(dim=0)

        row = {
            "model_label": model_label,
            "module_name": name,
            "layer": parsed["layer"],
            "expert": parsed["expert"],
            "projection": parsed["projection"],
            "rows": int(weight.shape[0]),
            "cols": int(weight.shape[1]),
            "numel": int(weight.numel()),
            "zero_count": int(zero_mask.sum().item()),
            "sparsity": float(zero_mask.float().mean().item()),
            "zero_row_ratio": float(zero_row_mask.float().mean().item()),
            "zero_col_ratio": float(zero_col_mask.float().mean().item()),
            "mean_abs_weight": float(weight.abs().mean().item()),
            "max_abs_weight": float(weight.abs().max().item()),
            "l2_norm": float(weight.norm().item()),
        }
        module_rows.append(row)

    module_rows.sort(key=lambda row: (row["layer"], row["expert"], PROJECTION_ORDER.index(row["projection"])))
    return module_rows


def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_csv(path, rows):
    ensure_parent(path)
    if not rows:
        Path(path).write_text("")
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    ensure_parent(path)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def downsample_mask(mask, max_side):
    rows, cols = mask.shape
    row_step = max(1, math.ceil(rows / max_side))
    col_step = max(1, math.ceil(cols / max_side))
    return mask[::row_step, ::col_step]


def build_twilight_palette(num_colors):
    import matplotlib

    if num_colors <= 0:
        return []
    positions = [0.5] if num_colors == 1 else [0.15 + 0.7 * idx / (num_colors - 1) for idx in range(num_colors)]
    return [matplotlib.colormaps[TWILIGHT_CMAP](position) for position in positions]


def save_heatmap(matrix, row_labels, col_labels, title, path, dpi, cmap=TWILIGHT_CMAP, vmin=0.0, vmax=1.0):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_parent(path)
    fig_width = max(8, min(18, 0.45 * max(1, len(col_labels))))
    fig_height = max(6, min(18, 0.35 * max(1, len(row_labels))))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=90)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_pattern_mask(weight, title, path, dpi, max_side):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    zero_mask = (weight == 0).numpy().astype(float)
    zero_mask = downsample_mask(zero_mask, max_side=max_side)
    ensure_parent(path)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(zero_mask, aspect="auto", cmap=TWILIGHT_CMAP, vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel("Input Channel")
    ax.set_ylabel("Output Channel")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_projection_summary_plot(layer_projection_rows, path, dpi):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_parent(path)
    rows = layer_projection_rows
    models = sorted({row["model_label"] for row in rows})
    projections = list(PROJECTION_ORDER)
    colors = build_twilight_palette(len(projections))

    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.8 / max(1, len(projections))
    x_positions = list(range(len(models)))
    for offset, projection in enumerate(projections):
        values = []
        for model in models:
            matched = [row["mean_zero_col_ratio"] for row in rows if row["model_label"] == model and row["projection"] == projection]
            values.append(mean(matched))
        xs = [x + (offset - (len(projections) - 1) / 2) * width for x in x_positions]
        ax.bar(xs, values, width=width, label=projection, color=colors[offset])
    ax.set_title("Mean Zero Column Ratio by Model and Projection")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_depth_projection_bar_plot(depth_projection_rows, path, dpi):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_parent(path)
    colors = build_twilight_palette(len(DEPTH_BUCKET_ORDER))
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.8 / max(1, len(DEPTH_BUCKET_ORDER))
    x_positions = list(range(len(PROJECTION_ORDER)))

    for offset, depth_bucket in enumerate(DEPTH_BUCKET_ORDER):
        values = []
        for projection in PROJECTION_ORDER:
            matched = [
                row["mean_zero_col_ratio"]
                for row in depth_projection_rows
                if row["depth_bucket"] == depth_bucket and row["projection"] == projection
            ]
            values.append(mean(matched))
        xs = [x + (offset - (len(DEPTH_BUCKET_ORDER) - 1) / 2) * width for x in x_positions]
        ax.bar(xs, values, width=width, label=depth_bucket, color=colors[offset])

    ax.set_title("Zero Column Ratio by Layer Depth and Projection")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(PROJECTION_ORDER, rotation=20)
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_expert_rank_plot(expert_projection_rows, projection, path, dpi):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ranked_rows = build_ranked_expert_projection_rows(expert_projection_rows, projection)
    if not ranked_rows:
        return

    ensure_parent(path)
    metrics = (
        ("mean_zero_col_ratio", "Mean Zero Col Ratio"),
        ("max_zero_col_ratio", "Max Zero Col Ratio"),
        ("collapse_incidence", "Collapse Incidence"),
    )
    colors = build_twilight_palette(len(metrics))
    expert_labels = [f"expert_{row['expert']}" for row in ranked_rows]
    y_positions = list(range(len(ranked_rows)))
    fig_height = max(6, min(18, 0.35 * len(ranked_rows)))
    fig, axes = plt.subplots(1, 3, figsize=(18, fig_height), sharey=True)

    for idx, (ax, (metric_key, metric_label)) in enumerate(zip(axes, metrics)):
        values = [row[metric_key] for row in ranked_rows]
        ax.barh(y_positions, values, color=colors[idx])
        ax.set_title(metric_label)
        ax.set_xlim(0.0, 1.0)
        ax.invert_yaxis()
        if idx == 0:
            ax.set_yticks(y_positions)
            ax.set_yticklabels(expert_labels)
        else:
            ax.set_yticks(y_positions)
            ax.set_yticklabels([])

    fig.suptitle(f"{projection} Expert Zero-Column Risk Ranking", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def build_projection_heatmap_rows(module_rows, projection, metric_key):
    filtered = [row for row in module_rows if row["projection"] == projection]
    if not filtered:
        return [], [], []

    layers = sorted({row["layer"] for row in filtered})
    experts = sorted({row["expert"] for row in filtered})
    matrix = []
    for layer in layers:
        layer_values = []
        for expert in experts:
            matched = [row[metric_key] for row in filtered if row["layer"] == layer and row["expert"] == expert]
            layer_values.append(matched[0] if matched else 0.0)
        matrix.append(layer_values)
    row_labels = [f"layer_{layer}" for layer in layers]
    col_labels = [f"expert_{expert}" for expert in experts]
    return matrix, row_labels, col_labels


def build_layer_projection_matrix(layer_projection_rows, metric_key):
    if not layer_projection_rows:
        return [], [], []

    layers = sorted({row["layer"] for row in layer_projection_rows})
    projections = [projection for projection in PROJECTION_ORDER if any(row["projection"] == projection for row in layer_projection_rows)]
    matrix = []
    for layer in layers:
        matrix.append(
            [
                next(
                    (
                        row[metric_key]
                        for row in layer_projection_rows
                        if row["layer"] == layer and row["projection"] == projection
                    ),
                    0.0,
                )
                for projection in projections
            ]
        )
    row_labels = [f"layer_{layer}" for layer in layers]
    col_labels = list(projections)
    return matrix, row_labels, col_labels


def summarize_model(module_rows, expert_rows, layer_projection_rows):
    projection_summary = {}
    for projection in PROJECTION_ORDER:
        rows = [row for row in module_rows if row["projection"] == projection]
        if not rows:
            continue
        projection_summary[projection] = {
            "module_count": len(rows),
            "mean_zero_col_ratio": mean([row["zero_col_ratio"] for row in rows]),
            "max_zero_col_ratio": max(row["zero_col_ratio"] for row in rows),
            "collapse_incidence": collapse_incidence([row["zero_col_ratio"] for row in rows]),
        }

    return {
        "module_count": len(module_rows),
        "expert_count": len(expert_rows),
        "layer_projection_count": len(layer_projection_rows),
        "projection_summary": projection_summary,
    }


def analyze_single_model(model_path, output_root, dpi, max_pattern_plots, pattern_max_side, trust_remote_code):
    model_path = str(Path(model_path).expanduser())
    model_label = sanitize_label(Path(model_path).name)
    model_output_dir = Path(output_root) / model_label
    model_output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(model_path, trust_remote_code=trust_remote_code)
    module_rows = collect_module_rows(model, model_label)
    expert_rows = aggregate_expert_rows(module_rows)
    layer_projection_rows = aggregate_layer_projection_rows(module_rows)
    depth_projection_rows = aggregate_layer_depth_projection_rows(module_rows)
    expert_projection_rows = aggregate_expert_projection_rows(module_rows)
    summary = summarize_model(module_rows, expert_rows, layer_projection_rows)

    write_json(model_output_dir / "summary.json", summary)
    write_csv(model_output_dir / "per_module.csv", module_rows)
    write_csv(model_output_dir / "per_expert.csv", expert_rows)
    write_csv(model_output_dir / "per_layer_projection.csv", layer_projection_rows)
    write_csv(model_output_dir / "per_depth_projection.csv", depth_projection_rows)
    write_csv(model_output_dir / "per_expert_projection.csv", expert_projection_rows)

    matrix, row_labels, col_labels = build_layer_projection_matrix(layer_projection_rows, "mean_zero_col_ratio")
    if matrix:
        save_heatmap(
            matrix,
            row_labels,
            col_labels,
            f"{model_label} layer x projection mean zero column ratio",
            model_output_dir / "layer_projection_zero_col.png",
            dpi=dpi,
        )

    if depth_projection_rows:
        save_depth_projection_bar_plot(
            depth_projection_rows,
            model_output_dir / "shallow_mid_deep_projection_bar.png",
            dpi=dpi,
        )

    for projection in PROJECTION_ORDER:
        matrix, row_labels, col_labels = build_projection_heatmap_rows(module_rows, projection, "zero_col_ratio")
        if matrix:
            save_heatmap(
                matrix,
                row_labels,
                col_labels,
                f"{model_label} {projection} zero column ratio",
                model_output_dir / f"expert_zero_col_heatmap_{projection}.png",
                dpi=dpi,
            )
        save_expert_rank_plot(
            expert_projection_rows,
            projection,
            model_output_dir / f"expert_collapse_rank_{projection}.png",
            dpi=dpi,
        )

    del model
    return {
        "model_label": model_label,
        "summary": summary,
        "module_rows": module_rows,
        "expert_rows": expert_rows,
        "layer_projection_rows": layer_projection_rows,
        "depth_projection_rows": depth_projection_rows,
        "expert_projection_rows": expert_projection_rows,
    }


def main():
    parser = build_parser()
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    all_results = []
    all_layer_projection_rows = []
    all_module_rows = []
    all_depth_projection_rows = []
    all_expert_projection_rows = []

    for model_path in args.model:
        result = analyze_single_model(
            model_path=model_path,
            output_root=output_root,
            dpi=args.dpi,
            max_pattern_plots=args.max_pattern_plots,
            pattern_max_side=args.pattern_max_side,
            trust_remote_code=args.trust_remote_code,
        )
        all_results.append({"model_label": result["model_label"], **result["summary"]})
        all_layer_projection_rows.extend(result["layer_projection_rows"])
        all_module_rows.extend(result["module_rows"])
        all_depth_projection_rows.extend(result["depth_projection_rows"])
        all_expert_projection_rows.extend(result["expert_projection_rows"])

    write_json(output_root / "combined_summary.json", all_results)
    write_csv(output_root / "combined_per_module.csv", all_module_rows)
    write_csv(output_root / "combined_per_layer_projection.csv", all_layer_projection_rows)
    write_csv(output_root / "combined_per_depth_projection.csv", all_depth_projection_rows)
    write_csv(output_root / "combined_per_expert_projection.csv", all_expert_projection_rows)

    if len(args.model) > 1 and all_layer_projection_rows:
        save_projection_summary_plot(
            all_layer_projection_rows,
            output_root / "compare_models_zero_col_summary.png",
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()
