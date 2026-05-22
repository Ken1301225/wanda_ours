import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path


EXPERT_NAME_RE = re.compile(r"(?:^|\.)(?:layers)\.(\d+)\..*\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)$")
PROJECTION_ORDER = ("gate_proj", "up_proj", "down_proj")


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
    module_weights = {}

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
        module_weights[name] = weight

    module_rows.sort(key=lambda row: (row["layer"], row["expert"], PROJECTION_ORDER.index(row["projection"])))
    return module_rows, module_weights


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


def save_heatmap(matrix, row_labels, col_labels, title, path, dpi, cmap="viridis", vmin=0.0, vmax=1.0):
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
    ax.imshow(zero_mask, aspect="auto", cmap="gray_r", vmin=0.0, vmax=1.0)
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
    stats = {
        "mean_sparsity": "Mean Sparsity",
        "mean_zero_col_ratio": "Mean Zero Column Ratio",
        "mean_zero_row_ratio": "Mean Zero Row Ratio",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (metric_key, metric_label) in zip(axes, stats.items()):
        width = 0.8 / max(1, len(projections))
        x_positions = list(range(len(models)))
        for offset, projection in enumerate(projections):
            values = []
            for model in models:
                matched = [row[metric_key] for row in rows if row["model_label"] == model and row["projection"] == projection]
                values.append(mean(matched))
            xs = [x + (offset - (len(projections) - 1) / 2) * width for x in x_positions]
            ax.bar(xs, values, width=width, label=projection)
        ax.set_title(metric_label)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(models, rotation=25, ha="right")
        ax.set_ylim(0.0, 1.0)
    axes[0].legend()
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
            "mean_sparsity": mean([row["sparsity"] for row in rows]),
            "mean_zero_row_ratio": mean([row["zero_row_ratio"] for row in rows]),
            "mean_zero_col_ratio": mean([row["zero_col_ratio"] for row in rows]),
            "max_zero_row_ratio": max(row["zero_row_ratio"] for row in rows),
            "max_zero_col_ratio": max(row["zero_col_ratio"] for row in rows),
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
    module_rows, module_weights = collect_module_rows(model, model_label)
    expert_rows = aggregate_expert_rows(module_rows)
    layer_projection_rows = aggregate_layer_projection_rows(module_rows)
    summary = summarize_model(module_rows, expert_rows, layer_projection_rows)

    write_json(model_output_dir / "summary.json", summary)
    write_csv(model_output_dir / "per_module.csv", module_rows)
    write_csv(model_output_dir / "per_expert.csv", expert_rows)
    write_csv(model_output_dir / "per_layer_projection.csv", layer_projection_rows)

    matrix, row_labels, col_labels = build_layer_projection_matrix(layer_projection_rows, "mean_sparsity")
    if matrix:
        save_heatmap(
            matrix,
            row_labels,
            col_labels,
            f"{model_label} layer x projection mean sparsity",
            model_output_dir / "layer_projection_sparsity.png",
            dpi=dpi,
        )

    for projection in PROJECTION_ORDER:
        matrix, row_labels, col_labels = build_projection_heatmap_rows(module_rows, projection, "sparsity")
        if matrix:
            save_heatmap(
                matrix,
                row_labels,
                col_labels,
                f"{model_label} {projection} sparsity",
                model_output_dir / f"expert_sparsity_heatmap_{projection}.png",
                dpi=dpi,
            )

        matrix, row_labels, col_labels = build_projection_heatmap_rows(module_rows, projection, "zero_col_ratio")
        if matrix:
            save_heatmap(
                matrix,
                row_labels,
                col_labels,
                f"{model_label} {projection} zero column ratio",
                model_output_dir / f"zero_col_ratio_heatmap_{projection}.png",
                dpi=dpi,
            )

        matrix, row_labels, col_labels = build_projection_heatmap_rows(module_rows, projection, "zero_row_ratio")
        if matrix:
            save_heatmap(
                matrix,
                row_labels,
                col_labels,
                f"{model_label} {projection} zero row ratio",
                model_output_dir / f"zero_row_ratio_heatmap_{projection}.png",
                dpi=dpi,
            )

    selected_rows = select_pattern_samples(module_rows, max_pattern_plots=max_pattern_plots)
    for row in selected_rows:
        weight = module_weights[row["module_name"]]
        save_pattern_mask(
            weight,
            f"{model_label} {row['module_name']}",
            model_output_dir / "weight_zero_pattern_samples" / f"{row['layer']:03d}_expert_{row['expert']:03d}_{row['projection']}.png",
            dpi=dpi,
            max_side=pattern_max_side,
        )

    del model
    return {
        "model_label": model_label,
        "summary": summary,
        "module_rows": module_rows,
        "expert_rows": expert_rows,
        "layer_projection_rows": layer_projection_rows,
    }


def main():
    parser = build_parser()
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    all_results = []
    all_layer_projection_rows = []
    all_module_rows = []

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

    write_json(output_root / "combined_summary.json", all_results)
    write_csv(output_root / "combined_per_module.csv", all_module_rows)
    write_csv(output_root / "combined_per_layer_projection.csv", all_layer_projection_rows)

    if len(args.model) > 1 and all_layer_projection_rows:
        save_projection_summary_plot(
            all_layer_projection_rows,
            output_root / "compare_models_projection_summary.png",
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()
