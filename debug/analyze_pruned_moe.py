import argparse
import math
import re
from collections import defaultdict
from pathlib import Path


EXPERT_NAME_RE = re.compile(r"(?:^|\.)(?:layers)\.(\d+)\..*\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)$")
PROJECTION_ORDER = ("gate_proj", "up_proj", "down_proj")
DEPTH_BUCKET_ORDER = ("shallow", "mid", "deep")
TWILIGHT_CMAP = "twilight"


def build_parser():
    parser = argparse.ArgumentParser(description="Analyze significant sparse rows and columns in pruned MoE expert weights.")
    parser.add_argument("--model", action="append", required=True, help="Path to a pruned model directory. Repeat for multiple models.")
    parser.add_argument("--output-dir", required=True, help="Directory where analysis artifacts will be written.")
    parser.add_argument("--trust-remote-code", action="store_true", default=True, help="Load models with trust_remote_code=True.")
    parser.add_argument("--dpi", type=int, default=180, help="Saved figure DPI.")
    parser.add_argument("--significant-threshold", type=float, default=0.8, help="A row or column is significant when its zero fraction is >= this threshold.")
    parser.add_argument("--weight-max-side", type=int, default=256, help="Maximum rendered width or height for each weight heatmap panel.")
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


def matrix_shape(weight):
    rows = len(weight)
    cols = len(weight[0]) if rows else 0
    return rows, cols


def matrix_to_nested_list(weight):
    if hasattr(weight, "tolist"):
        return weight.tolist()
    return [list(row) for row in weight]


def compute_significant_zero_ratios(weight, significant_threshold):
    matrix = matrix_to_nested_list(weight)
    rows, cols = matrix_shape(matrix)
    if rows == 0 or cols == 0:
        return {
            "sparsity": 0.0,
            "significant_zero_col_ratio": 0.0,
            "significant_zero_row_ratio": 0.0,
            "max_col_zero_fraction": 0.0,
            "max_row_zero_fraction": 0.0,
        }

    zero_count = 0
    row_zero_fractions = []
    col_zero_counts = [0] * cols

    for row in matrix:
        row_zero_count = 0
        for col_idx, value in enumerate(row):
            if value == 0:
                zero_count += 1
                row_zero_count += 1
                col_zero_counts[col_idx] += 1
        row_zero_fractions.append(row_zero_count / cols)

    col_zero_fractions = [count / rows for count in col_zero_counts]
    significant_zero_row_ratio = mean([1.0 if value >= significant_threshold else 0.0 for value in row_zero_fractions])
    significant_zero_col_ratio = mean([1.0 if value >= significant_threshold else 0.0 for value in col_zero_fractions])

    return {
        "sparsity": zero_count / (rows * cols),
        "significant_zero_col_ratio": significant_zero_col_ratio,
        "significant_zero_row_ratio": significant_zero_row_ratio,
        "max_col_zero_fraction": max(col_zero_fractions),
        "max_row_zero_fraction": max(row_zero_fractions),
    }


def depth_bucket_for_position(position, total_positions):
    bucket_idx = min(len(DEPTH_BUCKET_ORDER) - 1, (len(DEPTH_BUCKET_ORDER) * position) // max(1, total_positions))
    return DEPTH_BUCKET_ORDER[bucket_idx]


def annotate_depth_buckets(module_rows):
    rows_by_model = defaultdict(list)
    for row in module_rows:
        rows_by_model[row.get("model_label", "model")].append(row)

    annotated_rows = []
    for model_label, rows in rows_by_model.items():
        layers = sorted({row["layer"] for row in rows})
        layer_to_bucket = {
            layer: depth_bucket_for_position(position, len(layers))
            for position, layer in enumerate(layers)
        }
        for row in rows:
            annotated_rows.append({**row, "depth_bucket": layer_to_bucket[row["layer"]]})

    annotated_rows.sort(
        key=lambda row: (
            row.get("model_label", "model"),
            row["layer"],
            row["expert"],
            PROJECTION_ORDER.index(row["projection"]),
        )
    )
    return annotated_rows


def aggregate_layer_projection_rows(module_rows):
    grouped = defaultdict(list)
    for row in module_rows:
        key = (row["model_label"], row["layer"], row["projection"])
        grouped[key].append(row)

    aggregated = []
    for (model_label, layer, projection), rows in sorted(grouped.items()):
        col_values = [row["significant_zero_col_ratio"] for row in rows]
        row_values = [row["significant_zero_row_ratio"] for row in rows]
        aggregated.append(
            {
                "model_label": model_label,
                "layer": layer,
                "projection": projection,
                "expert_count": len(rows),
                "mean_significant_zero_col_ratio": mean(col_values),
                "max_significant_zero_col_ratio": max(col_values),
                "mean_significant_zero_row_ratio": mean(row_values),
                "max_significant_zero_row_ratio": max(row_values),
                "significant_col_collapse_incidence": collapse_incidence(col_values),
                "significant_row_collapse_incidence": collapse_incidence(row_values),
            }
        )
    return aggregated


def aggregate_layer_depth_projection_rows(module_rows):
    annotated_rows = annotate_depth_buckets(module_rows)
    grouped = defaultdict(list)
    for row in annotated_rows:
        key = (row["model_label"], row["depth_bucket"], row["projection"])
        grouped[key].append(row["significant_zero_col_ratio"])

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
                        "mean_significant_zero_col_ratio": mean(values),
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
        col_values = [row["significant_zero_col_ratio"] for row in rows]
        row_values = [row["significant_zero_row_ratio"] for row in rows]
        aggregated.append(
            {
                "model_label": model_label,
                "expert": expert,
                "projection": projection,
                "layer_count": len(rows),
                "mean_significant_zero_col_ratio": mean(col_values),
                "max_significant_zero_col_ratio": max(col_values),
                "significant_col_collapse_incidence": collapse_incidence(col_values),
                "mean_significant_zero_row_ratio": mean(row_values),
                "max_significant_zero_row_ratio": max(row_values),
                "significant_row_collapse_incidence": collapse_incidence(row_values),
            }
        )
    return aggregated


def aggregate_projection_summary_rows(module_rows):
    grouped = defaultdict(list)
    for row in module_rows:
        key = (row["model_label"], row["projection"])
        grouped[key].append(row)

    aggregated = []
    for (model_label, projection), rows in sorted(grouped.items()):
        col_values = [row["significant_zero_col_ratio"] for row in rows]
        row_values = [row["significant_zero_row_ratio"] for row in rows]
        aggregated.append(
            {
                "model_label": model_label,
                "projection": projection,
                "module_count": len(rows),
                "mean_significant_zero_col_ratio": mean(col_values),
                "max_significant_zero_col_ratio": max(col_values),
                "significant_col_collapse_incidence": collapse_incidence(col_values),
                "mean_significant_zero_row_ratio": mean(row_values),
                "max_significant_zero_row_ratio": max(row_values),
                "significant_row_collapse_incidence": collapse_incidence(row_values),
            }
        )
    return aggregated


def build_ranked_expert_projection_rows(expert_projection_rows, projection):
    ranked = [row for row in expert_projection_rows if row["projection"] == projection]
    return sorted(
        ranked,
        key=lambda row: (
            row["mean_significant_zero_col_ratio"],
            row["max_significant_zero_col_ratio"],
            row["significant_col_collapse_incidence"],
            -row["expert"],
        ),
        reverse=True,
    )


def select_weight_overview_rows(module_rows):
    annotated_rows = annotate_depth_buckets(module_rows)
    selected = {}
    for row in annotated_rows:
        key = (row["depth_bucket"], row["projection"])
        if key not in selected:
            selected[key] = row
            continue
        current = selected[key]
        current_score = (
            current["significant_zero_col_ratio"],
            current.get("significant_zero_row_ratio", 0.0),
            current.get("sparsity", 0.0),
        )
        candidate_score = (
            row["significant_zero_col_ratio"],
            row.get("significant_zero_row_ratio", 0.0),
            row.get("sparsity", 0.0),
        )
        if candidate_score > current_score:
            selected[key] = row
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


def collect_module_rows(model, model_label, significant_threshold):
    import torch.nn as nn

    module_rows = []
    module_weights = {}

    for name, module in model.named_modules():
        parsed = parse_expert_module_name(name)
        if parsed is None or not isinstance(module, nn.Linear):
            continue

        weight = module.weight.detach().float().cpu()
        stats = compute_significant_zero_ratios(weight, significant_threshold=significant_threshold)

        row = {
            "model_label": model_label,
            "module_name": name,
            "layer": parsed["layer"],
            "expert": parsed["expert"],
            "projection": parsed["projection"],
            "rows": int(weight.shape[0]),
            "cols": int(weight.shape[1]),
            "sparsity": float(stats["sparsity"]),
            "significant_zero_col_ratio": float(stats["significant_zero_col_ratio"]),
            "significant_zero_row_ratio": float(stats["significant_zero_row_ratio"]),
            "max_col_zero_fraction": float(stats["max_col_zero_fraction"]),
            "max_row_zero_fraction": float(stats["max_row_zero_fraction"]),
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


def format_cell(value, float_precision):
    if isinstance(value, float):
        return f"{value:.{float_precision}f}"
    return str(value)


def format_ascii_table(rows, columns, float_precision=4):
    headers = [label for _, label in columns]
    body = [[format_cell(row.get(key, ""), float_precision) for key, _ in columns] for row in rows]
    widths = []
    for idx, header in enumerate(headers):
        widths.append(max(len(header), max((len(row[idx]) for row in body), default=0)))

    def format_row(values):
        padded = [value.ljust(widths[idx]) for idx, value in enumerate(values)]
        return "| " + " | ".join(padded) + " |"

    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    lines = [separator, format_row(headers), separator]
    lines.extend(format_row(row) for row in body)
    lines.append(separator)
    return "\n".join(lines)


def write_ascii_report(path, title, rows, columns, float_precision=4):
    ensure_parent(path)
    table = format_ascii_table(rows, columns, float_precision=float_precision)
    Path(path).write_text(f"{title}\n\n{table}\n")


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


def save_projection_summary_plot(projection_summary_rows, path, dpi):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_parent(path)
    models = sorted({row["model_label"] for row in projection_summary_rows})
    colors = build_twilight_palette(len(PROJECTION_ORDER))

    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.8 / max(1, len(PROJECTION_ORDER))
    x_positions = list(range(len(models)))
    for offset, projection in enumerate(PROJECTION_ORDER):
        values = []
        for model in models:
            matched = [
                row["mean_significant_zero_col_ratio"]
                for row in projection_summary_rows
                if row["model_label"] == model and row["projection"] == projection
            ]
            values.append(mean(matched))
        xs = [x + (offset - (len(PROJECTION_ORDER) - 1) / 2) * width for x in x_positions]
        ax.bar(xs, values, width=width, label=projection, color=colors[offset])
    ax.set_title("Mean Significant Zero Column Ratio by Model and Projection")
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
                row["mean_significant_zero_col_ratio"]
                for row in depth_projection_rows
                if row["depth_bucket"] == depth_bucket and row["projection"] == projection
            ]
            values.append(mean(matched))
        xs = [x + (offset - (len(DEPTH_BUCKET_ORDER) - 1) / 2) * width for x in x_positions]
        ax.bar(xs, values, width=width, label=depth_bucket, color=colors[offset])

    ax.set_title("Significant Zero Column Ratio by Layer Depth and Projection")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(PROJECTION_ORDER, rotation=20)
    ax.set_ylim(0.0, 1.0)
    ax.legend()
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


def downsample_weight(weight, max_side):
    if hasattr(weight, "shape"):
        rows = int(weight.shape[0])
        cols = int(weight.shape[1])
        row_step = max(1, math.ceil(rows / max_side))
        col_step = max(1, math.ceil(cols / max_side))
        return weight[::row_step, ::col_step].numpy()

    rows, cols = matrix_shape(weight)
    row_step = max(1, math.ceil(rows / max_side))
    col_step = max(1, math.ceil(cols / max_side))
    return [row[::col_step] for row in weight[::row_step]]


def save_weight_overview_heatmap(selected_rows, module_weights, path, dpi, max_side, model_label):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_parent(path)
    fig, axes = plt.subplots(len(DEPTH_BUCKET_ORDER), len(PROJECTION_ORDER), figsize=(15, 12))
    selected_weights = []
    for row in selected_rows.values():
        selected_weights.append(downsample_weight(module_weights[row["module_name"]], max_side=max_side))

    if selected_weights:
        global_abs_max = max(float(abs(weight).max()) for weight in selected_weights)
    else:
        global_abs_max = 1.0
    if global_abs_max == 0:
        global_abs_max = 1.0

    image = None
    for row_idx, depth_bucket in enumerate(DEPTH_BUCKET_ORDER):
        for col_idx, projection in enumerate(PROJECTION_ORDER):
            ax = axes[row_idx][col_idx]
            selected = selected_rows.get((depth_bucket, projection))
            if selected is None:
                ax.axis("off")
                ax.set_title(f"{depth_bucket} / {projection}\nN/A")
                continue

            weight = downsample_weight(module_weights[selected["module_name"]], max_side=max_side)
            image = ax.imshow(
                weight,
                aspect="auto",
                cmap=TWILIGHT_CMAP,
                vmin=-global_abs_max,
                vmax=global_abs_max,
            )
            ax.set_title(
                f"{depth_bucket} / {projection}\nlayer={selected['layer']} expert={selected['expert']}",
                fontsize=10,
            )
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(f"{model_label} Weight Overview", y=0.995)
    if image is not None:
        fig.colorbar(image, ax=axes, fraction=0.02, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def analyze_single_model(model_path, output_root, dpi, significant_threshold, trust_remote_code, weight_max_side):
    model_path = str(Path(model_path).expanduser())
    model_label = sanitize_label(Path(model_path).name)
    model_output_dir = Path(output_root) / model_label
    model_output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(model_path, trust_remote_code=trust_remote_code)
    module_rows, module_weights = collect_module_rows(
        model,
        model_label,
        significant_threshold=significant_threshold,
    )
    layer_projection_rows = aggregate_layer_projection_rows(module_rows)
    depth_projection_rows = aggregate_layer_depth_projection_rows(module_rows)
    expert_projection_rows = aggregate_expert_projection_rows(module_rows)
    projection_summary_rows = aggregate_projection_summary_rows(module_rows)

    write_ascii_report(
        model_output_dir / "summary.txt",
        f"Summary for {model_label}",
        projection_summary_rows,
        columns=[
            ("projection", "projection"),
            ("module_count", "modules"),
            ("mean_significant_zero_col_ratio", "mean_sig_col"),
            ("max_significant_zero_col_ratio", "max_sig_col"),
            ("significant_col_collapse_incidence", "sig_col_inc"),
            ("mean_significant_zero_row_ratio", "mean_sig_row"),
            ("max_significant_zero_row_ratio", "max_sig_row"),
            ("significant_row_collapse_incidence", "sig_row_inc"),
        ],
    )
    write_ascii_report(
        model_output_dir / "depth_projection_summary.txt",
        f"Depth x Projection Summary for {model_label}",
        depth_projection_rows,
        columns=[
            ("depth_bucket", "depth"),
            ("projection", "projection"),
            ("module_count", "modules"),
            ("mean_significant_zero_col_ratio", "mean_sig_col"),
        ],
    )

    for projection in PROJECTION_ORDER:
        ranked_rows = build_ranked_expert_projection_rows(expert_projection_rows, projection)
        write_ascii_report(
            model_output_dir / f"expert_risk_rank_{projection}.txt",
            f"Expert Risk Rank for {model_label} / {projection}",
            ranked_rows,
            columns=[
                ("expert", "expert"),
                ("layer_count", "layers"),
                ("mean_significant_zero_col_ratio", "mean_sig_col"),
                ("max_significant_zero_col_ratio", "max_sig_col"),
                ("significant_col_collapse_incidence", "sig_col_inc"),
                ("mean_significant_zero_row_ratio", "mean_sig_row"),
                ("max_significant_zero_row_ratio", "max_sig_row"),
                ("significant_row_collapse_incidence", "sig_row_inc"),
            ],
        )

    matrix, row_labels, col_labels = build_layer_projection_matrix(
        layer_projection_rows,
        "mean_significant_zero_col_ratio",
    )
    if matrix:
        save_heatmap(
            matrix,
            row_labels,
            col_labels,
            f"{model_label} layer x projection mean significant zero column ratio",
            model_output_dir / "layer_projection_significant_zero_col.png",
            dpi=dpi,
        )

    if depth_projection_rows:
        save_depth_projection_bar_plot(
            depth_projection_rows,
            model_output_dir / "shallow_mid_deep_projection_bar.png",
            dpi=dpi,
        )

    for projection in PROJECTION_ORDER:
        matrix, row_labels, col_labels = build_projection_heatmap_rows(
            module_rows,
            projection,
            "significant_zero_col_ratio",
        )
        if matrix:
            save_heatmap(
                matrix,
                row_labels,
                col_labels,
                f"{model_label} {projection} significant zero column ratio",
                model_output_dir / f"expert_significant_zero_col_heatmap_{projection}.png",
                dpi=dpi,
            )

    selected_rows = select_weight_overview_rows(module_rows)
    save_weight_overview_heatmap(
        selected_rows,
        module_weights,
        model_output_dir / "weight_overview_heatmap.png",
        dpi=dpi,
        max_side=weight_max_side,
        model_label=model_label,
    )

    del model
    return {
        "model_label": model_label,
        "module_rows": module_rows,
        "layer_projection_rows": layer_projection_rows,
        "depth_projection_rows": depth_projection_rows,
        "expert_projection_rows": expert_projection_rows,
        "projection_summary_rows": projection_summary_rows,
    }


def main():
    parser = build_parser()
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    all_layer_projection_rows = []
    all_depth_projection_rows = []
    all_expert_projection_rows = []
    all_projection_summary_rows = []

    for model_path in args.model:
        result = analyze_single_model(
            model_path=model_path,
            output_root=output_root,
            dpi=args.dpi,
            significant_threshold=args.significant_threshold,
            trust_remote_code=args.trust_remote_code,
            weight_max_side=args.weight_max_side,
        )
        all_layer_projection_rows.extend(result["layer_projection_rows"])
        all_depth_projection_rows.extend(result["depth_projection_rows"])
        all_expert_projection_rows.extend(result["expert_projection_rows"])
        all_projection_summary_rows.extend(result["projection_summary_rows"])

    write_ascii_report(
        output_root / "combined_summary.txt",
        "Combined Summary",
        all_projection_summary_rows,
        columns=[
            ("model_label", "model"),
            ("projection", "projection"),
            ("module_count", "modules"),
            ("mean_significant_zero_col_ratio", "mean_sig_col"),
            ("max_significant_zero_col_ratio", "max_sig_col"),
            ("significant_col_collapse_incidence", "sig_col_inc"),
            ("mean_significant_zero_row_ratio", "mean_sig_row"),
            ("max_significant_zero_row_ratio", "max_sig_row"),
            ("significant_row_collapse_incidence", "sig_row_inc"),
        ],
    )
    write_ascii_report(
        output_root / "combined_depth_projection_summary.txt",
        "Combined Depth x Projection Summary",
        all_depth_projection_rows,
        columns=[
            ("model_label", "model"),
            ("depth_bucket", "depth"),
            ("projection", "projection"),
            ("module_count", "modules"),
            ("mean_significant_zero_col_ratio", "mean_sig_col"),
        ],
    )
    for projection in PROJECTION_ORDER:
        ranked_rows = build_ranked_expert_projection_rows(all_expert_projection_rows, projection)
        write_ascii_report(
            output_root / f"combined_expert_risk_rank_{projection}.txt",
            f"Combined Expert Risk Rank / {projection}",
            ranked_rows,
            columns=[
                ("model_label", "model"),
                ("expert", "expert"),
                ("layer_count", "layers"),
                ("mean_significant_zero_col_ratio", "mean_sig_col"),
                ("max_significant_zero_col_ratio", "max_sig_col"),
                ("significant_col_collapse_incidence", "sig_col_inc"),
                ("mean_significant_zero_row_ratio", "mean_sig_row"),
                ("max_significant_zero_row_ratio", "max_sig_row"),
                ("significant_row_collapse_incidence", "sig_row_inc"),
            ],
        )

    if len(args.model) > 1 and all_projection_summary_rows:
        save_projection_summary_plot(
            all_projection_summary_rows,
            output_root / "compare_models_significant_zero_col_summary.png",
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()
