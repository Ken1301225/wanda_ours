# Pruned MoE Analysis

`debug/analyze_pruned_moe.py` analyzes saved pruned MoE models and writes static artifacts focused on expert-column collapse.
This is a static weight analysis only. It does not use calibration data or routing traces.

## Usage

Single model:

```bash
python debug/analyze_pruned_moe.py \
  --model /path/to/pruned_model \
  --output-dir debug_outputs/run_a
```

Multiple models:

```bash
python debug/analyze_pruned_moe.py \
  --model /path/to/moe_wanda_model \
  --model /path/to/wanda_model \
  --model /path/to/sparsegpt_model \
  --output-dir debug_outputs/compare_methods
```

## Outputs

For each model the script writes:

- `summary.json`
- `per_module.csv`
- `per_expert.csv`
- `per_layer_projection.csv`
- `per_depth_projection.csv`
- `per_expert_projection.csv`
- `layer_projection_zero_col.png`
- `shallow_mid_deep_projection_bar.png`
- `expert_zero_col_heatmap_{projection}.png`
- `expert_collapse_rank_{projection}.png`

When multiple models are passed it also writes:

- `combined_summary.json`
- `combined_per_module.csv`
- `combined_per_layer_projection.csv`
- `combined_per_depth_projection.csv`
- `combined_per_expert_projection.csv`
- `compare_models_zero_col_summary.png`

## What To Look For

- `layer_projection_zero_col.png`: whether zeroed input columns become more common in deeper layers, and whether `gate/up/down` differ systematically.
- `shallow_mid_deep_projection_bar.png`: a direct shallow-vs-mid-vs-deep comparison for each projection.
- `expert_zero_col_heatmap_*`: whether a few experts are consistently more prone to column collapse across layers.
- `expert_collapse_rank_*`: whether the same experts stay risky when ranked by average severity, worst-case severity, and collapse frequency.
- `compare_models_zero_col_summary.png`: whether one pruning run causes materially more column collapse than another.

## Useful Angles

- Severity vs frequency: `mean_zero_col_ratio` tells you how bad collapse gets, while `collapse_incidence` tells you how often it happens.
- Depth trend: compare shallow, mid, and deep buckets to see whether collapse accumulates with layer depth.
- Projection asymmetry: compare `gate_proj`, `up_proj`, and `down_proj` to find which projection is structurally most fragile.
- Expert concentration: use the expert heatmaps and rank plots to see whether collapse is spread across experts or concentrated in a few weak ones.
- Cross-run comparison: if you pass multiple pruned models, the combined CSVs and summary plot let you compare methods or hyperparameters on the same zero-column metric.
