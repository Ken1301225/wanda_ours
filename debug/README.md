# Pruned MoE Analysis

`debug/analyze_pruned_moe.py` analyzes saved pruned MoE models and focuses on significant sparse rows and columns.

A row or column is treated as significant when its zero fraction is at least the configured threshold:

```bash
python debug/analyze_pruned_moe.py \
  --model /path/to/pruned_model \
  --output-dir debug_outputs/run_a \
  --significant-threshold 0.8
```

You can repeat `--model` to compare multiple pruned runs:

```bash
python debug/analyze_pruned_moe.py \
  --model /path/to/moe_wanda_model \
  --model /path/to/another_model \
  --output-dir debug_outputs/compare_methods \
  --significant-threshold 0.8
```

## Outputs

For each model the script writes:

- `summary.txt`
- `depth_projection_summary.txt`
- `expert_risk_rank_gate_proj.txt`
- `expert_risk_rank_up_proj.txt`
- `expert_risk_rank_down_proj.txt`
- `layer_projection_significant_zero_col.png`
- `shallow_mid_deep_projection_bar.png`
- `expert_significant_zero_col_heatmap_gate_proj.png`
- `expert_significant_zero_col_heatmap_up_proj.png`
- `expert_significant_zero_col_heatmap_down_proj.png`
- `weight_overview_heatmap.png`

When multiple models are passed it also writes:

- `combined_summary.txt`
- `combined_depth_projection_summary.txt`
- `combined_expert_risk_rank_gate_proj.txt`
- `combined_expert_risk_rank_up_proj.txt`
- `combined_expert_risk_rank_down_proj.txt`
- `compare_models_significant_zero_col_summary.png`

## What To Look For

- `summary.txt`: whether `gate/up/down` differ in significant sparse row and column behavior at the whole-model level.
- `depth_projection_summary.txt`: whether shallow, mid, and deep layers differ systematically.
- `layer_projection_significant_zero_col.png`: a quick layer-by-layer view of significant sparse columns across `gate/up/down`.
- `shallow_mid_deep_projection_bar.png`: a compact depth-bucket comparison for each projection.
- `expert_significant_zero_col_heatmap_*`: whether a few experts are consistently more vulnerable than the rest.
- `expert_risk_rank_*.txt`: which experts remain risky by average severity, worst-case severity, and collapse frequency.
- `weight_overview_heatmap.png`: raw weight heatmaps arranged as a `3 x 3` grid
  - rows: shallow / mid / deep
  - columns: `gate_proj` / `up_proj` / `down_proj`
  - each panel: the expert with the highest significant sparse-column ratio in that bucket

## Useful Angles

- Significant columns vs significant rows: some projections may collapse along one axis much more than the other.
- Depth trend: whether structured collapse accumulates with layer depth.
- Projection asymmetry: whether `gate_proj`, `up_proj`, and `down_proj` show different failure modes.
- Expert concentration: whether the problem is diffuse or concentrated in a few fragile experts.
- Visual shape: whether weight heatmaps look stripe-like, blocky, or diffuse after pruning.
