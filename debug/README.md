# Pruned MoE Analysis

`debug/analyze_pruned_moe.py` analyzes saved pruned MoE models and writes static artifacts for debugging sparsity structure.

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
- `layer_projection_sparsity.png`
- `expert_sparsity_heatmap_{projection}.png`
- `zero_col_ratio_heatmap_{projection}.png`
- `zero_row_ratio_heatmap_{projection}.png`
- `weight_zero_pattern_samples/*.png`

When multiple models are passed it also writes:

- `combined_summary.json`
- `combined_per_module.csv`
- `combined_per_layer_projection.csv`
- `compare_models_projection_summary.png`

## What To Look For

- `layer_projection_sparsity.png`: whether `gate/up/down` are pruned at similar levels across layers.
- `expert_sparsity_heatmap_*`: whether a few experts are much sparser than the rest.
- `zero_col_ratio_heatmap_*`: whether many input channels are fully wiped out.
- `zero_row_ratio_heatmap_*`: whether many output channels are fully wiped out.
- `weight_zero_pattern_samples/*.png`: whether the actual zero pattern is diffuse, striped, or collapsed into large blocks.
