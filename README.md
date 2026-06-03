# MoE-Wanda

本项目实现了面向 MoE-SwiGLU 语言模型的 MoE-Wanda 剪枝方法，并提供 Qwen1.5-MoE-A2.7B 与 DeepSeek-V2-Lite 的实验脚本、消融脚本、结果日志和论文草稿。

MoE-Wanda 的核心目标是修正普通 Wanda 在 MoE 模型上的结构失配：普通 Wanda 主要使用线性层输入激活尺度，而 MoE-Wanda 显式引入 expert 路由暴露度、`gate_proj` / `up_proj` / `down_proj` 的不同扰动形式、dense-softmax 路由统计、routing power 以及 router-trace expert clustering。

## 项目结构

```text
.
├── main.py                         # Qwen1.5-MoE 剪枝入口
├── main_dsv2.py                    # DeepSeek-V2-Lite 剪枝入口
├── lib/
│   ├── moe_wanda.py                # MoE-Wanda 打分、hook、聚类和 mask 构造
│   ├── prune.py                    # Qwen 剪枝流程
│   ├── prune_dsv2.py               # DeepSeek 剪枝流程
│   ├── eval.py                     # WikiText2 PPL 评估
│   └── data.py                     # 校准数据加载
├── scripts/
│   ├── qwen1_5.sh                  # Qwen 单次运行
│   ├── dsv2.sh                     # DeepSeek 单次运行
│   ├── run_multi_sparsity_dsv2.sh  # DeepSeek 多稀疏度顺序运行
│   ├── ablate_moe_wanda_qwen1_5.sh # Qwen 消融实验
│   ├── ablate_moe_wanda_dsv2.sh    # DeepSeek 消融实验
│   └── run_benchmark_linear_gemm_2_4.sh
├── output/
│   ├── qwen/                       # Qwen 日志和结果
│   ├── dsv2/                       # DeepSeek 日志和结果
│   └── tex/moe_wanda_article.tex   # 论文草稿
└── debug/
    └── analyze_pruned_moe.py       # 已剪枝模型的稀疏结构分析
```

## 环境

需要一个可用的 CUDA PyTorch 环境。模型通过 Hugging Face Transformers 加载，代码中使用：

```text
trust_remote_code=True
torch_dtype=torch.bfloat16
device_map="auto"
```

常用依赖：

```bash
pip install torch transformers accelerate datasets sentencepiece protobuf
```

编译论文草稿：

```bash
xelatex -interaction=nonstopmode -halt-on-error \
  -output-directory output/tex \
  output/tex/moe_wanda_article.tex
```

如果修改了表格、引用或标题，建议连续编译两遍。

## 模型路径和缓存

脚本默认使用本地 Hugging Face cache 路径。建议通过环境变量覆盖，不直接改代码：

```bash
MODEL_ROOT=/data1/ldk/huggingface/hub
HF_CACHE_ROOT=/data1/ldk/huggingface
MODEL=/path/to/model/snapshot
```

主要默认模型：

- Qwen: `Qwen/Qwen1.5-MoE-A2.7B`
- DeepSeek: `deepseek-ai/DeepSeek-V2-Lite`

脚本会设置这些缓存变量：

```text
HF_HOME
HF_DATASETS_CACHE
HF_HUB_CACHE
```

如果本地没有模型，可以下载到指定目录：

```bash
huggingface-cli download deepseek-ai/DeepSeek-V2-Lite --local-dir /path/to/model
```

## 单次运行

### Qwen1.5-MoE

```bash
CUDA_DEVICE=0 \
SPARSITY_RATIO=0.5 \
ROUTING_MODE=dense_softmax \
ROUTING_POWER=1.5 \
CLUSTER_EXPERTS=true \
CLUSTER_K=15 \
bash scripts/qwen1_5.sh
```

注意：当前 `scripts/qwen1_5.sh` 的命令体里直接传入的是：

```text
--sparsity_type 2:4
```

因此它默认用于 Qwen 的结构化 `2:4` 剪枝。

### DeepSeek-V2-Lite

```bash
CUDA_DEVICE=0 \
SPARSITY_RATIO=0.5 \
ROUTING_MODE=dense_softmax \
ROUTING_POWER=1.0 \
CLUSTER_EXPERTS=true \
CLUSTER_K=5 \
DIAGNOSTICS=false \
bash scripts/dsv2.sh
```

注意：当前 `scripts/dsv2.sh` 的命令体里直接传入的是：

```text
--sparsity_type unstructured
```

因此它默认用于 DeepSeek 的非结构化剪枝。如果要跑 DeepSeek `2:4`，建议使用 `scripts/run_multi_sparsity_dsv2.sh`，或者直接调用 `main_dsv2.py`。

## DeepSeek 多稀疏度实验

`scripts/run_multi_sparsity_dsv2.sh` 会在单张 GPU 上顺序运行多个稀疏度，并支持按结果文件续跑。

非结构化主实验示例：

```bash
CUDA_DEVICE=0 \
RUN_NAME=dsv2_unstructured_main \
SPARSITY_TYPE=unstructured \
SPARSITY_RATIOS="0.3 0.4 0.5 0.6 0.7" \
ROUTING_MODE=dense_softmax \
ROUTING_POWER=1.0 \
CLUSTER_EXPERTS=true \
CLUSTER_K=5 \
SAVE_MODEL=true \
DIAGNOSTICS=false \
bash scripts/run_multi_sparsity_dsv2.sh
```

DeepSeek `2:4` 示例：

```bash
CUDA_DEVICE=0 \
RUN_NAME=dsv2_2_4 \
SPARSITY_TYPE=2:4 \
SPARSITY_RATIOS="0.5" \
SAVE_MODEL=true \
bash scripts/run_multi_sparsity_dsv2.sh
```

结构化 `N:M` 稀疏有一个硬约束：

```text
sparsity_ratio = 0.5
```

每个稀疏度的输出目录类似：

```text
<RUN_DIR>/sparsity_<ratio>/output/log_moe_wanda.txt
<RUN_DIR>/sparsity_<ratio>/ckpt/                 # SAVE_MODEL=true 时保存
```

如果中断后要续跑，固定 `RUN_DIR` 并保持：

```bash
SKIP_EXISTING=true
```

脚本会跳过已有 `log_moe_wanda.txt` 的 case。

## 消融实验

### Qwen 消融

```bash
CUDA_DEVICES="0 1" \
MAX_PARALLEL=2 \
SPARSITY_RATIO=0.5 \
SPARSITY_TYPE=unstructured \
bash scripts/ablate_moe_wanda_qwen1_5.sh
```

### DeepSeek 消融

```bash
CUDA_DEVICES="0,1" \
MAX_PARALLEL=2 \
SPARSITY_RATIO=0.5 \
SPARSITY_TYPE=unstructured \
DIAGNOSTICS=false \
bash scripts/ablate_moe_wanda_dsv2.sh
```

DeepSeek 消融脚本会尽量做到一个任务占用一张 GPU。常用续跑参数：

```bash
SKIP_EXISTING=true
SKIP_EXISTING_MODE=result
ABLATION_ROOT=/path/to/existing/ablation
```

只想查看将要运行哪些命令时：

```bash
DRY_RUN=true bash scripts/ablate_moe_wanda_dsv2.sh
```

## 关键参数

`main.py` 和 `main_dsv2.py` 的主要参数：

```text
--model                         本地模型路径或 HF model id
--prune_method moe_wanda
--sparsity_ratio                目标稀疏率
--sparsity_type                 unstructured, 4:8, 或 2:4
--nsamples                      校准样本数量
--seed                          校准数据随机种子
--save                          结果日志目录
--save_model                    剪枝后模型保存目录
--moe_wanda_routing_mode        topk 或 dense_softmax
--moe_wanda_routing_power       路由权重指数 p
--moe_wanda_cluster_experts     启用 router-trace expert clustering
--moe_wanda_cluster_k           expert cluster 数量
--no_diagnostics                关闭详细诊断信息
```

结果日志格式：

```text
method  actual_sparsity  ppl_test
```

## Diagnostics

启用 diagnostics 时，Qwen 路径会额外写出：

```text
diagnostics_moe_wanda.jsonl
diagnostics_moe_wanda.txt
```

如果实验只需要 PPL，建议关闭 diagnostics：

```bash
DIAGNOSTICS=false
```

或者直接传入：

```bash
--no_diagnostics
```

这可以避免额外诊断统计，尤其适合消融实验。

## 已保存模型的 Zero-Shot 评估

本仓库的主入口负责剪枝和 WikiText2 PPL 评估。`output/qwen` 与 `output/dsv2` 下的 zero-shot 日志来自外部评估流程加载保存后的剪枝 checkpoint。

当前论文草稿中的 DeepSeek-V2-Lite 主结果来自：

```text
output/dsv2/eval-origin_20260602-191105.log
output/dsv2/load-wanda*.log
output/dsv2/load-wanda-cluster*.log
```

论文草稿路径：

```text
output/tex/moe_wanda_article.tex
```

## 2:4 GEMM Benchmark

运行结构化稀疏 GEMM 延迟测试：

```bash
DENSE_MODEL=/path/to/dense/model \
PRUNED_MODEL=/path/to/pruned_2_4/model \
OUT_DIR=output/benchmark_projection_table \
bash scripts/run_benchmark_linear_gemm_2_4.sh
```

输出包括 JSON 和 Markdown 格式的延迟统计，主要比较 MoE expert 的 `gate/up/down` projection。

## Debug 工具

分析已保存的剪枝 MoE 模型：

```bash
python debug/analyze_pruned_moe.py \
  --model /path/to/pruned_model \
  --output-dir debug_outputs/run_a
```

该脚本会统计模块、expert、projection 和 zero-pattern 信息。详细输出说明见：

```text
debug/README.md
```

## 注意事项

- DeepSeek-V2 的 gate forward 不直接返回 dense router logits；当前实现会用 `gate.weight` 重新构造 dense logits，用于 dense-softmax 路由统计。
- `dense_softmax` 在校准阶段使用所有 expert 的 softmax 概率；`topk` 只使用实际被路由到的 expert。
- 并行消融不会额外计算指标，但多个进程会竞争 CPU、内存、磁盘 I/O 和 HF/dataset cache。
- DeepSeek 模型加载较重，不建议盲目把并行数拉满；优先从 `MAX_PARALLEL=1` 或 `2` 开始测试总耗时。
