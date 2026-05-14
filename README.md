# Foothold

LLM 推理算子性能基准测试工具，支持 FP16 精度下 GEMM、Attention 子算子、LayerNorm/RMSNorm 的延迟测量。

## 环境

- Python ≥ 3.10
- NVIDIA GPU（8GB+ VRAM），CUDA ≥ 12.6
- 使用 [uv](https://docs.astral.sh/uv/) 管理依赖

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install pyyaml
```

## 用法

```bash
python run_all.py                          # 使用 config/default.yaml
python run_all.py --config config/custom.yaml
```

## 配置

编辑 [config/default.yaml](config/default.yaml) 修改参数扫描范围和测量精度：

```yaml
batch_sizes: [1, 2, 4, 8]
seq_lens: [256, 512, 1024, 2048, 4096]
hidden_dims: [2048, 4096]
num_heads: 32
dtype: "float16"
warmup_iters: 5
bench_iters: 100
max_memory_gb: 7.5      # 保护 8GB 显存，超限自动跳过
```

参数会被做笛卡尔积，每个 `[b, s, h]` 组合生成一组 shape。

## 输出

结果保存到 `results/` 目录：

| 文件 | 内容 |
|------|------|
| `gemm.csv` | Q/K/V/O 投影、FFN up/gate/down |

| `attention.csv` | QK^T matmul、softmax、score×V matmul |

| `norm.csv` | LayerNorm、RMSNorm |

| `all_operators.csv` | 所有结果汇总 |

CSV 字段：`op_name, b, s, h, time_ms`（GEMM 额外包含 `M, K, N`）。

## 算子覆盖

| 类别 | 算子 | Shape 示例 |
|------|------|-----------|
| GEMM | q_proj / k_proj / v_proj / o_proj | [b·s, h] × [h, h] |
| GEMM | ffn_up / ffn_gate | [b·s, h] × [h, 4h] |
| GEMM | ffn_down | [b·s, 4h] × [4h, h] |
| Attention | qk_matmul | [b·n_heads, s, d] × [b·n_heads, d, s] |
| Attention | softmax | [b, n_heads, s, s] |
| Attention | score_v_matmul | [b·n_heads, s, s] × [b·n_heads, s, d] |
| Norm | layernorm | [b, s, h] |
| Norm | rmsnorm | [b, s, h] |

## 测量方法

- 使用 `torch.cuda.Event` 精确计时
- 每个 shape 先 warmup 5 次，再正式测量 100 次取均值
- 结果单位为毫秒（ms）
