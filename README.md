# nano-vLLM benchmark

Minimal serving comparison between [nano-vllm](GeeeekExplorer/nano-vllm@bb823b3) and [nano-vllm-v1](slwang-ustc/nano-vllm-v1@357860a).

## How to run

```bash
git clone --recurse-submodules \
  https://github.com/junuxyz/nano-vllm-bench.git

cd nano-vllm-bench
```

```bash
uv sync
```

### nano-vllm

```bash
uv run python bench.py \
  --engine nano-vllm \
  --model "$HOME/huggingface/Qwen3-8B" \
  --num-requests 512 \
  --request-rate 1 \
  --min-input-len 128 \
  --max-input-len 2048 \
  --min-output-len 64 \
  --max-output-len 256 \
  --max-model-len 4096 \
  --token-budget 512 \
  --max-num-seqs 16 \
  --output results/nano-vllm-rate1.json
```

### nano-vllm-v1

```bash
uv run python bench.py \
  --engine nano-vllm-v1 \
  --model "$HOME/huggingface/Qwen3-8B" \
  --num-requests 512 \
  --request-rate 1 \
  --min-input-len 128 \
  --max-input-len 2048 \
  --min-output-len 64 \
  --max-output-len 256 \
  --max-model-len 4096 \
  --token-budget 512 \
  --max-num-seqs 16 \
  --output results/nano-vllm-v1-rate1.json
```

Results are written to `results/<engine>.json` by default, which includes throughput and p50/p90/p99 TTFT, TPOT, and ITL.
