#!/bin/bash
# Optimized training config for Hansard dataset on 80GB A100
# Following Chinchilla's law (20 tokens per parameter)
#
# Model: ~300M parameters (depth=24)
# Sequence length: 2048 tokens
# Batch size optimized for 80GB A100

python -m scripts.base_train \
  --run="hansard_d24" \
  --tokenizer_name="hansard" \
  --depth=24 \
  --max_seq_len=2048 \
  --device_batch_size=64 \
  --total_batch_size=524288 \
  --target_param_data_ratio=20 \
  --embedding_lr=0.2 \
  --unembedding_lr=0.004 \
  --matrix_lr=0.02 \
  --weight_decay=0.0 \
  --grad_clip=1.0 \
  --warmup_ratio=0.05 \
  --warmdown_ratio=0.2 \
  --final_lr_frac=0.0 \
  --eval_every=250 \
  --eval_tokens=10485760 \
  --core_metric_every=1000 \
  --sample_every=1000 \
  --save_every=2000

# Model architecture breakdown (depth=24):
# - Layers: 24
# - Dimension: 24 * 64 = 1536
# - Heads: (1536 + 127) // 128 = 13
# - Parameters: ~300M
#
# With Chinchilla ratio of 20:
# - Training tokens: 300M * 20 = 6B tokens
# - Iterations: 6B / 524288 ≈ 11,445 steps
#
# Batch size (per GPU):
# - device_batch_size=64, seq_len=2048 → 131,072 tokens/step
# - total_batch_size=524288 → grad_accum_steps=4 (on single GPU)
#
# Memory estimate for 80GB A100:
# - Model: ~1.2GB (fp32 params)
# - Activations: ~8-12GB (with grad checkpointing if needed)
# - Optimizer states: ~2.4GB
# - Batch: ~4GB
# - Total: ~16-20GB → plenty of headroom, can increase device_batch_size if desired
#
# To maximize throughput, you can try:
# - Increase device_batch_size to 96 or 128
# - Enable flash attention (if available)
# - Use torch.compile (already enabled by default)
