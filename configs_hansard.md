# Hansard Training Configs (80GB A100, Chinchilla-optimal)

## Small Model (~100M params, depth=16)
```bash
python -m scripts.base_train \
  --run="hansard_d16" \
  --tokenizer_name="hansard" \
  --depth=16 \
  --max_seq_len=2048 \
  --device_batch_size=96 \
  --total_batch_size=524288 \
  --target_param_data_ratio=20
```
- **Parameters**: ~100M
- **Optimal tokens**: 2B (Chinchilla 20:1)
- **Training steps**: ~3,815
- **Training time**: ~2-4 hours on A100

## Medium Model (~300M params, depth=24) ⭐ RECOMMENDED
```bash
python -m scripts.base_train \
  --run="hansard_d24" \
  --tokenizer_name="hansard" \
  --depth=24 \
  --max_seq_len=2048 \
  --device_batch_size=64 \
  --total_batch_size=524288 \
  --target_param_data_ratio=20
```
- **Parameters**: ~300M
- **Optimal tokens**: 6B (Chinchilla 20:1)
- **Training steps**: ~11,445
- **Training time**: ~8-12 hours on A100

## Large Model (~750M params, depth=36)
```bash
python -m scripts.base_train \
  --run="hansard_d36" \
  --tokenizer_name="hansard" \
  --depth=36 \
  --max_seq_len=2048 \
  --device_batch_size=32 \
  --total_batch_size=524288 \
  --target_param_data_ratio=20
```
- **Parameters**: ~750M
- **Optimal tokens**: 15B (Chinchilla 20:1)
- **Training steps**: ~28,611
- **Training time**: ~24-36 hours on A100

## Extra Large Model (~1.3B params, depth=48)
```bash
python -m scripts.base_train \
  --run="hansard_d48" \
  --tokenizer_name="hansard" \
  --depth=48 \
  --max_seq_len=2048 \
  --device_batch_size=24 \
  --total_batch_size=524288 \
  --target_param_data_ratio=20
```
- **Parameters**: ~1.3B
- **Optimal tokens**: 26B (Chinchilla 20:1)
- **Training steps**: ~49,593
- **Training time**: ~48-72 hours on A100

---

## Hyperparameter Notes

**Batch sizes** are optimized for 80GB A100:
- Reduce `device_batch_size` if you get OOM
- Gradient accumulation automatically adjusts to maintain `total_batch_size=524288`

**Learning rates** (defaults are good for most cases):
- `embedding_lr=0.2` - Adam LR for embeddings
- `unembedding_lr=0.004` - Adam LR for output layer
- `matrix_lr=0.02` - Muon LR for transformer weights

**LR schedule**:
- `warmup_ratio=0.05` - 5% warmup
- `warmdown_ratio=0.2` - 20% cosine decay
- `final_lr_frac=0.0` - decay to 0

**Evaluation**:
- `eval_every=250` - validate every 250 steps
- `core_metric_every=1000` - run eval benchmarks every 1000 steps
- `save_every=2000` - checkpoint every 2000 steps

**Sequence length**:
- `max_seq_len=2048` - good for parliamentary debates
- Can increase to 4096 if your dataset has longer contexts (reduce batch size accordingly)

---

## Quick Start

1. **Count your dataset tokens**:
   ```bash
   python scripts/count_tokens.py --split train
   ```

2. **Choose model size** based on dataset:
   - Dataset < 4B tokens → Use depth=16 (100M params)
   - Dataset 4-10B tokens → Use depth=24 (300M params) ⭐
   - Dataset 10-20B tokens → Use depth=36 (750M params)
   - Dataset > 20B tokens → Use depth=48 (1.3B params)

3. **Run training**:
   ```bash
   chmod +x train_hansard.sh
   ./train_hansard.sh
   ```

4. **Monitor with wandb**:
   ```bash
   # Change --run="hansard_d24" to enable wandb logging
   ```
