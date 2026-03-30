# EvWDiff: Training and Inference Guide

## Prerequisites

- Python 3.8+
- PyTorch 1.12+ with CUDA support
- Dependencies: `pip install pyyaml easydict absl-py tensorboard opencv-python`
- The EvLight codebase with all original dependencies installed
- SDE indoor dataset at: `/media/hzho0442/Project/Code/idea/Low-light/dataset/sde/indoor/sde_in_release`

## Dataset Structure

```
sde_in_release/
├── train/          # 36 sequences
│   └── i_0/
│       ├── low/    # low-light frames (.png) + events (.npz)
│       └── normal/ # ground truth frames (.png)
└── test/           # 7 sequences
    └── ...
```

Each `.npz` file contains events as `arr_0` with shape `(N, 4)` — columns: [timestamp, x, y, polarity].

---

## Training

### 1. Basic Training

```bash
cd /media/hzho0442/Project/Code/idea/Low-light/EvLight

python main_diff.py \
    --yaml_file=options/train/sde_in_diff.yaml \
    --log_dir=./log/evwdiff/sde_in
```

This will:
- Train for 100 epochs with batch size 4
- Save checkpoints to `./log/evwdiff/sde_in/`
- Save `checkpoint.pth.tar` (latest) and `model_best.pth.tar` (best validation loss)
- Save periodic checkpoints every 10 epochs (`checkpoint-000.pth.tar`, etc.)
- Log metrics to TensorBoard

### 2. Monitor Training

```bash
tensorboard --logdir=./log/evwdiff/sde_in
```

Key metrics to watch:
- `Train/noise`: noise prediction MSE loss (should decrease steadily)
- `Train/content`: L1 content loss
- `Train/PSNR`: training PSNR (should increase)
- `Valid/PSNR`, `Valid/SSIM`, `Valid/PSNR_star`: validation metrics

### 3. Resume Training

```bash
python main_diff.py \
    --yaml_file=options/train/sde_in_diff.yaml \
    --log_dir=./log/evwdiff/sde_in \
    --RESUME_PATH=./log/evwdiff/sde_in/checkpoint.pth.tar \
    --RESUME_SET_EPOCH
```

The `--RESUME_SET_EPOCH` flag restores the epoch counter, optimizer state, and scheduler state.

### 4. Override Batch Size

If you encounter GPU memory issues, reduce batch size:

```bash
python main_diff.py \
    --yaml_file=options/train/sde_in_diff.yaml \
    --log_dir=./log/evwdiff/sde_in \
    --TRAIN_BATCH_SIZE=2
```

Expected GPU memory usage:
- Batch size 4: ~20-24 GB
- Batch size 2: ~12-14 GB
- Batch size 1: ~8-10 GB

---

## Inference / Testing

### 1. Evaluate with Metrics

```bash
python main_diff.py \
    --yaml_file=options/train/sde_in_diff.yaml \
    --log_dir=./log/evwdiff/sde_in_test \
    --TEST_ONLY \
    --RESUME_PATH=./log/evwdiff/sde_in/model_best.pth.tar
```

This evaluates on the test set and reports PSNR, SSIM, and PSNR* metrics.

### 2. Evaluate with Visualization

```bash
python main_diff.py \
    --yaml_file=options/train/sde_in_diff.yaml \
    --log_dir=./log/evwdiff/sde_in_test \
    --TEST_ONLY \
    --RESUME_PATH=./log/evwdiff/sde_in/model_best.pth.tar \
    --VISUALIZE
```

This additionally saves prediction and ground truth images to `./log/evwdiff/sde_in_test/epoch-best/` organized by sequence name.

---

## Configuration Reference

Key parameters in `options/train/sde_in_diff.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TRAIN_BATCH_SIZE` | 4 | Training batch size |
| `END_EPOCH` | 100 | Total training epochs |
| `OPTIMIZER.LR` | 1e-4 | Learning rate |
| `EVWDIFF.ema_rate` | 0.999 | EMA decay rate |
| `EVWDIFF.diffusion.num_timesteps` | 200 | Total diffusion timesteps (T) |
| `EVWDIFF.diffusion.sampling_steps` | 10 | DDIM sampling steps (S) |
| `EVWDIFF.model.unet_ch` | 64 | U-Net base channels |
| `EVWDIFF.model.event_channels` | 64 | Event encoder output channels |
| `EVWDIFF.model.modality_dropout` | 0.15 | Event dropout probability |
| `EVWDIFF.model.snr_factor` | 3.0 | SNR normalization factor |
| `EVWDIFF.loss_weights.noise` | 1.0 | Weight for noise prediction loss |
| `EVWDIFF.loss_weights.content` | 1.0 | Weight for content (L1+SSIM) loss |
| `EVWDIFF.loss_weights.hf` | 0.1 | Weight for HF restoration loss |
| `EVWDIFF.loss_weights.tv` | 0.01 | Weight for total variation loss |

---

## Troubleshooting

**Out of Memory:** Reduce `TRAIN_BATCH_SIZE` to 2 or 1 via command line flag.

**NaN losses:** Check that the dataset path is correct and images are valid. The cosine annealing scheduler with `eta_min=1e-7` should prevent learning rate from going to zero.

**Slow convergence:** The diffusion noise loss should decrease within the first few epochs. If not, verify that the event `.npz` files are loading correctly (check for `arr_0` key).

**DDIM sampling artifacts:** Try increasing `sampling_steps` from 10 to 20 in the YAML config for better quality at the cost of slower inference.
