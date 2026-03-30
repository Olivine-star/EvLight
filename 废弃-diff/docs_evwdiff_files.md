# EvWDiff: New Files Description

This document describes each new file added to the EvLight codebase for the EvWDiff (Event-guided Wavelet Diffusion) implementation. **No existing files were modified.**

---

## 1. `egllie/models/evwdiff/__init__.py`

**Purpose:** Package initializer that exports the `EvWDiff` model class.

**Rationale:** Provides a clean import interface (`from egllie.models.evwdiff import EvWDiff`) consistent with the existing codebase structure.

---

## 2. `egllie/models/evwdiff/wavelet.py`

**Purpose:** Implements 2D Discrete Wavelet Transform (DWT) and Inverse DWT (IWT) using Haar wavelets with fixed (non-learnable) filter banks.

**Rationale:** Adapted from DiffLL's wavelet implementation. The DWT decomposes an image into four subbands (LL, HL, LH, HH) stacked along the batch dimension. This enables the core idea of performing diffusion only on the low-frequency (LL) component for efficiency, while handling high-frequency components separately via HFRM.

**Key Design:**
- K=1 decomposition level (single-level DWT), producing LL at half resolution (130x173 for SDE's 260x346)
- K=2 was rejected because it would reduce LL to 65x87, which is too small for effective U-Net processing

---

## 3. `egllie/models/evwdiff/unet.py`

**Purpose:** U-Net noise estimator for the diffusion process, with sinusoidal timestep embedding.

**Rationale:** Adapted from DiffLL's U-Net architecture. The key modification is parameterized `in_channels` (10 channels instead of DiffLL's 6) to accommodate the additional event and SNR condition channels:
- 3 channels: conditioned LL (from illumination-enhanced input)
- 3 channels: noisy LL (diffusion process)
- 3 channels: event features projected to LL space
- 1 channel: SNR map at LL scale

**Architecture:** 4-level encoder-decoder with channel multipliers [1, 2, 3, 4], self-attention at level 2, ResNet blocks with GroupNorm, and skip connections.

---

## 4. `egllie/models/evwdiff/hfrm.py`

**Purpose:** Standard High-Frequency Restoration Module (HFRM) with building blocks: `Depth_conv`, `CrossAttention`, `Dilated_Resblock`.

**Rationale:** Provides the foundational components used by the novel `EventGuidedHFRM`. The cross-attention and dilated residual blocks are reused directly. This file is adapted from DiffLL's `mods.py` to serve as a building block library.

---

## 5. `egllie/models/evwdiff/event_guided_hfrm.py` (Novel)

**Purpose:** Event-Guided High-Frequency Restoration Module — the core novel contribution of EvWDiff.

**Rationale:** Standard HFRM (from DiffLL) processes only RGB high-frequency coefficients and cannot leverage event camera data. This module fuses RGB HF and Event HF features using:

1. **Dual processing paths:** Separate dilated residual blocks for RGB and event features per subband (HL, LH, HH)
2. **Cross-attention:** RGB features attend to event features to incorporate edge/structure information from events
3. **SNR-guided soft gating:** A learnable-temperature sigmoid gating mechanism that:
   - High SNR regions → trust RGB HF more (less noise, good texture)
   - Low SNR regions → trust Event HF more (events capture edges reliably in darkness)
4. **Fusion:** Concatenation of gated features with original RGB features, followed by 1x1 convolution

This directly implements the idea's core insight: events provide reliable edge information in dark regions where RGB signal-to-noise ratio is poor.

---

## 6. `egllie/models/evwdiff/evwdiff_model.py` (Main Model)

**Purpose:** The complete EvWDiff model integrating all components.

**Rationale:** Combines EvLight's illumination estimation with DiffLL's wavelet diffusion framework, enhanced by event camera guidance:

**Pipeline:**
1. **Illumination estimation** (replicated from EvLight's `IlluminationNet` to avoid modifying existing code): predicts illumination map from low-light input
2. **Light-up:** `enhanced = low_img * illum + low_img`
3. **SNR computation:** Signal-to-noise ratio map from enhanced vs blur images
4. **Event encoding:** 32-ch voxel grid → 64-ch features via `EventEncoder`
5. **DWT:** Single-level Haar wavelet decomposition → LL + HF subbands
6. **Low-frequency diffusion:** Conditional DDIM on LL with event + SNR conditions (10-ch U-Net input)
7. **High-frequency restoration:** Event-guided HFRM with SNR gating on HF subbands
8. **IWT:** Inverse wavelet transform → final enhanced image

**Key Technical Decisions:**
- **Modality dropout (p=0.15):** Randomly zeros event features during training to prevent the model from becoming overly dependent on events
- **Paired timestep sampling:** `t` and `T-t-1` paired in each batch for stable training (from DiffLL)
- **`_pad_to_multiple()`:** Handles non-power-of-2 spatial dimensions (SDE's 260x346 → LL of 130x173) by padding to multiples of 8 for U-Net compatibility
- **DDIM sampling:** 10-step deterministic sampling for fast inference

**Parameters:** ~22.1M total

---

## 7. `egllie/core/launch_diff.py`

**Purpose:** Training and evaluation launcher for EvWDiff, separate from the original `launch.py`.

**Rationale:** Handles diffusion-specific training requirements:
- **EMAHelper:** Exponential moving average of model weights (rate=0.999) for stable evaluation
- **Multi-loss training:** noise_loss (MSE) + content_loss (L1 + 1-SSIM) + hf_loss (MSE on HF) + TV regularization
- **Gradient clipping:** max_norm=1.0 for training stability
- **EMA validation:** Backup weights → apply EMA → validate → restore original weights
- **Visualization:** Saves prediction and ground truth images during testing

---

## 8. `main_diff.py`

**Purpose:** Entry point for EvWDiff training and evaluation, separate from the original `main.py`.

**Rationale:** Uses the same absl flags pattern as the original codebase but launches `DiffusionLaunch` instead of `ParallelLaunch`. Supports all original CLI flags (resume, test-only, visualization, batch size override).

---

## 9. `options/train/sde_in_diff.yaml`

**Purpose:** Configuration file for training EvWDiff on the SDE indoor dataset.

**Rationale:** Defines all hyperparameters:
- **Training:** batch_size=4, epochs=100, lr=1e-4, Adam optimizer, cosine annealing
- **Diffusion:** T=200 timesteps, 10 DDIM sampling steps, linear beta schedule [1e-4, 0.02]
- **Architecture:** U-Net ch=64, ch_mult=[1,2,3,4], 2 res blocks; event 32→64 channels
- **Losses:** noise=1.0, content=1.0, hf=0.1, tv=0.01
- **Dataset:** Full SDE resolution (260x346), 32-ch event voxel grid
