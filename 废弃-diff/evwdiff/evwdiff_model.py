"""
EvWDiff: Event-guided Wavelet Diffusion for Low-Light Image Enhancement.

Architecture:
  1. Illumination estimation + initial light-up (from EvLight)
  2. SNR map computation (from EvLight)
  3. Event feature encoding
  4. Wavelet decomposition (K=1)
  5. Low-frequency: conditional diffusion on LL coefficients (from DiffLL)
     - Event LL features concatenated as additional condition
     - SNR LL map as additional condition channel
  6. High-frequency: Event-guided HFRM with SNR gating (novel)
  7. Wavelet reconstruction -> enhanced image
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from egllie.models.evwdiff.wavelet import DWT, IWT
from egllie.models.evwdiff.unet import DiffusionUNet
from egllie.models.evwdiff.event_guided_hfrm import EventGuidedHFRM


def data_transform(x):
    """Map [0,1] to [-1,1]."""
    return 2 * x - 1.0


def inverse_data_transform(x):
    """Map [-1,1] to [0,1]."""
    return torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)


def get_beta_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timesteps):
    if beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "quad":
        betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5,
                            num_diffusion_timesteps, dtype=np.float64) ** 2
    elif beta_schedule == "cosine":
        steps = num_diffusion_timesteps + 1
        s = 0.008
        t = np.linspace(0, num_diffusion_timesteps, steps, dtype=np.float64)
        alphas_cumprod = np.cos(((t / num_diffusion_timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = np.clip(betas, 0, 0.999)
    else:
        raise NotImplementedError(beta_schedule)
    return betas


class IlluminationNet(nn.Module):
    """Illumination estimation network. Replicates EvLight's IllumiinationNet
    without importing it, to avoid modifying the original code."""
    def __init__(self, illum_level=1, base_chs=48):
        super().__init__()
        self.ill_extractor = nn.Sequential(
            nn.Conv2d(illum_level + 3, illum_level * 2, 3, 1, 1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(illum_level * 2, base_chs, 3, 1, 1),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.reduce = nn.Sequential(
            nn.Conv2d(base_chs, 1, 1, 1, 0),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, ill_map, low_img):
        """
        Args:
            ill_map: illumination prior [B, 1, H, W]
            low_img: low-light image [B, 3, H, W]
        Returns:
            illumination: [B, 1, H, W]
        """
        feat = self.ill_extractor(torch.cat([ill_map, low_img], dim=1))
        return self.reduce(feat)


class EventEncoder(nn.Module):
    """Lightweight event voxel grid encoder."""
    def __init__(self, voxel_channels=32, out_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(voxel_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, voxel_grid):
        return self.encoder(voxel_grid)


class EventLLProjector(nn.Module):
    """Project event features to LL wavelet-domain condition channels."""
    def __init__(self, in_channels=64, out_channels=3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(32, out_channels, 1, 1, 0),
        )

    def forward(self, x):
        return self.proj(x)


class EvWDiff(nn.Module):
    """Event-guided Wavelet Diffusion Model for Low-Light Enhancement.

    Training mode: computes diffusion loss on LL coefficients + HF restoration loss.
    Inference mode: iterative DDIM sampling on LL + HFRM on HF -> IDWT reconstruction.

    Config parameters (from yaml via EasyDict):
        diffusion.num_timesteps: total diffusion steps T (default 200)
        diffusion.sampling_steps: DDIM sampling steps S (default 10)
        diffusion.beta_schedule: "linear" | "cosine"
        diffusion.beta_start: 0.0001
        diffusion.beta_end: 0.02
        model.unet_ch: base channel of U-Net (default 64)
        model.ch_mult: channel multipliers (default [1,2,3,4])
        model.event_channels: event encoder output (default 64)
        model.voxel_grid_channel: voxel channels (default 32)
        model.snr_factor: SNR map scaling (default 3.0)
        model.modality_dropout: probability of dropping event condition (default 0.15)
    """
    def __init__(self, cfg):
        super().__init__()

        # Config
        diff_cfg = cfg.diffusion
        model_cfg = cfg.model

        self.snr_factor = float(model_cfg.get('snr_factor', 3.0))
        self.modality_dropout = float(model_cfg.get('modality_dropout', 0.15))
        self.gamma = float(model_cfg.get('gamma', 0.4))  # Gamma correction for dark images

        # Diffusion schedule
        self.num_timesteps = int(diff_cfg.num_timesteps)
        self.sampling_steps = int(diff_cfg.sampling_steps)
        # Truncated diffusion: start from t_start instead of T-1 during inference
        # Smaller = more conservative refinement (less noise). Default: T//2
        self.t_start_infer = int(diff_cfg.get('t_start_infer', self.num_timesteps // 2))
        betas = get_beta_schedule(
            diff_cfg.beta_schedule,
            diff_cfg.beta_start, diff_cfg.beta_end,
            self.num_timesteps
        )
        self.register_buffer('betas', torch.from_numpy(betas).float())

        # Illumination network
        self.illum_net = IlluminationNet(
            illum_level=int(model_cfg.get('illum_level', 1)),
            base_chs=int(model_cfg.get('illum_base_chs', 48))
        )

        # Event encoder
        self.event_encoder = EventEncoder(
            voxel_channels=int(model_cfg.get('voxel_grid_channel', 32)),
            out_channels=int(model_cfg.get('event_channels', 64))
        )

        # Event LL projector: maps event features (after DWT) to 3 channels for U-Net condition
        event_ch = int(model_cfg.get('event_channels', 64))
        self.event_ll_proj = EventLLProjector(in_channels=event_ch, out_channels=3)

        # Wavelet transforms
        self.dwt = DWT()
        self.iwt = IWT()

        # U-Net noise estimator
        # Input channels: condition_LL(3) + noisy_LL(3) + event_LL(3) + snr_LL(1) = 10
        unet_in_ch = 3 + 3 + 3 + 1
        self.unet = DiffusionUNet(
            in_channels=unet_in_ch,
            ch=int(model_cfg.get('unet_ch', 64)),
            out_ch=3,
            ch_mult=tuple(model_cfg.get('ch_mult', [1, 2, 3, 4])),
            num_res_blocks=int(model_cfg.get('num_res_blocks', 2)),
            dropout=float(model_cfg.get('dropout', 0.0)),
        )

        # Event-guided HFRM for high-frequency restoration
        self.ev_hfrm = EventGuidedHFRM(
            rgb_channels=3,
            event_channels=event_ch,
            hidden_channels=64
        )

    @staticmethod
    def compute_alpha(beta, t):
        beta = torch.cat([torch.zeros(1, device=beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    def _snr_generate(self, enhanced_img, blurred_img):
        """Compute SNR map from light-up image and its blurred version."""
        dark = enhanced_img[:, 0:1] * 0.299 + enhanced_img[:, 1:2] * 0.587 + enhanced_img[:, 2:3] * 0.114
        light = blurred_img[:, 0:1] * 0.299 + blurred_img[:, 1:2] * 0.587 + blurred_img[:, 2:3] * 0.114
        noise = torch.abs(dark - light)
        mask = light / (noise + 0.0001)

        b, _, h, w = mask.shape
        mask_max = mask.reshape(b, -1).max(dim=1)[0].reshape(b, 1, 1, 1)
        mask = mask * self.snr_factor / (mask_max + 0.0001)
        mask = torch.clamp(mask, 0.0, 1.0)
        return mask

    def _pad_to_even(self, x):
        """Pad input to even H and W for DWT compatibility."""
        _, _, h, w = x.shape
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        return x, h, w

    def _downsample_event(self, event_feat, target_h, target_w):
        """Downsample event features to match wavelet-domain spatial size."""
        if event_feat.shape[2] != target_h or event_feat.shape[3] != target_w:
            event_feat = F.interpolate(event_feat, size=(target_h, target_w),
                                       mode='bilinear', align_corners=False)
        return event_feat

    @staticmethod
    def _pad_to_multiple(x, multiple=8):
        """Pad spatial dims to nearest multiple (needed for U-Net downsampling)."""
        _, _, h, w = x.shape
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        return x, h, w

    def _run_unet(self, condition_ll, noisy_ll, event_ll, snr_ll, t):
        """Run U-Net with automatic padding/unpadding."""
        # Pad all inputs to U-Net compatible size
        cond_p, orig_h, orig_w = self._pad_to_multiple(condition_ll, 8)
        noisy_p = self._pad_to_multiple(noisy_ll, 8)[0]
        ev_p = self._pad_to_multiple(event_ll, 8)[0]
        snr_p = self._pad_to_multiple(snr_ll, 8)[0]

        unet_input = torch.cat([cond_p, noisy_p, ev_p, snr_p], dim=1)
        out = self.unet(unet_input, t)
        return out[:, :, :orig_h, :orig_w]

    def _ddim_sample(self, condition_ll, event_ll, snr_ll, b,
                     x0_init=None, t_start=None):
        """DDIM deterministic sampling for residual LL coefficients.

        Args:
            x0_init: Initial residual estimate (default: zeros = no prior).
                     Used for truncated diffusion: noise is added at t_start level.
            t_start: Starting timestep for truncated diffusion (default: T-1 = full).
                     Smaller t_start means less denoising = more conservative refinement.
        """
        if t_start is None:
            t_start = self.num_timesteps - 1

        skip = self.num_timesteps // self.sampling_steps
        seq = list(range(0, self.num_timesteps, skip))
        if seq[-1] != self.num_timesteps - 1:
            seq[-1] = self.num_timesteps - 1

        # Truncate: only use timesteps <= t_start
        seq = [s for s in seq if s <= t_start]
        if len(seq) == 0 or seq[-1] != t_start:
            seq.append(t_start)
            seq.sort()
        seq_next = [-1] + list(seq[:-1])

        n, c, h, w = condition_ll.shape

        if x0_init is None:
            x0_init = torch.zeros(n, c, h, w, device=condition_ll.device)

        # Initialize from noisy x0_init at t_start level
        a_start = (1 - b).cumprod(dim=0)[t_start]
        noise = torch.randn(n, c, h, w, device=condition_ll.device)
        x = a_start.sqrt() * x0_init + (1 - a_start).sqrt() * noise

        for i, j in zip(reversed(list(seq)), reversed(list(seq_next))):
            t = (torch.ones(n, device=x.device) * i)
            next_t = (torch.ones(n, device=x.device) * j)
            at = self.compute_alpha(b, t.long())
            at_next = self.compute_alpha(b, next_t.long())

            et = self._run_unet(condition_ll, x, event_ll, snr_ll, t)
            x0_t = (x - et * (1 - at).sqrt()) / at.sqrt()

            # Deterministic DDIM (eta=0)
            c2 = (1 - at_next).sqrt()
            x = at_next.sqrt() * x0_t + c2 * et

        return x

    def forward(self, batch):
        """
        Args:
            batch: dict with keys from EvLight dataloader:
                - lowligt_image: [B,3,H,W] low-light RGB in [0,1]
                - normalligt_image: [B,3,H,W] ground truth in [0,1]
                - event_free: [B,32,H,W] event voxel grid
                - lowlight_image_blur: [B,3,H,W] blurred low-light
                - ill_list: list of illumination maps

        Returns:
            dict with 'pred', 'gt', and loss components during training.
        """
        low_img = batch['lowligt_image']
        gt_img = batch['normalligt_image']
        event_voxel = batch['event_free']
        low_blur = batch['lowlight_image_blur']
        ill_map = batch['ill_list'][0]

        # 1. Illumination estimation + light-up + gamma boost
        illum = self.illum_net(ill_map, low_img)
        enhanced = low_img * illum + low_img
        enhanced_blur = low_blur * illum + low_blur
        # Gamma correction to boost extremely dark images into visible range
        # Without this, enhanced ≈ 0 and the wavelet condition is uninformative
        gamma = self.gamma
        enhanced = torch.clamp(enhanced, 1e-3, 1).pow(gamma)
        enhanced_blur = torch.clamp(enhanced_blur, 1e-3, 1).pow(gamma)

        # 2. SNR map
        snr_map = self._snr_generate(enhanced, enhanced_blur)

        # 3. Event encoding
        event_feat = self.event_encoder(event_voxel)  # [B, 64, H, W]

        # Modality dropout during training: randomly zero event features
        if self.training and self.modality_dropout > 0:
            drop_mask = (torch.rand(event_feat.shape[0], 1, 1, 1,
                                    device=event_feat.device) > self.modality_dropout).float()
            event_feat = event_feat * drop_mask

        # 4. Pad to even dimensions for DWT
        enhanced_padded, orig_h, orig_w = self._pad_to_even(enhanced)
        gt_padded = F.pad(gt_img, (0, enhanced_padded.shape[3] - orig_w,
                                    0, enhanced_padded.shape[2] - orig_h), mode='reflect')
        event_padded = F.pad(event_feat, (0, enhanced_padded.shape[3] - orig_w,
                                           0, enhanced_padded.shape[2] - orig_h), mode='reflect')
        snr_padded = F.pad(snr_map, (0, enhanced_padded.shape[3] - orig_w,
                                      0, enhanced_padded.shape[2] - orig_h), mode='reflect')

        # 5. Wavelet decomposition (K=1)
        n = enhanced_padded.shape[0]
        enhanced_norm = data_transform(enhanced_padded)

        input_dwt = self.dwt(enhanced_norm)
        input_LL = input_dwt[:n]          # [B, 3, H/2, W/2]
        input_HF = input_dwt[n:]          # [3B, 3, H/2, W/2] - HL, LH, HH

        # Event features at LL scale
        ll_h, ll_w = input_LL.shape[2], input_LL.shape[3]
        event_ll = self._downsample_event(event_padded, ll_h, ll_w)
        event_ll_cond = self.event_ll_proj(event_ll)  # [B, 3, H/2, W/2]

        # SNR at LL scale
        snr_ll = F.interpolate(snr_padded, size=(ll_h, ll_w),
                               mode='bilinear', align_corners=False)

        b = self.betas

        if self.training:
            # === Residual Diffusion training ===
            # Diffusion target is the RESIDUAL: gt_LL - input_LL
            # This is much easier to learn than full gt_LL from noise
            gt_norm = data_transform(gt_padded)
            gt_dwt = self.dwt(gt_norm)
            gt_LL = gt_dwt[:n]
            gt_HF = gt_dwt[n:]

            # Residual target
            residual_LL = gt_LL - input_LL

            # Forward diffusion on residual
            t = torch.randint(0, self.num_timesteps,
                              (n // 2 + 1,), device=low_img.device)
            t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
            a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
            e = torch.randn_like(residual_LL)
            noisy_residual = residual_LL * a.sqrt() + e * (1 - a).sqrt()

            # U-Net predicts noise in residual space
            noise_pred = self._run_unet(input_LL, noisy_residual, event_ll_cond, snr_ll, t.float())

            # Differentiable x0 prediction (predicted residual)
            residual_pred = (noisy_residual - noise_pred * (1 - a).sqrt()) / a.sqrt()

            # Reconstruct full LL = input_LL + predicted_residual
            pred_LL = input_LL + residual_pred

            # HF restoration with event guidance
            restored_HF = self.ev_hfrm(input_HF, event_ll, snr_ll)

            # Differentiable reconstruction (gradient flows to U-Net)
            pred_img = self.iwt(torch.cat([pred_LL, restored_HF], dim=0))
            pred_img = inverse_data_transform(pred_img)
            pred_img = pred_img[:, :, :orig_h, :orig_w]

            return {
                'pred': pred_img,
                'gt': gt_img,
                'noise_pred': noise_pred,
                'noise_gt': e,
                'restored_HF': restored_HF,
                'gt_HF': gt_HF,
                'x0_pred_LL': pred_LL,
                'gt_LL': gt_LL,
                'alpha_bar': a,  # For timestep-weighted content loss
            }
        else:
            # === Inference: Multi-step or single-step residual prediction ===
            if self.t_start_infer > 0:
                # Multi-step: Truncated DDIM from t_start
                residual_LL = self._ddim_sample(
                    input_LL, event_ll_cond, snr_ll, b,
                    x0_init=torch.zeros_like(input_LL),
                    t_start=self.t_start_infer
                )
            else:
                # Single-step: direct x0 prediction at a mid-level timestep
                # More stable than DDIM, matches training behavior
                t_mid = self.num_timesteps // 4  # Use moderate noise level
                a_mid = (1 - b).cumprod(dim=0)[t_mid]
                noise = torch.randn_like(input_LL)
                # x_t = sqrt(a) * 0 + sqrt(1-a) * noise (zero residual + noise)
                x_t = (1 - a_mid).sqrt() * noise
                t_tensor = torch.ones(n, device=input_LL.device) * t_mid
                noise_pred = self._run_unet(input_LL, x_t, event_ll_cond, snr_ll, t_tensor)
                residual_LL = (x_t - noise_pred * (1 - a_mid).sqrt()) / a_mid.sqrt()
            denoise_LL = input_LL + residual_LL

            # HF restoration
            restored_HF = self.ev_hfrm(input_HF, event_ll, snr_ll)

            # Reconstruct
            pred_img = self.iwt(torch.cat([denoise_LL, restored_HF], dim=0))
            pred_img = inverse_data_transform(pred_img)
            pred_img = pred_img[:, :, :orig_h, :orig_w]

            return {
                'pred': pred_img,
                'gt': gt_img,
            }
