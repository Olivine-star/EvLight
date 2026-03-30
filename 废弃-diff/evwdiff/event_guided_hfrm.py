"""
Event-Guided High-Frequency Restoration Module with SNR gating.
Core novelty: fuses RGB and Event high-frequency wavelet coefficients
using SNR-guided soft gating + cross-attention.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from egllie.models.evwdiff.hfrm import Depth_conv, CrossAttention, Dilated_Resblock


class EventGuidedHFRM(nn.Module):
    """Event-Guided HFRM that replaces the standard HFRM.

    Instead of processing only RGB high-frequency coefficients,
    it fuses RGB HF and Event HF features using SNR-guided soft gating:
      - High SNR regions -> trust RGB HF (less noise, good texture)
      - Low SNR regions -> trust Event HF (edges/structure from events)

    Args:
        rgb_channels: channels of RGB HF coefficients (3)
        event_channels: channels of event features at this scale
        hidden_channels: internal processing channels (64)
        snr_temperature: temperature for sigmoid gating (learnable)
    """
    def __init__(self, rgb_channels=3, event_channels=64, hidden_channels=64):
        super().__init__()

        # RGB HF processing path
        self.rgb_head = Depth_conv(rgb_channels, hidden_channels)
        self.rgb_dilated_HL = Dilated_Resblock(hidden_channels, hidden_channels)
        self.rgb_dilated_LH = Dilated_Resblock(hidden_channels, hidden_channels)
        self.rgb_dilated_HH = Dilated_Resblock(hidden_channels, hidden_channels)

        # Event HF processing path
        self.event_proj = nn.Sequential(
            nn.Conv2d(event_channels, hidden_channels, 3, 1, 1),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.event_dilated_HL = Dilated_Resblock(hidden_channels, hidden_channels)
        self.event_dilated_LH = Dilated_Resblock(hidden_channels, hidden_channels)
        self.event_dilated_HH = Dilated_Resblock(hidden_channels, hidden_channels)

        # Cross-attention: RGB attends to Event structure
        self.cross_attn_HL = CrossAttention(hidden_channels, num_heads=8)
        self.cross_attn_LH = CrossAttention(hidden_channels, num_heads=8)
        self.cross_attn_HH_fuse = nn.Conv2d(hidden_channels * 2, hidden_channels, 3, 1, 1)

        # SNR gating - learnable temperature
        self.snr_temperature = nn.Parameter(torch.tensor(5.0))

        # Fusion and output
        self.fusion_HL = nn.Conv2d(hidden_channels * 2, hidden_channels, 1)
        self.fusion_LH = nn.Conv2d(hidden_channels * 2, hidden_channels, 1)
        self.fusion_HH = nn.Conv2d(hidden_channels * 2, hidden_channels, 1)

        self.conv_tail = Depth_conv(hidden_channels, rgb_channels)

    def forward(self, rgb_hf, event_feat, snr_map):
        """
        Args:
            rgb_hf: RGB high-frequency coefficients [3*B, C_rgb, H, W]
                     stacked as [HL; LH; HH] along batch dim
            event_feat: Event features at this wavelet scale [B, C_ev, H, W]
            snr_map: SNR map at this scale [B, 1, H, W], range [0,1]

        Returns:
            Restored HF coefficients [3*B, C_rgb, H, W]
        """
        b = rgb_hf.shape[0] // 3
        residual = rgb_hf

        # Split RGB HF into 3 subbands
        rgb_HL = self.rgb_head(rgb_hf[:b])
        rgb_LH = self.rgb_head(rgb_hf[b:2*b])
        rgb_HH = self.rgb_head(rgb_hf[2*b:])

        # Project event features
        ev = self.event_proj(event_feat)

        # Process RGB path
        rgb_HL = self.rgb_dilated_HL(rgb_HL)
        rgb_LH = self.rgb_dilated_LH(rgb_LH)
        rgb_HH = self.rgb_dilated_HH(rgb_HH)

        # Process Event path
        ev_HL = self.event_dilated_HL(ev)
        ev_LH = self.event_dilated_LH(ev)
        ev_HH = self.event_dilated_HH(ev)

        # Cross-attention: RGB structure enhanced by Event edges
        rgb_HL_att = self.cross_attn_HL(rgb_HL, ev_HL)
        rgb_LH_att = self.cross_attn_LH(rgb_LH, ev_LH)

        # HH: fuse event cross-refs for diagonal details
        hh_fused = self.cross_attn_HH_fuse(torch.cat([
            self.cross_attn_HL(rgb_HH, ev_HH),
            self.cross_attn_LH(rgb_HH, ev_HH)
        ], dim=1))

        # SNR-guided soft gating
        # High SNR -> trust RGB more; Low SNR -> trust Event more
        snr_weight = torch.sigmoid(snr_map * self.snr_temperature)  # [B,1,H,W]
        ev_weight = 1.0 - snr_weight

        # Fuse with SNR gating
        fused_HL = self.fusion_HL(torch.cat([
            rgb_HL_att * snr_weight + ev_HL * ev_weight,
            rgb_HL
        ], dim=1))
        fused_LH = self.fusion_LH(torch.cat([
            rgb_LH_att * snr_weight + ev_LH * ev_weight,
            rgb_LH
        ], dim=1))
        fused_HH = self.fusion_HH(torch.cat([
            hh_fused * snr_weight + ev_HH * ev_weight,
            rgb_HH
        ], dim=1))

        # Output
        out = self.conv_tail(torch.cat([fused_HL, fused_LH, fused_HH], dim=0))
        return out + residual
