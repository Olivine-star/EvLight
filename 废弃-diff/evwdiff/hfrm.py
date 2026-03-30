"""
High-Frequency Restoration Module (HFRM).
Adapted from DiffLL (Jiang et al., 2023).
"""
import math
import torch
import torch.nn as nn


class Depth_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depth_conv = nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch)
        self.point_conv = nn.Conv2d(in_ch, out_ch, 1, 1, 0)

    def forward(self, x):
        return self.point_conv(self.depth_conv(x))


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        self.attention_head_size = dim // num_heads
        self.query = Depth_conv(dim, dim)
        self.key = Depth_conv(dim, dim)
        self.value = Depth_conv(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, ctx):
        q = self.query(hidden_states).permute(0, 2, 1, 3)
        k = self.key(ctx).permute(0, 2, 1, 3)
        v = self.value(ctx).permute(0, 2, 1, 3)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        probs = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(probs, v).permute(0, 2, 1, 3).contiguous()
        return out


class Dilated_Resblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, dilation=1), nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 2, dilation=2), nn.LeakyReLU(),
            nn.Conv2d(out_channels, in_channels, 3, 1, 3, dilation=3), nn.LeakyReLU(),
            nn.Conv2d(in_channels, in_channels, 3, 1, 2, dilation=2), nn.LeakyReLU(),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, dilation=1),
        )

    def forward(self, x):
        return self.model(x) + x


class HFRM(nn.Module):
    """High-Frequency Restoration Module from DiffLL."""
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.conv_head = Depth_conv(in_channels, out_channels)
        self.dilated_block_LH = Dilated_Resblock(out_channels, out_channels)
        self.dilated_block_HL = Dilated_Resblock(out_channels, out_channels)
        self.cross_attention0 = CrossAttention(out_channels, num_heads=8)
        self.dilated_block_HH = Dilated_Resblock(out_channels, out_channels)
        self.conv_HH = nn.Conv2d(out_channels * 2, out_channels, 3, 1, 1)
        self.cross_attention1 = CrossAttention(out_channels, num_heads=8)
        self.conv_tail = Depth_conv(out_channels, in_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        residual = x
        x = self.conv_head(x)
        x_HL = x[:b // 3, ...]
        x_LH = x[b // 3:2 * b // 3, ...]
        x_HH = x[2 * b // 3:, ...]

        x_HH_LH = self.cross_attention0(x_LH, x_HH)
        x_HH_HL = self.cross_attention1(x_HL, x_HH)

        x_HL = self.dilated_block_HL(x_HL)
        x_LH = self.dilated_block_LH(x_LH)
        x_HH = self.dilated_block_HH(self.conv_HH(torch.cat((x_HH_LH, x_HH_HL), dim=1)))

        out = self.conv_tail(torch.cat((x_HL, x_LH, x_HH), dim=0))
        return out + residual
