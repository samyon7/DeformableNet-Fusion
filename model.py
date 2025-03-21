import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d  # Import deformable conv
import numpy as np

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, window_size=8, shift_size=0, num_heads=8):
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=window_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # Padding size
        if self.shift_size > 0:
            pad_l = pad_t = 0
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
            _, _, Hp, Wp = x.shape
        else:
            Hp, Wp = H, W

        # Window Partitioning
        x = x.unfold(2, self.window_size, self.window_size).unfold(3, self.window_size, self.window_size).permute(0, 2, 3, 4, 5, 1).contiguous() # (B, H//window_size, W//window_size, window_size, window_size, C)
        x = x.view(-1, self.window_size * self.window_size, C) # (B*H//window_size*W//window_size, window_size*window_size, C)

        # Windowed Attention
        x = self.norm1(x)
        x = self.attn(x)
        x = self.norm2(x)
        x = self.ffn(x)

        x = x.view(B, Hp // self.window_size, Wp // self.window_size, self.window_size, self.window_size, C).permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, Hp, Wp, C)  # (B, H, W, C)

        # Unpad if needed
        if self.shift_size > 0:
            x = x[:, :H, :W, :]
        x = x.permute(0, 3, 1, 2)
        return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads=8):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape  # N = window_size * window_size
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


# Deformable Convolution Block
class DeformConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeformConvBlock, self).__init__()
        self.offset_conv = nn.Conv2d(in_channels, 18, kernel_size=3, padding=1)
        self.deform_conv = DeformConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        offset = self.offset_conv(x)
        x = self.deform_conv(x, offset)
        x = self.norm(x)
        return self.relu(x)


# Cross Attention Transformer Module (CATM)
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(CrossAttention, self).__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
        self.conv = nn.Conv2d(dim, dim // 2, kernel_size=1) 

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        x1 = x1.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x2 = x2.flatten(2).transpose(1, 2)  # (B, H*W, C)
        out, _ = self.attn(x1, x2, x2)  # x1 is query, x2 is key and value
        out = out.transpose(1, 2).reshape(B, C, H, W)  # Kembali ke (B, C, H, W)
        out = self.conv(out)  
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
        return out


# Adaptive Fusion Block (AFB)
class AdaptiveFusionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AdaptiveFusionBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        x = self.conv1(x1) + self.conv2(x2)
        x = self.norm(x)
        return self.relu(x)


# DeformableNetFusion Architecture
class DeformableNetFusion(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, swin_dim=256):
        super(DeformableNetFusion, self).__init__()

        # Encoder
        self.enc1 = DeformConvBlock(in_channels, 64)
        self.enc2 = DeformConvBlock(64, 128)
        self.enc3 = DeformConvBlock(128, 256)

        # Transformer bottleneck
        self.trans = SwinTransformerBlock(dim=swin_dim) # Ensure the dimension is fits

        # Decoder
        self.dec3 = AdaptiveFusionBlock(128)
        self.dec2 = AdaptiveFusionBlock(128)
        self.dec1 = AdaptiveFusionBlock(64)

        # Cross-Attention
        self.cross_attn1 = CrossAttention(256)
        self.cross_attn2 = CrossAttention(128)

        # Upsampling
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        # Output Layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, kernel_size=2))
        e3 = self.enc3(F.max_pool2d(e2, kernel_size=2))

        # Transformer bottleneck
        t = self.trans(e3)

        # Decoder
        d3 = self.dec3(self.up3(t), self.cross_attn1(e3, t))
        d2 = self.dec2(self.up2(d3), self.cross_attn2(e2, d3))
        d1 = self.dec1(d2, e1)

        # Output
        out = self.final_conv(d1)
        return self.sigmoid(out)
