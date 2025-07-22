
import torch
import torch.nn as nn
# ──────────────────────────────────────────────────────────────
# 1. ConvBlock & UNet (기존과 동일)
# ──────────────────────────────────────────────────────────────
class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding='same')
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding='same')
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(x)))

class UNet1D(nn.Module):
    """down: 360→180→90→45→22 , up: 22→45→90→180→360"""
    def __init__(self, in_channels=2, out_channels=1, base_ch=64):
        super().__init__()
        self.enc1 = ConvBlock1D(in_channels, base_ch)
        self.pool1= nn.MaxPool1d(2)
        
        self.enc2 = ConvBlock1D(base_ch, base_ch*2)
        self.pool2= nn.MaxPool1d(2)
        
        self.enc3 = ConvBlock1D(base_ch*2, base_ch*4)
        self.pool3= nn.MaxPool1d(2)
        
        self.enc4 = ConvBlock1D(base_ch*4, base_ch*8)
        self.pool4= nn.MaxPool1d(2)          # 45→22
        
        self.bottleneck = ConvBlock1D(base_ch*8, base_ch*16)
        
        self.upconv4 = nn.ConvTranspose1d(base_ch*16, base_ch*8, 2, 2, output_padding=1)
        self.dec4    = ConvBlock1D(base_ch*16, base_ch*8)
        
        self.upconv3 = nn.ConvTranspose1d(base_ch*8, base_ch*4, 2, 2)
        self.dec3    = ConvBlock1D(base_ch*8, base_ch*4)
        
        self.upconv2 = nn.ConvTranspose1d(base_ch*4, base_ch*2, 2, 2)
        self.dec2    = ConvBlock1D(base_ch*4, base_ch*2)
        
        self.upconv1 = nn.ConvTranspose1d(base_ch*2, base_ch, 2, 2)
        self.dec1    = ConvBlock1D(base_ch*2, base_ch)
        
        self.outconv = nn.Conv1d(base_ch, out_channels, 1)
        
    def forward(self, x):
        e1 = self.enc1(x); p1 = self.pool1(e1)
        e2 = self.enc2(p1); p2 = self.pool2(e2)
        e3 = self.enc3(p2); p3 = self.pool3(e3)
        e4 = self.enc4(p3); p4 = self.pool4(e4)
        b  = self.bottleneck(p4)
        d4 = self.dec4(torch.cat([self.upconv4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.upconv3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.upconv2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.upconv1(d2), e1], 1))
        return self.outconv(d1)
