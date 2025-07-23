#!/usr/bin/env python3
# coding: utf-8
"""
1‑D U‑Net for LiDAR beam sequence (length = 360, no mask)
- 입력 열 순서: 0_range, 0_intensity, …, 359_range, 359_intensity
- 채널 2개: (intensity, range)
- 출력 1개: 회귀 score
"""

import json, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
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
# ──────────────────────────────────────────────────────────────
# 2. Dataset (마스크 제거, 0 포함 정규화)
# ──────────────────────────────────────────────────────────────
class LidarBeamDataset(Dataset):
    """
    input_csv  : scan_time, 0_range,0_intensity, …, 359_range,359_intensity
    score_csv  : 0_score … 359_score
    반환 tensor:
        x : (N, 2, 360)   – 0채널 intensity, 1채널 range
        y : (N, 360)
        t : scan_time (float)
    """
    def __init__(self, input_csv:str, score_csv:str,
                 seq_len:int=360, stats_file:str|None=None):
        super().__init__()
        df_in = pd.read_csv(input_csv, index_col=0)
        df_sc = pd.read_csv(score_csv, index_col=0)
        df    = df_in.join(df_sc, how="inner")
        self.scan_times = df.index.astype(float).values
        intens, rng, score = [], [], []
        for i in range(seq_len):
            rng   .append(df[f"{i}_range"    ].values)
            intens.append(df[f"{i}_intensity"].values)
            score .append(df[f"{i}_score"    ].values)
        range_np  = np.stack(rng,    0).T     # (N,360)
        intens_np = np.stack(intens, 0).T
        score_np  = np.stack(score,  0).T
        # ── 전체 분포 기준 min/max (0 포함)
        i_min, i_max = intens_np.min(), intens_np.max()
        r_min, r_max = range_np .min(), range_np .max()
        intens_norm = (intens_np - i_min) / (i_max - i_min + 1e-6)
        range_norm  = (range_np  - r_min) / (r_max - r_min + 1e-6)
        self.x = torch.from_numpy(np.stack([intens_norm, range_norm], 1)).float()
        self.y = torch.from_numpy(score_np).float()
        self.stats_dict = {"i_min":float(i_min),"i_max":float(i_max),
                           "r_min":float(r_min),"r_max":float(r_max)}
        if stats_file:
            with open(stats_file, "w") as f: json.dump(self.stats_dict, f, indent=2)
    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx], self.y[idx], float(self.scan_times[idx])
# ──────────────────────────────────────────────────────────────
# 3. Train / Val / Test 루프 (동일, mask 관련 삭제)
# ──────────────────────────────────────────────────────────────
def main():
    import torch.optim as optim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    input_csv  = "/home/yeonggi/tb3/model/cnn/input/input_2N_final.csv"
    score_csv  = "/home/yeonggi/tb3/model/cnn/input/score_2N_final.csv"
    seq_len    = 360
    batch_size, epochs, lr = 32, 400, 1e-3
    best_path  = "./best_unet_360.pth"
    stats_file = "./beam_stats_360.json"

    ds_full = LidarBeamDataset(input_csv, score_csv, seq_len, stats_file)
    N = len(ds_full)
    n_train, n_val = int(0.7*N), int(0.15*N)
    n_test = N - n_train - n_val
    tr_ds, va_ds, te_ds = random_split(ds_full, [n_train, n_val, n_test],
                                       generator=torch.Generator().manual_seed(42))
    tr_ld = DataLoader(tr_ds, batch_size, shuffle=True)
    va_ld = DataLoader(va_ds, batch_size, shuffle=False)
    te_ld = DataLoader(te_ds, batch_size, shuffle=False)
    print(f"Dataset: total={N} (train {n_train} / val {n_val} / test {n_test})")
    print("Stats :", ds_full.stats_dict)

    model = UNet1D(2,1,64).to(device)
    optim_ = optim.Adam(model.parameters(), lr=lr)
    best_val, tr_hist, va_hist = float('inf'), [], []

    for ep in range(1, epochs+1):
        # ── Train ─────────────────────────────────────────────
        model.train(); run = 0.0
        for x,y,_ in tr_ld:
            x,y = x.to(device), y.to(device)
            optim_.zero_grad()
            loss = F.mse_loss(model(x).squeeze(1), y)
            loss.backward(); optim_.step()
            run += loss.item()
        tr_loss = run / len(tr_ld)
        # ── Val ───────────────────────────────────────────────
        model.eval(); run = 0.0
        with torch.no_grad():
            for x,y,_ in va_ld:
                x,y = x.to(device), y.to(device)
                run += F.mse_loss(model(x).squeeze(1), y).item()
        va_loss = run / len(va_ld)
        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), best_path)
        tr_hist.append(tr_loss); va_hist.append(va_loss)
        print(f"[{ep:3d}/{epochs}] train={tr_loss:.4f}  val={va_loss:.4f}")

    # ── Loss plot ─────────────────────────────────────────────
    plt.figure(); plt.plot(tr_hist,label="Train"); plt.plot(va_hist,label="Val")
    plt.grid(True); plt.legend(); plt.show()

    # ── Test ────────────────────────────────────────────────
    model.load_state_dict(torch.load(best_path)); model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for x,y,_ in te_ld:
            x,y = x.to(device), y.to(device)
            test_loss += F.mse_loss(model(x).squeeze(1), y).item()
    print(f"Test MSE = {test_loss/len(te_ld):.4f}")

if __name__ == "__main__":
    main()
