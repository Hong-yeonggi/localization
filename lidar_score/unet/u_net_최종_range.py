#!/usr/bin/env python3
# coding: utf-8
"""
1‑D U‑Net  (입력 = range 1 채널, 길이 360)
- 학습 종료 후: ① Loss 곡선 출력  ② 테스트 예측 CSV 저장
"""

import os, json
import numpy as np
import pandas as pd
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

# ───────────────────────────
# 1.  네트워크 정의
# ───────────────────────────
class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding='same')
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding='same')
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.act   = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        return self.act(self.bn2(self.conv2(x)))

class UNet1D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_ch=64):
        super().__init__()
        self.enc1 = ConvBlock1D(in_channels, base_ch)
        self.pool1= nn.MaxPool1d(2)

        self.enc2 = ConvBlock1D(base_ch, base_ch*2)
        self.pool2= nn.MaxPool1d(2)

        self.enc3 = ConvBlock1D(base_ch*2, base_ch*4)
        self.pool3= nn.MaxPool1d(2)

        self.enc4 = ConvBlock1D(base_ch*4, base_ch*8)
        self.pool4= nn.MaxPool1d(2)

        self.bottleneck = ConvBlock1D(base_ch*8, base_ch*16)

        self.up4 = nn.ConvTranspose1d(base_ch*16, base_ch*8, 2, 2, output_padding=1)
        self.dec4= ConvBlock1D(base_ch*16, base_ch*8)

        self.up3 = nn.ConvTranspose1d(base_ch*8, base_ch*4, 2, 2)
        self.dec3= ConvBlock1D(base_ch*8, base_ch*4)

        self.up2 = nn.ConvTranspose1d(base_ch*4, base_ch*2, 2, 2)
        self.dec2= ConvBlock1D(base_ch*4, base_ch*2)

        self.up1 = nn.ConvTranspose1d(base_ch*2, base_ch, 2, 2)
        self.dec1= ConvBlock1D(base_ch*2, base_ch)

        self.outc = nn.Conv1d(base_ch, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x); p1 = self.pool1(e1)
        e2 = self.enc2(p1); p2 = self.pool2(e2)
        e3 = self.enc3(p2); p3 = self.pool3(e3)
        e4 = self.enc4(p3); p4 = self.pool4(e4)
        b  = self.bottleneck(p4)
        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.outc(d1)

# ───────────────────────────
# 2.  Dataset (Range 1‑채널)
# ───────────────────────────
class LidarBeamDataset(Dataset):
    def __init__(self, input_csv, score_csv, seq_len=360, stats_file=None):
        super().__init__()
        df_in = pd.read_csv(input_csv , index_col=0)
        df_sc = pd.read_csv(score_csv, index_col=0)
        df    = df_in.join(df_sc, how="inner")

        self.scan_times = df.index.astype(float).values
        rng, score = [], []
        for i in range(seq_len):
            rng  .append(df[f"{i}_range" ].values)
            score.append(df[f"{i}_score" ].values)
        range_np = np.stack(rng, 0).T      # (N,360)
        score_np = np.stack(score,0).T

        r_min, r_max = range_np.min(), range_np.max()
        range_norm   = (range_np - r_min) / (r_max - r_min + 1e-6)

        self.x = torch.from_numpy(range_norm[:,None,:]).float()   # (N,1,360)
        self.y = torch.from_numpy(score_np).float()
        self.stats_dict = {"r_min":float(r_min), "r_max":float(r_max)}
        if stats_file:
            with open(stats_file,"w") as f: json.dump(self.stats_dict,f,indent=2)
    def __len__(self):  return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], float(self.scan_times[idx])

# ───────────────────────────
# 3.  Train / Val / Test
# ───────────────────────────
def main():
    import torch.optim as optim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    input_csv = "/home/yeonggi/score_based_localization/lidar_score/unet/input_2N_final.csv"
    score_csv = "/home/yeonggi/score_based_localization/lidar_score/unet/score_2N_final.csv"

    batch_size, epochs, lr = 32, 400, 1e-3
    best_path  = "./best_unet_360_range.pth"
    stats_file = "./beam_stats_360_range.json"
    test_csv_out = "./test_pred_range.csv"      # ### NEW

    ds_full = LidarBeamDataset(input_csv, score_csv, 360, stats_file)
    r_min, r_max = ds_full.stats_dict["r_min"], ds_full.stats_dict["r_max"]  # ### NEW

    N = len(ds_full)
    n_tr, n_val = int(0.7*N), int(0.15*N)
    n_te = N - n_tr - n_val
    tr_ds, va_ds, te_ds = random_split(
        ds_full, [n_tr, n_val, n_te],
        torch.Generator().manual_seed(42)
    )
    tr_ld = DataLoader(tr_ds, batch_size, shuffle=True)
    va_ld = DataLoader(va_ds, batch_size, shuffle=False)
    te_ld = DataLoader(te_ds, batch_size, shuffle=False)

    print(f"Dataset: {N}  (train {n_tr} / val {n_val} / test {n_te})")
    print("Range stats:", ds_full.stats_dict)

    model  = UNet1D(1,1,64).to(device)
    optim_ = optim.Adam(model.parameters(), lr=lr)

    best_val = float('inf')
    train_losses, val_losses = [], []          # ### NEW

    # ───── 학습 루프 ─────
    for ep in range(1, epochs+1):
        # Train
        model.train(); run = 0.0
        for x,y,_ in tr_ld:
            x,y = x.to(device), y.to(device)
            optim_.zero_grad()
            loss = F.mse_loss(model(x).squeeze(1), y)
            loss.backward(); optim_.step()
            run += loss.item()
        tr_loss = run / len(tr_ld)

        # Val
        model.eval(); run = 0.0
        with torch.no_grad():
            for x,y,_ in va_ld:
                x,y = x.to(device), y.to(device)
                run += F.mse_loss(model(x).squeeze(1), y).item()
        va_loss = run / len(va_ld)

        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), best_path)

        train_losses.append(tr_loss); val_losses.append(va_loss)   # ### NEW
        print(f"[{ep:3d}/{epochs}]  train={tr_loss:.4f}   val={va_loss:.4f}")

    # ───── Loss 곡선 ─────
    plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses,   label="Val")
    plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
    plt.grid(True); plt.legend()
    #plt.title("U‑Net (Range only) Learning Curve")
    plt.show()

    # ───── Test & CSV 저장 ─────
    model.load_state_dict(torch.load(best_path)); model.eval()

    all_rows, test_loss = [], 0.0
    with torch.no_grad():
        for xT,yT,sT in te_ld:
            xT,yT = xT.to(device), yT.to(device)
            outT  = model(xT).squeeze(1)        # (B,360)
            test_loss += F.mse_loss(outT, yT).item()

            # ---- CSV 로우 누적 ----
            x_np = xT.cpu().numpy()             # (B,1,360)  (정규화 범위)
            y_np = yT.cpu().numpy()             # (B,360)
            p_np = outT.cpu().numpy()           # (B,360)
            s_np = sT.numpy()                   # (B,)
            B = x_np.shape[0]
            for b in range(B):
                scan_time = s_np[b]
                range_norm = x_np[b,0,:]                     # (360,)
                range_orig = range_norm*(r_max-r_min)+r_min  # 역정규화
                y_true  = y_np[b,:]
                y_pred  = p_np[b,:]
                for j in range(360):
                    all_rows.append({
                        "scan_time" : scan_time,
                        "beam_idx"  : j,
                        "range_m"   : float(range_orig[j]),
                        "score_true": float(y_true[j]),
                        "score_pred": float(y_pred[j]),
                    })

    test_loss /= len(te_ld)
    print(f"Test MSE = {test_loss:.4f}")

    df_out = pd.DataFrame(all_rows)
    df_out.to_csv(test_csv_out, index=False)
    print(f"Saved test prediction CSV → {test_csv_out}  (rows={len(df_out)})")

if __name__ == "__main__":
    main()
