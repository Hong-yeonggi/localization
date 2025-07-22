#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data

import math, json, csv
import numpy as np
import torch
from sensor_msgs.msg import LaserScan
import matplotlib.pyplot as plt

from model_unet_최종 import UNet1D          # 360‑빔용 U‑Net

# ─────────── 설정 ───────────
SEQ_LEN         = 360
RANGE_MAX_MODEL = 100.0

STATS_FILE      = "./beam_stats_360.json"
MODEL_PATH      = "./best_unet_360.pth"
CSV_LOG_PATH    = "result.csv"

# 학습 시 저장해둔 min/max 로드
with open(STATS_FILE) as f:
    stats = json.load(f)
MIN_INT, MAX_INT = stats["i_min"], stats["i_max"]
MIN_RNG, MAX_RNG = stats["r_min"], stats["r_max"]

# ────────────────────────────
def normalize(int_arr, rng_arr):
    """min‑max 스케일 (0 포함). 반환 dtype=float32"""
    i_norm = (int_arr - MIN_INT) / (MAX_INT - MIN_INT + 1e-6)
    r_norm = (rng_arr - MIN_RNG) / (MAX_RNG - MIN_RNG + 1e-6)
    return i_norm.astype(np.float32), r_norm.astype(np.float32)

# ────────────────────────────
class StaticScanNode(Node):
    def __init__(self):
        super().__init__("static_scan_node")

        # (A) 모델
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = UNet1D(2,1,64).to(self.device)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.model.eval()
        self.get_logger().info("U‑Net model loaded.")

        # (B) QoS / 토픽
        qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, depth=1)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)
        self.pub_static = self.create_publisher(LaserScan, '/static_scan', qos)

        # (C) CSV
        self.csv_fp = open(CSV_LOG_PATH, "w", newline="")
        self.csv_wr = csv.writer(self.csv_fp)
        self.csv_wr.writerow(["stamp_sec","beam_idx","range","intensity","pred_score"])

        # (D) 실시간 시각화
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8,8))
        self.scat = self.ax.scatter([], [], c=[], vmin=0.0, vmax=1.0, cmap="jet", s=18)
        self.fig.colorbar(self.scat, ax=self.ax).set_label("Pred_Score")
        self.ax.set_aspect("equal")
        self.ax.plot(0,0,"kx"); self.ax.set_xlim(-15,15); self.ax.set_ylim(-15,15)
        self.fig.tight_layout()

    # ────────── 콜백 ──────────
    # ────────── 콜백 ──────────
    def scan_callback(self, msg: LaserScan):
        # ①  원본 데이터 그대로 보관 (NaN 포함)
        r_raw = np.asarray(msg.ranges,      dtype=np.float32)   # ← 그대로 퍼블리시할 값
        i_raw = np.asarray(msg.intensities, dtype=np.float32)

        # ②  모델 입력용 배열 복사  (무효값 → 0)
        r_proc = r_raw.copy()
        i_proc = i_raw.copy()

        invalid_r = (~np.isfinite(r_proc)) | (r_proc < 0) | (r_proc > RANGE_MAX_MODEL)
        r_proc[invalid_r] = 0.0
        i_proc[~np.isfinite(i_proc)] = 0.0

        # (2) 정규화
        i_norm, r_norm = normalize(i_proc, r_proc)

        # (3) 추론
        x = torch.from_numpy(np.stack([i_norm, r_norm], 0)).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            pred = self.model(x).cpu().numpy().squeeze()   # (360,)

        # (4) CSV 기록  ── 원본(NaN 포함) 사용
        stamp = msg.header.stamp.sec
        for b in range(SEQ_LEN):
            self.csv_wr.writerow([stamp, b,
                                  float(r_raw[b]), float(i_raw[b]),  # ← r_raw, i_raw
                                  float(pred[b])])
        self.csv_fp.flush()

        # (5) /static_scan 발행  ── NaN 복원
        import math
        ints_out = np.clip(pred, 0.0, 1.0).astype(np.float32)
        nan_mask = (~np.isfinite(r_raw)) | (~np.isfinite(i_raw))
        ints_out[nan_mask] = math.nan                       ### 변경

        ls_out = LaserScan()
        ls_out.header          = msg.header
        ls_out.angle_min       = msg.angle_min
        ls_out.angle_increment = msg.angle_increment
        ls_out.angle_max       = msg.angle_max
        ls_out.time_increment  = msg.time_increment
        ls_out.scan_time       = msg.scan_time
        ls_out.range_min       = msg.range_min
        ls_out.range_max       = msg.range_max
        ls_out.ranges          = r_raw.tolist()              ### NaN 그대로
        ls_out.intensities     = ints_out.tolist()
        self.pub_static.publish(ls_out)

        # (6) 시각화  ── 0 처리된 r_proc 활용
        angles = msg.angle_min + np.arange(SEQ_LEN) * msg.angle_increment
        valid  = r_proc > 0.0
        pts    = np.column_stack((r_proc[valid]*np.cos(angles[valid]),
                                  r_proc[valid]*np.sin(angles[valid])))
        self.scat.set_offsets(pts)
        self.scat.set_array(pred[valid])
        self.ax.set_title(f"sec={stamp}")
        plt.draw(); plt.pause(0.001)


    # ────────── 종료 ──────────
    def destroy_node(self):
        if self.csv_fp: self.csv_fp.close()
        super().destroy_node()

# ────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = StaticScanNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        plt.ioff(); plt.show()

if __name__ == "__main__":
    main()
