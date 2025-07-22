#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import math
import numpy as np
import torch
import torch.nn.functional as F
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

from sensor_msgs.msg import LaserScan

from model_unet import UNet1D


MU_INT, STD_INT = 193.48711790601712, 59.43788363862114
MIN_INT, MAX_INT= -2.1785283694615183, 0.7825460542005909

MU_RNG,  STD_RNG= 3.914726674776115, 4.048242314156207
MIN_RNG, MAX_RNG= -0.9272482861022202, 15.186160651521476


range_max_model = 100.0
seq_len         = 252
threshold_score = 0.8

placeholder_range = 0.0  # 무효빔 => 0

#=====================================================
# 1) pad_and_mask_scan
#=====================================================
def pad_and_mask_scan(ranges, intensities, range_max=100.0):
    """
    1) 만약 N <= 252 => 남은 부분은 pad=0, mask=0
       (훈련 시 로직: Inf/NaN → 값=0, mask=1)
    2) 만약 N > 252 => 균일 간격으로 '다운샘플링'하여 252 길이로 맞춤
       (Inf/NaN → 값=0, mask=1)
    """
    seq_len = 252
    intens_out = np.zeros(seq_len, dtype=np.float32)
    range_out  = np.zeros(seq_len, dtype=np.float32)
    mask_out   = np.zeros(seq_len, dtype=np.float32)

    N = len(ranges)
    if N == 0:
        # 빔이 하나도 없는 경우 => 이미 0 초기화 (mask=0)
        return intens_out, range_out, mask_out

    if N <= seq_len:
        #-----------#
        #  (1) N <= 252
        #-----------#
        # 0~(N-1) 빔은 그대로 복사 + mask=1
        for i in range(N):
            mask_out[i] = 1.0

            r = ranges[i]
            it = intensities[i]
            # inf/nan/음수/큰값 => 값=0, mask=1 유지
            if (math.isinf(r) or math.isnan(r) or r < 0 or r > range_max):
                range_out[i] = 0.0
            else:
                range_out[i] = r

            if (math.isinf(it) or math.isnan(it)):
                intens_out[i] = 0.0
            else:
                intens_out[i] = it

        # i=N..251 => 이미 0 초기화(mask=0, 값=0)
    else:
        #-----------#
        #  (2) N > 252 => 다운샘플링
        #-----------#
        # 예: idxs = [0, 0.99, 1.98, ... N-1] → 정수화
        idxs = np.linspace(0, N-1, seq_len).astype(int)

        for i in range(seq_len):
            mask_out[i] = 1.0

            r = ranges[idxs[i]]
            it = intensities[idxs[i]]
            # inf/nan/음수/큰값 => 값=0, mask=1 유지
            if (math.isinf(r) or math.isnan(r) or r < 0 or r > range_max):
                range_out[i] = 0.0
            else:
                range_out[i] = r

            if (math.isinf(it) or math.isnan(it)):
                intens_out[i] = 0.0
            else:
                intens_out[i] = it

    return intens_out, range_out, mask_out

#=====================================================
# 2) apply_norm_inference (유효빔만 변환)
#=====================================================
def apply_norm_inference(i_arr, r_arr, m_arr):
    """
    i_arr, r_arr, m_arr: (seq_len,)
    mask=1 위치만 Z-score => MinMax
    """
    valid = (m_arr>0)
    i_valid = i_arr[valid]
    r_valid = r_arr[valid]

    # Z-score
    i_z = (i_valid - MU_INT)/(STD_INT+1e-6)
    r_z = (r_valid - MU_RNG)/(STD_RNG+1e-6)

    # MinMax
    i_norm = (i_z - MIN_INT)/(MAX_INT - MIN_INT +1e-6)
    r_norm = (r_z - MIN_RNG)/(MAX_RNG - MIN_RNG +1e-6)

    # 원래 자리에 복원
    i_out = i_arr.copy()
    r_out = r_arr.copy()
    i_out[valid] = i_norm
    r_out[valid] = r_norm

    return i_out, r_out

#=====================================================
# ROS Node
#=====================================================
class StaticScanNode(Node):
    def __init__(self):
        super().__init__('static_scan_node')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # U-Net 모델 로드
        self.model = UNet1D(in_channels=2, out_channels=1, base_ch=64).to(self.device)
        self.model.load_state_dict(torch.load("best_unet.pth", map_location=self.device))
        self.model.eval()

        # Subscriber : /scan → callback
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            depth=1
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos_profile)

        # /static_scan 퍼블리시
        self.static_pub = self.create_publisher(LaserScan, '/static_scan', qos_profile)
        self.static_pub = self.create_publisher(LaserScan, '/static_scan', 10)

        self.get_logger().info("static_scan_node (U-Net) initialized.")

    def scan_callback(self, msg: LaserScan):
        # 1) ranges, intens => numpy
        ranges_np      = np.array(msg.ranges, dtype=np.float32)
        intensities_np = np.array(msg.intensities, dtype=np.float32)
        stamp_sec      = msg.header.stamp.sec

        # 2) pad+mask
        i_arr, r_arr, m_arr = pad_and_mask_scan(ranges_np, intensities_np, range_max_model)

        # 3) 표준화/정규화 => (유효빔만)
        i_norm, r_norm = apply_norm_inference(i_arr, r_arr, m_arr)

        # 4) tensor => forward
        x_np = np.stack([i_norm, r_norm], axis=0)  # (2,252)
        x_ts = torch.from_numpy(x_np).unsqueeze(0).float().to(self.device)  # (1,2,252)
        m_ts = torch.from_numpy(m_arr).unsqueeze(0).float().to(self.device) # (1,252)

        with torch.no_grad():
            pred_ts = self.model(x_ts, mask=m_ts)  # => (1,252)
        pred_np = pred_ts.squeeze(0).cpu().numpy() # => (252,)

        # 5) threshold => out_ranges / out_intens
        out_ranges      = np.full((seq_len,), float('inf'), dtype=np.float32)
        out_intensities = np.zeros((seq_len,), dtype=np.float32)

        # pred_np[i]>=threshold_score → 유효
        for i in range(seq_len):
            if pred_np[i] >= threshold_score:
                out_ranges[i]      = r_arr[i]
                out_intensities[i] = i_arr[i]

        # 6) LaserScan 생성
        static_scan = LaserScan()
        static_scan.header = msg.header  # stamp, frame_id 동일

        static_scan.angle_min       = msg.angle_min
        static_scan.angle_max       = msg.angle_max
        static_scan.angle_increment = msg.angle_increment
        static_scan.time_increment  = msg.time_increment
        static_scan.scan_time       = msg.scan_time
        static_scan.range_min       = msg.range_min
        static_scan.range_max       = msg.range_max

        static_scan.ranges      = out_ranges.tolist()
        static_scan.intensities = out_intensities.tolist()

        self.static_pub.publish(static_scan)

        # 로그 예시
        kept_count = (pred_np >= threshold_score).sum()
        self.get_logger().info(
            f"Published /static_scan stamp={stamp_sec}, #beams>={threshold_score} = {kept_count}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = StaticScanNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=="__main__":
    main()