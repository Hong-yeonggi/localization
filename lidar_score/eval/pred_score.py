#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
/initialpose  → 로봇 위치 저장
/scan         → U‑Net 예측 + Truth 계산 (거리 기반 e^{-d})
마우스 클릭   → CSV + PNG 4장(pred/truth × base/slam) 저장
"""

import rclpy, json, csv, cv2, numpy as np, torch, math, yaml
np.float = float
from rclpy.node import Node
from rclpy.qos  import qos_profile_sensor_data, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped
import tf_transformations as tft
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
from model_unet_최종 import UNet1D
import os
# ───────── 사용자 설정 ─────────
SEQ_LEN = 360
RANGE_MAX_MODEL = 100.0
POINT_RADIUS = 1
STATS_FILE   = "./beam_stats_360_2.json"
MODEL_PATH   = "./best_unet_360_2.pth"
SAVE_DIR = "/home/csilab/tb3/Lidar_score/eval"
MAP_PRED_DIR   = os.path.join(SAVE_DIR, "map_pred")
MAP_TRUE_DIR   = os.path.join(SAVE_DIR, "map_true")
SLAM_PRED_DIR  = os.path.join(SAVE_DIR, "slam_pred")
SLAM_TRUE_DIR  = os.path.join(SAVE_DIR, "slam_true")
BASE_MAP_PATH   = "./map_mod_noglass.png"   # 건축 도면(회색)
MAP_RES_M_PER_PX = 0.05                     # m / pixel
ORIGIN_X, ORIGIN_Y = 227, 681               # (0,0) → 픽셀

SLAM_MAP_PGM  = "./5_floor.pgm"
SLAM_MAP_YAML = "./5_floor.yaml"

# ───── Min‑Max 정규화 파라미터 ─────
with open(STATS_FILE) as f:
    st = json.load(f)
MIN_INT, MAX_INT = st["i_min"], st["i_max"]
MIN_RNG, MAX_RNG = st["r_min"], st["r_max"]
def normalize(i_arr, r_arr):
    i = (i_arr - MIN_INT) / (MAX_INT - MIN_INT + 1e-6)
    r = (r_arr - MIN_RNG) / (MAX_RNG - MIN_RNG + 1e-6)
    return i.astype(np.float32), r.astype(np.float32)

# ───────── 1) 도면 → Distance‑Transform Truth 맵 ─────────
if Path(BASE_MAP_PATH).exists():
    # (a) 회색 읽기
    gray = cv2.imread(BASE_MAP_PATH, cv2.IMREAD_GRAYSCALE)

    # (b) 벽(선) 마스크  : 픽셀<254 ⇒ 벽,  ≥254 ⇒ 자유공간
    line_mask = np.where(gray < 254, 0, 255).astype(np.uint8)

    # (c) 거리변환 — 배경 픽셀마다 “최근접 벽까지 거리(px)”
    dist_img_px = cv2.distanceTransform(line_mask, cv2.DIST_L2, 3)  # float32

    H, W = dist_img_px.shape

    def truth_lookup(x_m: float, y_m: float) -> float:
        """월드[m] → Truth Score( e^{-distance[m]} )."""
        u = int(round(ORIGIN_X +  x_m / MAP_RES_M_PER_PX))
        v = int(round(ORIGIN_Y -  y_m / MAP_RES_M_PER_PX))
        if 0 <= u < W and 0 <= v < H:
            dist_m = dist_img_px[v, u] * MAP_RES_M_PER_PX
            return math.exp(-dist_m)
        return 0.0
else:
    truth_lookup = lambda _x, _y: 0.0
    H = W = None

# (도면 컬러판 준비 – 오버레이용)
base_map_color = (cv2.cvtColor(cv2.imread(BASE_MAP_PATH, 0),
                               cv2.COLOR_GRAY2BGR)
                  if Path(BASE_MAP_PATH).exists() else None)

# ───────── 2) SLAM 맵 로드 (오버레이용) ─────────
slam_bgr = None
if Path(SLAM_MAP_PGM).exists() and Path(SLAM_MAP_YAML).exists():
    with open(SLAM_MAP_YAML) as f:
        yml = yaml.safe_load(f)
    slam_reso   = float(yml["resolution"])          # m / px
    slam_origin = yml["origin"]                     # [ox, oy, θ]
    slam_gray   = cv2.imread(SLAM_MAP_PGM, cv2.IMREAD_GRAYSCALE)
    slam_bgr    = cv2.cvtColor(slam_gray, cv2.COLOR_GRAY2BGR)
    SH, SW      = slam_bgr.shape[:2]

# ───────── 3) ROS 노드 정의 ─────────
class StaticScanNode(Node):
    def __init__(self):
        super().__init__("static_scan_node")

        # 모델
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = UNet1D(2, 1, 64).to(self.device)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.model.eval()
        self.get_logger().info("U‑Net 모델 로드")

        # 상태
        self.robot_pose = None
        self.last_scan  = None   # 클릭 시 파일 저장용 캐시

        # Sub/Pub
        self.create_subscription(PoseWithCovarianceStamped, "/initialpose",
                                 self.cb_pose, qos_profile_sensor_data)
        self.create_subscription(LaserScan, "/scan",
                                 self.cb_scan, qos_profile_sensor_data)
        qos_pub = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, depth=1)
        self.pub_static = self.create_publisher(LaserScan, "/static_scan", qos_pub)

        # 실시간 산점도
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.scat = self.ax.scatter([], [], c=[], vmin=0, vmax=1, cmap="jet", s=18)
        self.fig.colorbar(self.scat, ax=self.ax).set_label("Pred_Score")
        self.ax.set_aspect("equal")
        self.ax.plot(0, 0, "kx")
        self.ax.set_xlim(-15, 15); self.ax.set_ylim(-15, 15)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.get_logger().info("클릭 시 결과 저장")

    # ─── /initialpose ───
    def cb_pose(self, msg: PoseWithCovarianceStamped):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        _, _, yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.robot_pose = (p.x, p.y, yaw)          # 최신 pose 보관
        # self.last_scan["pose"] 업데이트는 **제거** (on_click에서 새롭게 사용)

    # ─── /scan ───
    def cb_scan(self, msg: LaserScan):
        if self.robot_pose is None:
            self.get_logger().warn("초기 위치(/initialpose) 대기 중"); return
        rx, ry, ryaw = self.robot_pose

        r_raw = np.asarray(msg.ranges, dtype=np.float32)
        i_raw = np.asarray(msg.intensities, dtype=np.float32)
        if i_raw.size == 0: i_raw = np.zeros_like(r_raw)
        if r_raw.size != SEQ_LEN:                    # 패딩
            pad = SEQ_LEN - r_raw.size
            r_raw = np.pad(r_raw, (0, pad))[:SEQ_LEN]
            i_raw = np.pad(i_raw, (0, pad))[:SEQ_LEN]

        # NaN → 0
        r_in = r_raw.copy()
        invalid = (~np.isfinite(r_in)) | (r_in < 0) | (r_in > RANGE_MAX_MODEL)
        r_in[invalid] = 0.0
        i_in = i_raw.copy(); i_in[~np.isfinite(i_in)] = 0.0

        # U‑Net 추론
        i_n, r_n = normalize(i_in, r_in)
        x_ts = torch.from_numpy(np.stack([i_n, r_n], 0)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.model(x_ts).cpu().numpy().squeeze()   # (360,)

        # Truth Score (거리 기반 e^{-d})
        theta = msg.angle_min + np.arange(SEQ_LEN) * msg.angle_increment
        xs = rx + r_in * np.cos(theta + ryaw)
        ys = ry + r_in * np.sin(theta + ryaw)

        truth = np.array([truth_lookup(x, y) for x, y in zip(xs, ys)], np.float32)
        mae   = float(np.abs(pred - truth)[~invalid].mean()) if (~invalid).any() else 0.0

        # MAE
        valid = ~invalid
        mae   = float(np.abs(pred - truth)[valid].mean()) if valid.any() else 0.0

        # 산점도 갱신
        pts = np.column_stack((r_in[valid] * np.cos(theta[valid]),
                               r_in[valid] * np.sin(theta[valid])))
        self.scat.set_offsets(pts); self.scat.set_array(pred[valid])
        plt.draw(); plt.pause(0.001)

        # /static_scan 발행
        ls = LaserScan()
        ls.header = msg.header
        ls.angle_min, ls.angle_increment, ls.angle_max = \
            msg.angle_min, msg.angle_increment, msg.angle_max
        ls.time_increment, ls.scan_time = msg.time_increment, msg.scan_time
        ls.range_min, ls.range_max = msg.range_min, msg.range_max
        ls.ranges      = r_raw.tolist()
        ls.intensities = np.clip(pred, 0, 1).astype(np.float32).tolist()
        self.pub_static.publish(ls)


        # ── 클릭용 캐시 (pose, xs·ys·truth 는 저장해 두지만 on_click 에서 갱신 가능)
        self.last_scan = dict(
            stamp   = msg.header.stamp.sec,
            r_raw   = r_raw, i_raw=i_raw,
            r_in    = r_in,  theta=theta,        # ‼ 추가
            pred    = pred,
            invalid = invalid,
            pose_at_scan = self.robot_pose,      # 스캔 찍힐 때 pose
        )

    # ─── 클릭: CSV + PNG ───
    def on_click(self, _):
        if self.last_scan is None:
            self.get_logger().warn("저장할 스캔 없음"); return

        # 1) ‘현재’ pose 로 Truth, 좌표, MAE 재계산
        if self.robot_pose is None:
            self.get_logger().warn("/initialpose 가 아직 없음"); return
        rx, ry, ryaw = self.robot_pose

        r_in   = self.last_scan["r_in"]
        theta  = self.last_scan["theta"]
        xs     = rx + r_in * np.cos(theta + ryaw)
        ys     = ry + r_in * np.sin(theta + ryaw)
        truth  = np.array([truth_lookup(x, y) for x, y in zip(xs, ys)], np.float32)

        pred      = self.last_scan["pred"]
        invalid   = self.last_scan["invalid"]
        r_raw     = self.last_scan["r_raw"]
        i_raw     = self.last_scan["i_raw"]
        mae       = float(np.abs(pred - truth)[~invalid].mean()) if (~invalid).any() else 0.0
        stamp     = self.last_scan["stamp"]

        # 2) CSV 저장 (같은 이름 규칙)
        with open(f"scan_{stamp}.csv", "w", newline="") as fp:
            wr = csv.writer(fp)
            wr.writerow(["stamp","beam","x","y","yaw","range","intensity","pred","truth","err"])
            for b in range(SEQ_LEN):
                if invalid[b]: continue
                wr.writerow([stamp, b, rx, ry, ryaw,
                             r_raw[b], i_raw[b], pred[b], truth[b],
                             abs(pred[b]-truth[b])])

        # 3) PNG 오버레이 (함수 호출 시 최신 xs, ys, truth 사용)
        self.save_overlay(stamp, xs, ys, pred, truth, invalid, rx, ry)

        self.get_logger().info(f"scan_{stamp}.csv & PNG 저장  | MAE={mae:.4f}")

    # ───── 오버레이 저장 도우미 ─────
    def save_overlay(self, t, xs, ys, pred, truth, invalid, rx, ry):
        cmap = cm.get_cmap("jet")

        # (a) 건축 도면
        if base_map_color is not None:
            for tag, scores in [("pred", pred), ("truth", truth)]:
                ov = base_map_color.copy()
                colors = (cmap(scores)[:, :3] * 255).astype(np.uint8)

                u = np.round(ORIGIN_X + xs / MAP_RES_M_PER_PX).astype(int)
                v = np.round(ORIGIN_Y - ys / MAP_RES_M_PER_PX).astype(int)
                inside = (u >= 0) & (u < ov.shape[1]) & \
                        (v >= 0) & (v < ov.shape[0]) & (~invalid)

                for uu, vv, col in zip(u[inside], v[inside], colors[inside]):
                    cv2.circle(ov, (uu, vv), POINT_RADIUS,
                            color=tuple(int(c) for c in col[::-1]),  # RGB→BGR
                            thickness=-1, lineType=cv2.LINE_AA)

                # 로봇 위치 (검은색 ×)
                cx = int(round(ORIGIN_X + rx / MAP_RES_M_PER_PX))
                cy = int(round(ORIGIN_Y - ry / MAP_RES_M_PER_PX))
                cv2.drawMarker(ov, (cx, cy), (0, 0, 0),
                            markerType=cv2.MARKER_CROSS, markerSize=7, thickness=1)

                cv2.imwrite(f"{tag}_{t}.png", ov)

        # ---------- (b) SLAM 맵 ----------
        if slam_bgr is not None:
            ox, oy, _ = slam_origin
            for tag, scores in [("pred", pred), ("truth", truth)]:
                ov = slam_bgr.copy()
                colors = (cmap(scores)[:, :3] * 255).astype(np.uint8)

                u = np.round((xs - ox) / slam_reso).astype(int)
                v = np.round(SH - 1 - (ys - oy) / slam_reso).astype(int)
                inside = (u >= 0) & (u < SW) & (v >= 0) & (v < SH) & (~invalid)

                for uu, vv, col in zip(u[inside], v[inside], colors[inside]):
                    cv2.circle(ov, (uu, vv), POINT_RADIUS,
                            color=tuple(int(c) for c in col[::-1]), thickness=-1,
                            lineType=cv2.LINE_AA)

                cx = int(round((rx - ox) / slam_reso))
                cy = int(round(SH - 1 - (ry - oy) / slam_reso))
                cv2.drawMarker(ov, (cx, cy), (0, 0, 0),
                            markerType=cv2.MARKER_CROSS, markerSize=7, thickness=1)

                cv2.imwrite(f"{tag}_slam_{t}.png", ov)

# ───────── main ─────────
def main(args=None):
    rclpy.init(args=args)
    node = StaticScanNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node(); rclpy.shutdown()
        plt.ioff(); plt.show()

if __name__ == "__main__":
    main()