#!/usr/bin/env python3
# save_amcl_csv.py
#
#  /amcl_pose  →  CSV  (t, x, y, z, qx, qy, qz, qw, yaw)
# -------------------------------------------------------------------

import os, csv, math, datetime, rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# ───── 사용자 설정 ─────────────────────────────────────────────
OUT_DIR = os.path.expanduser('/home/csilab/tb3/path/path/amcl_logs/spot4')          # 저장 폴더
FILE_FMT = 'amcl_pose_%Y%m%d_%H%M%S.csv'             # 파일 이름 포맷
# -------------------------------------------------------------------

def quat_to_yaw(q):
    """quaternion → yaw(rad)"""
    return math.atan2(2*(q.w*q.z + q.x*q.y),
                      1 - 2*(q.y*q.y + q.z*q.z))

class AmclCsvLogger(Node):

    def __init__(self):
        super().__init__('amcl_csv_logger')

        # ── CSV 준비 ──
        os.makedirs(OUT_DIR, exist_ok=True)
        fname = datetime.datetime.now().strftime(FILE_FMT)
        self._full_path = os.path.join(OUT_DIR, fname)
        self._fp = open(self._full_path, 'w', newline='')
        self._writer = csv.writer(self._fp)
        self._writer.writerow(
            ['t_sec', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'yaw_rad']
        )
        self.get_logger().info(f'Logging to {self._full_path}')

        # ── QoS: AMCL 은 RELIABLE / KEEP_LAST 10 ──
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST
        )

        self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.cb_amcl,
            qos
        )

    def cb_amcl(self, msg: PoseWithCovarianceStamped):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = quat_to_yaw(q)
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self._writer.writerow(
            [f'{t:.6f}', p.x, p.y, p.z, q.x, q.y, q.z, q.w, yaw]
        )

    # Ctrl‑C 시 안전 종료
    def destroy_node(self):
        self._fp.close()
        self.get_logger().info(f'Saved file: {self._full_path}')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = AmclCsvLogger()
    node.get_logger().info('Subscribed to /amcl_pose  (Ctrl‑C to stop)')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
