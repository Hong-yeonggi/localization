#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
import math
import time

class InitialPosePublisher(Node):
    def __init__(self, x, y, yaw_rad):
        super().__init__('initial_pose_publisher')
        # latching behavior 처럼 한 번만 publish: 큐 깊이 1
        self.pub = self.create_publisher(
            PoseWithCovarianceStamped,
            '/initialpose',
            1
        )
        # 잠시 기다렸다가 publish
        self.timer = self.create_timer(0.5, self.publish_initial_pose)
        self.x = x
        self.y = y
        self.yaw = yaw_rad
        self.published = False

    def publish_initial_pose(self):
        if self.published:
            return

        msg = PoseWithCovarianceStamped()
        # header
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'

        # position
        msg.pose.pose.position.x = self.x
        msg.pose.pose.position.y = self.y
        msg.pose.pose.position.z = 0.0

        # orientation (yaw only)
        msg.pose.pose.orientation.z = math.sin(self.yaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(self.yaw / 2.0)

        # covariance 36개 (row-major)
        msg.pose.covariance = [0.0] * 36

        self.pub.publish(msg)
        self.get_logger().info(
            f'Published initialpose → x: {self.x:.2f}, y: {self.y:.2f}, yaw: {self.yaw:.2f} rad'
        )
        self.published = True
        # publish 끝나면 노드 종료
        # 혹은 spin 후 자동 종료를 원하면 아래 호출
        rclpy.shutdown()

def main():
    rclpy.init()
    x_init = 0.245
    y_init = -0.993
    yaw_init = 1.553  

    node = InitialPosePublisher(x_init, y_init, yaw_init)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
