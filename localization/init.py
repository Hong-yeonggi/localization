#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped

def main():
    rclpy.init()
    node = Node('initial_pose_publisher')
    pub = node.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)

    # 메시지 생성
    msg = PoseWithCovarianceStamped()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.header.frame_id = 'map'

    # 위치 설정
    msg.pose.pose.position.x = 12.7713
    msg.pose.pose.position.y = 0.00492525
    msg.pose.pose.position.z = 0.0

    # 방향 설정 (quaternion)
    msg.pose.pose.orientation.x = 0.0
    msg.pose.pose.orientation.y = 0.0
    msg.pose.pose.orientation.z = -0.701604
    msg.pose.pose.orientation.w = 0.712567

    # 공분산: 6×6 대각 성분만 0.25, 나머지 0.0
    msg.pose.covariance = [0.25 if i % 7 == 0 else 0.0 for i in range(36)]

    node.get_logger().info('Publishing initial pose to /initialpose')
    pub.publish(msg)

    # 퍼블리시 보장 위해 잠시 spin
    rclpy.spin_once(node, timeout_sec=0.5)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
