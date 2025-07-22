#!/usr/bin/env python3
import os
import math
import bisect
import csv

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped

import matplotlib.pyplot as plt

##############################################################################
# 보조 함수
##############################################################################
def laser_beam_xy(range_val, angle):
    x = range_val*math.cos(angle)
    y = range_val*math.sin(angle)
    return x, y

def transform_xy(x_local, y_local, robot_x, robot_y, robot_yaw):
    X = x_local*math.cos(robot_yaw) - y_local*math.sin(robot_yaw)
    Y = x_local*math.sin(robot_yaw) + y_local*math.cos(robot_yaw)
    X += robot_x
    Y += robot_y
    return X, Y

##############################################################################
# Bag -> CSV (append) 함수
##############################################################################
def bag_to_csv_with_pose_append(
    bag_path, scan_topic, pose_topic, output_csv
):
    """
    - Bag에서 /tracked_pose, /scan 읽기
    - 각 scan 시점 근접 pose -> map coords
    - CSV 열: [scan_time, beam_idx, map_x, map_y, range]
    - 첫 호출 시, CSV가 없으면 헤더 + 데이터
      이후 호출 시, CSV가 이미 있으면 헤더 없이 이어쓰기
    """
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}

    pose_data = []
    scan_data = []

    # (1) 메시지 읽기
    while reader.has_next():
        topic_name, raw_data, t_stamp = reader.read_next()
        stamp_sec = t_stamp // 10**9
        stamp_nsec = t_stamp % 10**9
        time_float = stamp_sec + stamp_nsec*1e-9

        if topic_name == pose_topic:
            PoseMsgType = get_message(type_map[topic_name])
            pose_msg = deserialize_message(raw_data, PoseMsgType)

            px = pose_msg.pose.position.x
            py = pose_msg.pose.position.y
            # 쿼터니언->yaw
            ox, oy, oz, ow = pose_msg.pose.orientation.x, pose_msg.pose.orientation.y,\
                             pose_msg.pose.orientation.z, pose_msg.pose.orientation.w
            siny_cosp = 2.0*(ow*oz + ox*oy)
            cosy_cosp = 1.0-2.0*(oy*oy + oz*oz)
            yaw = math.atan2(siny_cosp, cosy_cosp)

            pose_data.append((time_float, px, py, yaw))

        elif topic_name == scan_topic:
            scan_data.append((time_float, raw_data))

    # (2) pose time 정렬
    pose_data.sort(key=lambda x: x[0])
    pose_times = [p[0] for p in pose_data]

    # (3) Append 모드
    file_exists = os.path.isfile(output_csv)
    with open(output_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        # 헤더는 없으면 쓴다
        if not file_exists:
            writer.writerow(["scan_time","beam_idx","map_x","map_y","range",'intensity'])

        for (scan_t, raw_data) in scan_data:
            # pose 탐색
            idx = bisect.bisect_left(pose_times, scan_t)
            if idx==0:
                matched_pose = pose_data[0]
            elif idx>=len(pose_data):
                matched_pose = pose_data[-1]
            else:
                prev_t = pose_times[idx-1]
                next_t = pose_times[idx]
                if abs(prev_t - scan_t)<abs(next_t - scan_t):
                    matched_pose=pose_data[idx-1]
                else:
                    matched_pose=pose_data[idx]
            _, px, py, yaw = matched_pose

            # Scan 역직렬화
            ScanMsgType = get_message(type_map[scan_topic])
            scan_msg = deserialize_message(raw_data, ScanMsgType)

            angle_min = scan_msg.angle_min
            angle_inc = scan_msg.angle_increment
            ranges = scan_msg.ranges
            intensity = scan_msg.intensities
            # print(f"DEBUG: scan_t={scan_t:.3f}, len(ranges)={len(ranges)}, len(intensities)={len(intensity)}")
            # if len(ranges) == len(intensity) and len(ranges)>0:
            #     print("  e.g. ranges[0] =", ranges[0], ", intensities[0] =", intensity[0])
            #     print("  e.g. ranges[-1] =", ranges[-1], ", intensities[-1] =", intensity[-1])
                
            for i, r_val in enumerate(ranges):
                angle_i = angle_min + i*angle_inc
                x_local, y_local = laser_beam_xy(r_val, angle_i)
                x_map, y_map = transform_xy(x_local, y_local, px, py, yaw)
                intensity_val = intensity[i]
                
                writer.writerow([
                    f"{scan_t:.6f}",
                    i,
                    f"{x_map:.6f}",
                    f"{y_map:.6f}",
                    f"{r_val:.6f}",
                    f"{intensity_val:.4f}"
                ])

    print(f"Appended {bag_path} => {output_csv}")

##############################################################################
# CSV -> Plot
##############################################################################
# def plot_scan_csv(csv_file):
#     import matplotlib.pyplot as plt
#     xs, ys = [], []
#     with open(csv_file, 'r') as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             # [scan_time, beam_idx, map_x, map_y, range]
#             x_map = float(row["map_x"])
#             y_map = float(row["map_y"])
#             xs.append(x_map)
#             ys.append(y_map)
#     plt.figure()
#     plt.scatter(xs, ys, s=2, c='blue', alpha=0.5)
#     plt.title("Merged CSV Plot (map coords)")
#     plt.axis('equal')
#     plt.xlabel("X")
#     plt.ylabel("Y")
#     plt.show()

##############################################################################
# main
##############################################################################
if __name__=="__main__":
    # 1) 첫 번째 Bag -> CSV (pose+scan)
    BAG1 = "/home/yeonggi/tb3/bag_slam/bag_floor/5층_1,2/rosbag2_2025_06_29-17_22_03_쓸만"
    BAG2 = "/home/yeonggi/tb3/bag_slam/bag_floor/5층_1,2/rosbag2_2025_06_29-19_03_37_쓸만"

    SCAN_TOPIC="/scan"
    POSE_TOPIC="/tracked_pose"
    OUTPUT_CSV="points_final_병합.csv"

    # 첫 번째 Bag
    bag_to_csv_with_pose_append(BAG1, SCAN_TOPIC, POSE_TOPIC, OUTPUT_CSV)
    # 두 번째 Bag -> 이어쓰기
    bag_to_csv_with_pose_append(BAG2, SCAN_TOPIC, POSE_TOPIC, OUTPUT_CSV)

    # 최종 merged CSV -> plot
    #plot_scan_csv(OUTPUT_CSV)
