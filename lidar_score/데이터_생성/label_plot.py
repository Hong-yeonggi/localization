#!/usr/bin/env python3
import os
import csv
import math
import cv2
import numpy as np

def main():
    # 1) CSV 파일: e.g. "scan_label.csv"
    #    열: [scan_time, beam_idx, map_x, map_y, range, dist_m, score]
    csv_file = "/home/yeonggi/tb3/bag_slam/bag_floor/point/input_final_병합.csv"
    if not os.path.isfile(csv_file):
        print(f"Error: {csv_file} not found.")
        return

    data = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            scan_t = float(row['scan_time'])
            beam_idx = int(row['beam_idx'])
            mx = float(row['map_x'])
            my = float(row['map_y'])
            rng = float(row['range'])
            intensity = float(row['intensity'])
            dist_m = float(row['dist_m'])
            score = float(row['score'])  # 0..1
            data.append((scan_t, beam_idx, mx, my, rng, intensity, dist_m, score))

    print(f"Loaded {len(data)} points from {csv_file}")

    # 2) 도면 이미지 로드
    map_file = "map_mod_noglass.png"
    if not os.path.isfile(map_file):
        print(f"Error: {map_file} not found.")
        return

    img = cv2.imread(map_file, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Cannot load {map_file} as color.")
        return

    h, w = img.shape[:2]
    print(f"Image shape=({h},{w})")

    # 3) 원점 픽셀, 스케일
    #    (map_x, map_y) -> 픽셀(px, py)
    (ox, oy) = (227, 683)
    pixel_scale = 0.05

    # 4) score -> color
    def color_from_score(sc):
        sc = max(0.0, min(1.0, sc))
        B = int(255*(1.0 - sc))  # score=0 => B=255
        G = 0
        R = int(255*sc)         # score=1 => R=255
        return (B, G, R)

    count_drawn = 0

    TS = 1743584532.47754

    for (scan_t, beam_idx, mx, my, rng, intensity, dist_m, sc) in data:
        #if scan_t == 1743584532.47754:
        #if sc == 1:
            
        px_float = (ox + mx/pixel_scale)
        py_float = (oy - my/pixel_scale)

        # NaN/inf 체크
        if (math.isnan(px_float) or math.isnan(py_float) or
            math.isinf(px_float) or math.isinf(py_float)):
            continue

        px = int(px_float)
        py = int(py_float)

        # 범위체크
        if 0 <= px < w and 0 <= py < h:
            col = color_from_score(sc)  # score-based color
            cv2.circle(img, (px, py), 1, col, -1)
            count_drawn += 1

    print(f"Drew {count_drawn} points")

    out_file = "label_final_plot_병합.png"
    cv2.imwrite(out_file, img)
    print(f"Saved overlay => {out_file}")

if __name__=="__main__":
    main()                                                  
