#!/usr/bin/env python3
import os
import math
import cv2
import numpy as np
import pandas as pd

def main():
    pred_csv = "/home/yeonggi/tb3/model/cnn/unet/test_pred_unet_noglass_norm.csv"
    xy_csv   = "/home/yeonggi/tb3/model/cnn/input/point_label_no_glass3.csv"

    if not os.path.isfile(pred_csv):
        print(f"Error: {pred_csv} not found.")
        return
    if not os.path.isfile(xy_csv):
        print(f"Error: {xy_csv} not found.")
        return

    # 1) DataFrame 로드
    df_pred = pd.read_csv(pred_csv)
    df_xy   = pd.read_csv(xy_csv)
    
    # 2) beam_rank 생성하여 병합
    df_xy["beam_rank"]   = df_xy.groupby("scan_time").cumcount()
    df_pred["beam_rank"] = df_pred.groupby("scan_time").cumcount()

    df_merged = pd.merge(
        df_xy,
        df_pred[["scan_time","beam_rank","score_true","score_pred"]],
        on=["scan_time","beam_rank"],
        how="inner"
    )
    print("Merged shape =", df_merged.shape)
    print(df_merged.head(5))

    # 3) 도면 이미지 로드
    map_file = "map_mod2.png"
    if not os.path.isfile(map_file):
        print(f"Error: {map_file} not found.")
        return

    img = cv2.imread(map_file, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Cannot load {map_file} as color image.")
        return

    h, w = img.shape[:2]
    print(f"Image shape=({h},{w})")

    # 4) 맵 원점(픽셀 좌표), 스케일 설정
    ox, oy = 227, 682      # 예시
    pixel_scale = 0.05     # 1픽셀당 0.05m

    # 5) 점 색상 함수: score_pred(0~1)를 R(빨강)와 B(파랑) 사이로 표현
    def color_from_score(score):
        # 안전하게 0~1 범위로 클램핑
        score = max(0.0, min(1.0, score))
        B = int(255 * (1.0 - score))  # score=0 -> 파랑
        G = 0
        R = int(255 * score)         # score=1 -> 빨강
        return (B, G, R)

    count_drawn = 0

    # 6) df_merged 루프
    for idx, row in df_merged.iterrows():
        mx = row['map_x']
        my = row['map_y']
        sc = row['score_true']       # (원본 점수, 필요 없으면 사용 안 해도 됨)
        sp = row['score_pred']  # 예측 점수
        #if sp > 0.90:
            
        # (1) 예측 점수가 inf or NaN이면 제외
        if math.isinf(sp) or math.isnan(sp):
            continue

        # (2) px_float, py_float 구하기
        px_float = ox + mx / pixel_scale
        py_float = oy - my / pixel_scale

        # (3) 좌표가 inf/NaN이면 제외
        if (math.isnan(px_float) or math.isnan(py_float) or
            math.isinf(px_float) or math.isinf(py_float)):
            continue

        px = int(px_float)
        py = int(py_float)

        # (4) 이미지 범위 체크
        if 0 <= px < w and 0 <= py < h:
            color = color_from_score(sp)
            cv2.circle(img, (px, py), 1, color, -1)
            count_drawn += 1

    print(f"Drew {count_drawn} points.")

    out_file = "/home/yeonggi/tb3/model/cnn/plot/score/test_pred_unet_noglass_norm.png"
    cv2.imwrite(out_file, img)
    print(f"Saved => {out_file}")

if __name__=="__main__":
    main()
