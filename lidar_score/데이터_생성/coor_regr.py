#!/usr/bin/env python3
#origin pixel = (234, 683)
#origin pixel = (226, 684)
#origin pixel = (227, 683), 최종

import os, math, csv
import cv2
import numpy as np

##############################################################################
# (1) CSV 로딩 함수
##############################################################################
def load_scan_map_csv(filename):
    data = []
    with open(filename, newline='') as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            data.append((
                float(r["scan_time"]),
                int(r["beam_idx"]),
                float(r["map_x"]),
                float(r["map_y"]),
                float(r["range"]),
                float(r["intensity"])
            ))
    return data                                                     

##############################################################################
# (2) 마우스 클릭 콜백
##############################################################################
_click_pt = None
def _on_mouse(event, x, y, flags, param):
    global _click_pt
    if event == cv2.EVENT_LBUTTONDOWN:
        _click_pt = (x, y)
        print(f"[INFO] Picked origin pixel = ({x}, {y})")           

##############################################################################
# (3) main
##############################################################################
def main():
    # A) Source CSV -----------------------------------------------------------------
    SRC_CSV = "/home/yeonggi/tb3/bag_slam/bag_floor/point/points_final_병합.csv"
    if not os.path.isfile(SRC_CSV):
        raise FileNotFoundError(SRC_CSV)
    src_data = load_scan_map_csv(SRC_CSV)
    N = len(src_data)
    print(f"[INFO] loaded {N} rows from {SRC_CSV}")

    # B) 맵 이미지 열기 & 클릭으로 원점 선택 ------------------------------------------
    MAP_IMG = "/home/yeonggi/tb3/bag_slam/bag_floor/point/map_mod_noglass.png"
    if not os.path.isfile(MAP_IMG):
        raise FileNotFoundError(MAP_IMG)
    img = cv2.imread(MAP_IMG, cv2.IMREAD_COLOR)                      
    if img is None:
        raise IOError(f"cannot load {MAP_IMG}")

    cv2.namedWindow("Pick Origin (click, then press space)")
    cv2.setMouseCallback("Pick Origin (click, then press space)", _on_mouse)
    while True:
        disp = img.copy()
        if _click_pt is not None:
            cv2.drawMarker(disp, _click_pt, (0, 0, 255),
                           markerType=cv2.MARKER_CROSS,
                           markerSize=12, thickness=2)
        cv2.imshow("Pick Origin (click, then press space)", disp)
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 32):                                            
            break
    cv2.destroyAllWindows()

    if _click_pt is None:
        print("[ERR] 원점을 클릭하지 않았습니다. 프로그램 종료.")
        return
    ox, oy = _click_pt                                               
    print(f"[INFO] Using origin pixel (ox, oy) = {ox, oy}")

    # C) distanceTransform  --------------------------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    line_mask = np.where(gray < 254, 0, 255).astype(np.uint8)
    dist_img = cv2.distanceTransform(line_mask, cv2.DIST_L2, 3)

    rows, cols = gray.shape
    pixel_scale = 0.05                                               # m / px

    # D) NaN/Inf 필터 --------------------------------------------------------------
    valid_mask = []
    for (_, _, mx, my, _, _) in src_data:
        valid_mask.append(not (math.isnan(mx) or math.isnan(my) or
                               math.isinf(mx) or math.isinf(my)))
    valid_mask = np.array(valid_mask, dtype=bool)

    # E) 결과 생성 ------------------------------------------------------------------
    results = []
    for i, (st, bi, mx, my, rng, intensity) in enumerate(src_data):
        if not valid_mask[i]:
            results.append((st, bi, mx, my, rng, intensity, float("inf"), 0.0))
            continue

        # ‘원점 맞추기’ : 회전없음, 스케일 1, 단순 평행이동
        px_float = ox +  mx / pixel_scale
        py_float = oy -  my / pixel_scale

        if (math.isnan(px_float) or math.isnan(py_float) or
            math.isinf(px_float) or math.isinf(py_float)):
            results.append((st, bi, mx, my, rng, intensity, float("inf"), 0.0))
            continue

        px, py = int(px_float), int(py_float)
        if 0 <= px < cols and 0 <= py < rows:
            dist_pix = dist_img[py, px]
            dist_m   = dist_pix * pixel_scale
            score    = math.exp(-dist_m)
        else:
            dist_m, score = float("inf"), 0.0

        results.append((st, bi, mx, my, rng, intensity, dist_m, score))

    # F) CSV 저장 -------------------------------------------------------------------
    OUT_CSV = "input_final_병합.csv"
    with open(OUT_CSV, "w", newline='') as f:
        w = csv.writer(f)
        w.writerow(["scan_time","beam_idx","map_x","map_y",
                    "range","intensity","dist_m","score"])
        for row in results:
            st, bi, mx, my, rng, intensity, dm, sc = row
            w.writerow([f"{st:.6f}", bi, f"{mx:.5f}", f"{my:.5f}",
                        f"{rng:.5f}", f"{intensity:.5f}",
                        f"{dm:.5f}", f"{sc:.5f}"])
    print(f"[INFO] saved  →  {OUT_CSV}\n[Done]")

##############################################################################
if __name__ == "__main__":
    main()
