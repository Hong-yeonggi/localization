#!/usr/bin/env python3
"""
오프라인: AMCL CSV ↔ plan.yaml 을 이용한 궤적 비교 플롯
"""
import yaml, csv, matplotlib.pyplot as plt, os

#plan_yaml = os.path.expanduser("~/tb3/path/plan.yaml")
amcl_csv   = os.path.expanduser("~/amcl_log.csv")

# plan_pts = []
# with open(plan_yaml) as f:
#     for d in yaml.safe_load_all(f):
#         if isinstance(d, dict) and isinstance(d.get("poses"), list):
#             for p in d["poses"]:
#                 pos = p["pose"]["position"]
#                 plan_pts.append((float(pos["x"]), float(pos["y"])))
#             break

# AMCL CSV 로드
amcl_x, amcl_y = [], []
with open(amcl_csv) as f:
    rdr = csv.DictReader(f)
    for row in rdr:
        amcl_x.append(float(row["x"]))
        amcl_y.append(float(row["y"]))

# 플롯
plt.figure()
if plan_pts:
    px, py = zip(*plan_pts)
    plt.plot(px, py, "k-", label="Plan Path")
plt.plot(amcl_x, amcl_y, "r--", linewidth=0.8, label="AMCL Pose")
plt.axis("equal")
plt.xlabel("X [m]");  plt.ylabel("Y [m]")
plt.title("Plan vs AMCL (offline CSV)")
plt.grid(True);  plt.legend()
plt.show()
