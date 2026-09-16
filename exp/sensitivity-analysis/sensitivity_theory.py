"""DCPAM 灵敏度分析——理论推导与数值验证。

坐标系约定：
  x : 激光轴方向（前后镜沿 x 排布，间距 80mm）
  y : 垂直方向（光路在 xz 平面内，y 不敏感）
  z : 反射镜法向在 xz 平面内的分量方向

关键公式：
  H = |L × w| / |L|        点到直线距离
  L = P2 - P1              激光轴向量
  w = T - P1               靶点到 P1 的向量

镜面反射：
  V = P - 2(n·P + d)·n     n=[√2/2, 0, √2/2], d=-n·M
  镜面平移 δM → 虚像点偏移 ΔV = 2(n·δM)·n

45° 镜面特性：
  x 方向平移 δx → ΔV = [δx, 0, δx]  (x 和 z 等量偏移)
  z 方向平移 δz → ΔV = [δz, 0, δz]
  → x/z 平移等价，且都改变 L 的方向 → 激光轴偏转
"""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path
import csv

# ---- 基线几何 ----
MIRROR_N = np.array([np.sqrt(2) / 2, 0, np.sqrt(2) / 2])
P1_M = np.array([0.0, 0.0, 23.0])
P2_M = np.array([80.0, 0.0, 23.0])
T_A = np.array([41.0, 0.0, -718.051])
T_B = np.array([341.0, 0.0, -718.051])  # 间距 300mm


def load_real_points():
    csv_path = Path(__file__).resolve().parent.parent.parent / "dataset" / "samples" / "spot-measurements.csv"
    with csv_path.open(newline="", encoding="utf-8") as f:
        r = next(csv.DictReader(f))
    return np.asarray(json.loads(r["front_real_point_device_mm"])), \
           np.asarray(json.loads(r["rear_real_point_device_mm"]))


def mirror(p, mp, n):
    n = n / np.linalg.norm(n)
    d = -float(n @ mp)
    return p - 2.0 * float(n @ p + d) * n


def rot(axis, deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    if axis == "x": return np.array([[1,0,0],[0,c,-s],[0,s,c]])
    if axis == "y": return np.array([[c,0,s],[0,1,0],[-s,0,c]])
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])


def H_of(T, fr, rr, f_pt, f_n, r_pt, r_n):
    P1 = mirror(fr, f_pt, f_n)
    P2 = mirror(rr, r_pt, r_n)
    L = P2 - P1
    w = T - P1
    return float(np.linalg.norm(np.cross(L, w)) / np.linalg.norm(L))


def perturb(name, delta, sign, T):
    s = sign * delta
    f_pt, f_n = P1_M.copy(), MIRROR_N.copy()
    r_pt, r_n = P2_M.copy(), MIRROR_N.copy()
    T_off = np.zeros(3)
    if name == "P1_dx": f_pt = P1_M + [s,0,0]
    elif name == "P1_dz": f_pt = P1_M + [0,0,s]
    elif name == "P1_rx": f_n = rot("x", s) @ MIRROR_N
    elif name == "P1_ry": f_n = rot("y", s) @ MIRROR_N
    elif name == "P1_rz": f_n = rot("z", s) @ MIRROR_N
    elif name == "P2_dx": r_pt = P2_M + [s,0,0]
    elif name == "P2_dz": r_pt = P2_M + [0,0,s]
    elif name == "P2_rx": r_n = rot("x", s) @ MIRROR_N
    elif name == "P2_ry": r_n = rot("y", s) @ MIRROR_N
    elif name == "P2_rz": r_n = rot("z", s) @ MIRROR_N
    elif name == "T_dx": T_off = [s,0,0]
    elif name == "T_dy": T_off = [0,s,0]
    elif name == "T_dz": T_off = [0,0,s]
    elif name == "rod_len": T_off = [0,0,-s]
    else: raise ValueError(name)
    return f_pt, f_n, r_pt, r_n, T + T_off


def compute_all_sensitivities(delta=1.0):
    fr, rr = load_real_points()
    H_A0 = H_of(T_A, fr, rr, P1_M, MIRROR_N, P2_M, MIRROR_N)
    H_B0 = H_of(T_B, fr, rr, P1_M, MIRROR_N, P2_M, MIRROR_N)
    dH0 = H_A0 - H_B0

    params = [
        ("P1_dx","mm"),("P1_dz","mm"),("P1_rx","°"),("P1_ry","°"),("P1_rz","°"),
        ("P2_dx","mm"),("P2_dz","mm"),("P2_rx","°"),("P2_ry","°"),("P2_rz","°"),
        ("T_dx","mm"),("T_dy","mm"),("T_dz","mm"),
    ]

    results = []
    for name, unit in params:
        f_pt, f_n, r_pt, r_n, TA_p = perturb(name, delta, +1, T_A)
        TB_p = perturb(name, delta, +1, T_B)[4]
        HA_p = H_of(TA_p, fr, rr, f_pt, f_n, r_pt, r_n)
        HB_p = H_of(TB_p, fr, rr, f_pt, f_n, r_pt, r_n)

        f_pt, f_n, r_pt, r_n, TA_m = perturb(name, delta, -1, T_A)
        TB_m = perturb(name, delta, -1, T_B)[4]
        HA_m = H_of(TA_m, fr, rr, f_pt, f_n, r_pt, r_n)
        HB_m = H_of(TB_m, fr, rr, f_pt, f_n, r_pt, r_n)

        eAp = (HA_p - H_A0) * 1000
        eAm = (HA_m - H_A0) * 1000
        eBp = (HB_p - H_B0) * 1000
        eBm = (HB_m - H_B0) * 1000
        edHp = ((HA_p - HB_p) - dH0) * 1000
        edHm = ((HA_m - HB_m) - dH0) * 1000

        a_A = (HA_p - HA_m) / (2 * delta) * 1000
        b_A = (HA_p + HA_m - 2 * H_A0) / (delta**2) * 1000
        asym = abs(b_A * delta / (a_A + 1e-12))
        a_dH = (edHp - edHm) / 2

        results.append({
            "param": name, "unit": unit,
            "eAp": float(eAp), "eAm": float(eAm),
            "eBp": float(eBp), "eBm": float(eBm),
            "edHp": float(edHp), "edHm": float(edHm),
            "unified": float(abs(a_A)),
            "dH_first": float(abs(a_dH)),
            "asym": float(asym),
        })
    return results, H_A0, H_B0, dH0


if __name__ == "__main__":
    results, HA, HB, dH = compute_all_sensitivities()
    print(f"H_A={HA:.4f}  H_B={HB:.4f}  ΔH={dH:.4f}")
    for r in sorted(results, key=lambda x: -x["unified"]):
        print(f"  {r['param']:>8} {r['unit']:>3}  单次={r['unified']:>8.1f}  差值={r['dH_first']:>8.1f}  非对称={r['asym']:.1%}")
