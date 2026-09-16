"""完整 13 自由度灵敏度分析：每镜 5 个有效自由度 + 靶点 3 个。

对每个自由度：
  - +δ 和 −δ 分别算 H_A, H_B, ΔH 的变化
  - 一阶导数 a = [H(+δ) − H(−δ)] / (2δ)  ← 统一灵敏度
  - 二阶系数 b = [H(+δ) + H(−δ) − 2H₀] / δ²  ← 非对称
  - 非对称度 = |b·δ / a|  (<100% 时线性近似可靠)
"""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path

MIRROR_N = np.array([np.sqrt(2) / 2, 0, np.sqrt(2) / 2])
P1_M = np.array([0.0, 0.0, 23.0])
P2_M = np.array([80.0, 0.0, 23.0])
T_A = np.array([41.0, 0.0, -718.051])
T_B = np.array([341.0, 0.0, -718.051])

import csv


def _load(csv_path):
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
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


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

    # 前镜平移
    if name == "P1_dx": f_pt = P1_M + [s, 0, 0]
    elif name == "P1_dz": f_pt = P1_M + [0, 0, s]
    # 前镜旋转（绕过镜面中心点 P1_M 的轴）
    elif name == "P1_rx": f_n = rot("x", s) @ MIRROR_N
    elif name == "P1_ry": f_n = rot("y", s) @ MIRROR_N
    elif name == "P1_rz": f_n = rot("z", s) @ MIRROR_N
    # 后镜平移
    elif name == "P2_dx": r_pt = P2_M + [s, 0, 0]
    elif name == "P2_dz": r_pt = P2_M + [0, 0, s]
    # 后镜旋转
    elif name == "P2_rx": r_n = rot("x", s) @ MIRROR_N
    elif name == "P2_ry": r_n = rot("y", s) @ MIRROR_N
    elif name == "P2_rz": r_n = rot("z", s) @ MIRROR_N
    # 靶点
    elif name == "T_dx": T_off = [s, 0, 0]
    elif name == "T_dy": T_off = [0, s, 0]
    elif name == "T_dz": T_off = [0, 0, s]
    elif name == "rod_len": T_off = [0, 0, -s]
    else: raise ValueError(name)
    return f_pt, f_n, r_pt, r_n, T + T_off


def main():
    base = Path(__file__).resolve().parent
    fr, rr = _load(base.parent.parent / "dataset" / "samples" / "spot-measurements.csv")

    H_A0 = H_of(T_A, fr, rr, P1_M, MIRROR_N, P2_M, MIRROR_N)
    H_B0 = H_of(T_B, fr, rr, P1_M, MIRROR_N, P2_M, MIRROR_N)
    dH0 = H_A0 - H_B0

    params = [
        # 前镜 5 个
        ("P1_dx", "mm", "前镜 x 平移"),
        ("P1_dz", "mm", "前镜 z 平移"),
        ("P1_rx", "°", "前镜绕 x 转"),
        ("P1_ry", "°", "前镜绕 y 转"),
        ("P1_rz", "°", "前镜绕 z 转"),
        # 后镜 5 个
        ("P2_dx", "mm", "后镜 x 平移"),
        ("P2_dz", "mm", "后镜 z 平移"),
        ("P2_rx", "°", "后镜绕 x 转"),
        ("P2_ry", "°", "后镜绕 y 转"),
        ("P2_rz", "°", "后镜绕 z 转"),
        # 靶点 3 个
        ("T_dx", "mm", "靶点 x"),
        ("T_dy", "mm", "靶点 y"),
        ("T_dz", "mm", "靶点 z / 杆长"),
    ]

    delta = 1.0
    print(f"真值: H_A={H_A0:.4f}  H_B={H_B0:.4f}  ΔH={dH0:.4f} mm\n")
    print("=" * 120)
    print(f"{'参数':>8} {'单位':>3} {'说明':<12} | {'单次A+':>8} {'单次A−':>8} | "
          f"{'单次B+':>8} {'单次B−':>8} | {'差值+':>8} {'差值−':>8} | "
          f"{'统一|a|':>8} {'非对称':>7}")
    print("-" * 120)

    results = []
    for name, unit, desc in params:
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
        b_A = (HA_p + HA_m - 2 * H_A0) / (delta ** 2) * 1000
        asym = abs(b_A * delta / (a_A + 1e-12))

        # 差值一阶
        a_dH = (edHp - edHm) / 2

        results.append({
            "param": name, "unit": unit, "desc": desc,
            "eAp": eAp, "eAm": eAm, "eBp": eBp, "eBm": eBm,
            "edHp": edHp, "edHm": edHm,
            "unified_um": abs(a_A),
            "dH_first_um": abs(a_dH),
            "asym": asym,
        })
        print(f"{name:>8} {unit:>3} {desc:<12} | {eAp:>8.1f} {eAm:>8.1f} | "
              f"{eBp:>8.1f} {eBm:>8.1f} | {edHp:>8.1f} {edHm:>8.1f} | "
              f"{abs(a_A):>8.1f} {asym:>6.1%}")

    # ---- 汇总排序 ----
    print("\n" + "=" * 80)
    print("按单次测量统一灵敏度 |a| 排序")
    print("=" * 80)
    print(f"{'参数':>8} {'单位':>3} {'说明':<12} | {'单次|a|':>10} {'差值|a|':>10} {'非对称':>8}")
    print("-" * 80)
    for r in sorted(results, key=lambda x: -x["unified_um"]):
        print(f"{r['param']:>8} {r['unit']:>3} {r['desc']:<12} | "
              f"{r['unified_um']:>10.1f} {r['dH_first_um']:>10.1f} {r['asym']:>8.1%}")

    # ---- y 平移验证（应该为 0）----
    print("\n" + "=" * 60)
    print("验证：y 平移（应该不敏感）")
    print("-" * 60)
    for name in ["P1_dy", "P2_dy"]:
        f_pt = (P1_M if "P1" in name else P2_M).copy()
        idx = 1
        f_pt[idx] += delta
        if "P1" in name:
            HA = H_of(T_A, fr, rr, f_pt, MIRROR_N, P2_M, MIRROR_N)
            HB = H_of(T_B, fr, rr, f_pt, MIRROR_N, P2_M, MIRROR_N)
        else:
            HA = H_of(T_A, fr, rr, P1_M, MIRROR_N, f_pt, MIRROR_N)
            HB = H_of(T_B, fr, rr, P1_M, MIRROR_N, f_pt, MIRROR_N)
        print(f"  {name} +1mm: ΔH_A={(HA-H_A0)*1000:.2f}μm  ΔH_B={(HB-H_B0)*1000:.2f}μm")

    (base / "mc_full_dofs.json").write_text(
        json.dumps({"true": {"H_A": H_A0, "H_B": H_B0, "dH": dH0},
                    "results": results}, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")
    print(f"\n输出: {base / 'mc_full_dofs.json'}")


if __name__ == "__main__":
    main()
