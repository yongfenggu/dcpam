"""正负不对称分析：对每个参数分别施加 +δ 和 −δ，分解一阶（对称）与二阶（非对称）分量。

回答三个问题：
1. 为什么镜面坐标偏移对差值影响大并放大？
2. 为什么前镜和后镜不对称？为什么 +δ 和 −δ 不对称？
3. 非对称情况下怎么给统一的灵敏度值？

数学基础：
  H(δ) = H(0) + a·δ + b·δ² + ...
  正向误差 = H(+δ) − H(0) = a·δ + b·δ²
  负向误差 = H(−δ) − H(0) = −a·δ + b·δ²

  一阶导数 a = [H(+δ) − H(−δ)] / (2δ)      ← 对称部分（线性灵敏度）
  二阶系数 b = [H(+δ) + H(−δ) − 2H(0)] / δ²  ← 非对称部分

  统一灵敏度 = |a|（一阶导数，正负通用）
  非对称度 = b·δ² / (|a|·δ) = |b·δ / a|（二阶 vs 一阶之比，越大越不对称）
"""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path

MIRROR_N = np.array([np.sqrt(2) / 2, 0, np.sqrt(2) / 2])
P1_MIRROR = np.array([0.0, 0.0, 23.0])
P2_MIRROR = np.array([80.0, 0.0, 23.0])

T_A = np.array([41.0, 0.0, -718.051])
T_B = np.array([341.0, 0.0, -718.051])

import csv


def _load_real_points(csv_path: Path):
    with csv_path.open(newline="", encoding="utf-8") as f:
        r = next(csv.DictReader(f))
    fr = np.asarray(json.loads(r["front_real_point_device_mm"]))
    rr = np.asarray(json.loads(r["rear_real_point_device_mm"]))
    return fr, rr


def mirror(point, mirror_pt, normal):
    n = normal / np.linalg.norm(normal)
    d = -float(n @ mirror_pt)
    return point - 2.0 * float(n @ point + d) * n


def rot_y(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def H_of(T, front_real, rear_real, f_pt, f_n, r_pt, r_n):
    P1 = mirror(front_real, f_pt, f_n)
    P2 = mirror(rear_real, r_pt, r_n)
    L = P2 - P1
    w = T - P1
    return float(np.linalg.norm(np.cross(L, w)) / np.linalg.norm(L))


def perturb(param_name, delta, sign):
    """返回 (f_pt, f_n, r_pt, r_n, T_override) 在指定参数、指定方向(+/-)的扰动。"""
    s = sign * delta
    f_pt, f_n = P1_MIRROR.copy(), MIRROR_N.copy()
    r_pt, r_n = P2_MIRROR.copy(), MIRROR_N.copy()
    T_off = np.zeros(3)

    if param_name == "P1_dx":
        f_pt = P1_MIRROR + np.array([s, 0, 0])
    elif param_name == "P1_dz":
        f_pt = P1_MIRROR + np.array([0, 0, s])
    elif param_name == "P2_dx":
        r_pt = P2_MIRROR + np.array([s, 0, 0])
    elif param_name == "P2_dz":
        r_pt = P2_MIRROR + np.array([0, 0, s])
    elif param_name == "P1_ry":
        f_n = rot_y(s) @ MIRROR_N
    elif param_name == "P2_ry":
        r_n = rot_y(s) @ MIRROR_N
    elif param_name == "T_dx":
        T_off = np.array([s, 0, 0])
    elif param_name == "T_dz":
        T_off = np.array([0, 0, s])
    elif param_name == "rod_len":
        T_off = np.array([0, 0, -s])
    else:
        raise ValueError(param_name)
    return f_pt, f_n, r_pt, r_n, T_off


def main():
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir.parent.parent / "dataset" / "samples" / "spot-measurements.csv"
    front_real, rear_real = _load_real_points(csv_path)

    H_A0 = H_of(T_A, front_real, rear_real, P1_MIRROR, MIRROR_N, P2_MIRROR, MIRROR_N)
    H_B0 = H_of(T_B, front_real, rear_real, P1_MIRROR, MIRROR_N, P2_MIRROR, MIRROR_N)
    dH0 = H_A0 - H_B0
    print(f"真值: H_A={H_A0:.4f}  H_B={H_B0:.4f}  ΔH={dH0:.4f} mm\n")

    params = ["P1_dx", "P1_dz", "P1_ry", "P2_dx", "P2_dz", "P2_ry",
              "T_dx", "T_dz", "rod_len"]
    units = {**{p: "mm" for p in params if "ry" not in p},
             **{p: "°" for p in params if "ry" in p}}

    # ---- 第一部分：固定 δ=1，正负方向分别多少 ----
    delta = 1.0
    print("=" * 90)
    print(f"单参数扰动 δ={delta} {units[params[0]]}：正向(+δ) vs 负向(−δ)  [单位: μm]")
    print("=" * 90)
    print(f"{'参数':>8} {'单位':>3} | {'H_A+':>8} {'H_A−':>8} | {'H_B+':>8} {'H_B−':>8} | "
          f"{'ΔH+':>8} {'ΔH−':>8} | {'一阶A':>8} {'非对称A':>9}")
    print("-" * 90)

    asym_results = []
    for p in params:
        u = units[p]
        # +delta
        f_pt, f_n, r_pt, r_n, T_off = perturb(p, delta, +1)
        HA_p = H_of(T_A + T_off, front_real, rear_real, f_pt, f_n, r_pt, r_n)
        HB_p = H_of(T_B + T_off, front_real, rear_real, f_pt, f_n, r_pt, r_n)
        dH_p = HA_p - HB_p
        # -delta
        f_pt, f_n, r_pt, r_n, T_off = perturb(p, delta, -1)
        HA_m = H_of(T_A + T_off, front_real, rear_real, f_pt, f_n, r_pt, r_n)
        HB_m = H_of(T_B + T_off, front_real, rear_real, f_pt, f_n, r_pt, r_n)
        dH_m = HA_m - HB_m

        err_Ap = (HA_p - H_A0) * 1000
        err_Am = (HA_m - H_A0) * 1000
        err_Bp = (HB_p - H_B0) * 1000
        err_Bm = (HB_m - H_B0) * 1000
        err_dHp = (dH_p - dH0) * 1000
        err_dHm = (dH_m - dH0) * 1000

        # 一阶导数（中心差分）
        a_A = (HA_p - HA_m) / (2 * delta) * 1000  # μm per unit
        # 二阶非对称
        b_A = (HA_p + HA_m - 2 * H_A0) / (delta ** 2) * 1000  # μm per unit²
        asym_A = b_A * delta ** 2 / (abs(a_A * delta) + 1e-12)  # 比值

        asym_results.append({
            "param": p, "unit": u,
            "err_A_plus_um": err_Ap, "err_A_minus_um": err_Am,
            "err_B_plus_um": err_Bp, "err_B_minus_um": err_Bm,
            "err_dH_plus_um": err_dHp, "err_dH_minus_um": err_dHm,
            "first_order_A_um_per_unit": a_A,
            "second_order_A_um_per_unit2": b_A,
            "asym_ratio_A": asym_A,
        })
        print(f"{p:>8} {u:>3} | {err_Ap:>8.1f} {err_Am:>8.1f} | "
              f"{err_Bp:>8.1f} {err_Bm:>8.1f} | {err_dHp:>8.1f} {err_dHm:>8.1f} | "
              f"{a_A:>8.1f} {asym_A:>9.1%}")

    # ---- 第二部分：非对称随 δ 大小变化 ----
    print("\n" + "=" * 90)
    print("非对称度随误差幅度变化（以前镜 x 平移为例）")
    print("=" * 90)
    print(f"{'δ (mm)':>8} | {'H_A正向(μm)':>12} {'H_A负向(μm)':>12} {'|正向|/|负向|':>14} "
          f"| {'一阶导数':>10} {'二阶系数':>10}")
    print("-" * 90)

    p = "P1_dx"
    for delta_i in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        f_pt, f_n, r_pt, r_n, T_off = perturb(p, delta_i, +1)
        HA_p = H_of(T_A + T_off, front_real, rear_real, f_pt, f_n, r_pt, r_n)
        f_pt, f_n, r_pt, r_n, T_off = perturb(p, delta_i, -1)
        HA_m = H_of(T_A + T_off, front_real, rear_real, f_pt, f_n, r_pt, r_n)

        err_p = (HA_p - H_A0) * 1000
        err_m = (HA_m - H_A0) * 1000
        a = (HA_p - HA_m) / (2 * delta_i) * 1000
        b = (HA_p + HA_m - 2 * H_A0) / (delta_i ** 2) * 1000
        ratio = abs(err_p) / (abs(err_m) + 1e-12)
        print(f"{delta_i:>8.2f} | {err_p:>12.2f} {err_m:>12.2f} {ratio:>14.3f} | "
              f"{a:>10.2f} {b:>10.2f}")

    # ---- 第三部分：为什么差值会放大——几何图解 ----
    print("\n" + "=" * 90)
    print("为什么差值会放大：镜面偏移 → 激光轴偏转 → 远处靶点 H 变化更大")
    print("=" * 90)

    # 镜面 x 偏移 δ → 虚像点偏移 ΔV = 2(n·δ)·n = [δ, 0, δ]
    # 这改变 L = P2-P1 的方向 → 激光轴偏转
    # 偏转角 θ ≈ ΔV_perp / |L|，ΔV_perp 是垂直于 L 的分量
    for delta_i in [1.0]:
        f_pt = P1_MIRROR + np.array([delta_i, 0, 0])
        P1_new = mirror(front_real, f_pt, MIRROR_N)
        P2_new = mirror(rear_real, P2_MIRROR, MIRROR_N)
        L_new = P2_new - P1_new
        L_old = mirror(rear_real, P2_MIRROR, MIRROR_N) - mirror(front_real, P1_MIRROR, MIRROR_N)

        # 偏转角
        angle = np.arccos(np.clip(L_new @ L_old / (np.linalg.norm(L_new) * np.linalg.norm(L_old)), -1, 1))
        angle_deg = np.rad2deg(angle)

        # 靶点到轴线的垂直距离变化
        # A 在近处(x≈41)，B 在远处(x≈341)，间距 300mm
        print(f"  前镜 x 偏移 {delta_i}mm:")
        print(f"    虚像点 P1 偏移: {mirror(front_real, f_pt, MIRROR_N) - mirror(front_real, P1_MIRROR, MIRROR_N)}")
        print(f"    激光轴 L: {L_old} → {L_new}")
        print(f"    激光轴偏转角: {angle_deg:.4f}° = {angle * 1e6:.1f} μrad")
        print(f"    |L|: {np.linalg.norm(L_old):.3f} → {np.linalg.norm(L_new):.3f} mm")
        print(f"    靶点 A (x≈{T_A[0]}): H {H_A0:.4f} → {H_of(T_A, front_real, rear_real, f_pt, MIRROR_N, P2_MIRROR, MIRROR_N):.4f} mm")
        print(f"    靶点 B (x≈{T_B[0]}): H {H_B0:.4f} → {H_of(T_B, front_real, rear_real, f_pt, MIRROR_N, P2_MIRROR, MIRROR_N):.4f} mm")
        print(f"    → A 变化 {(H_of(T_A, front_real, rear_real, f_pt, MIRROR_N, P2_MIRROR, MIRROR_N)-H_A0)*1000:+.1f}μm, "
              f"B 变化 {(H_of(T_B, front_real, rear_real, f_pt, MIRROR_N, P2_MIRROR, MIRROR_N)-H_B0)*1000:+.1f}μm")
        print(f"    → 差值变化 {((H_of(T_A, front_real, rear_real, f_pt, MIRROR_N, P2_MIRROR, MIRROR_N) - H_of(T_B, front_real, rear_real, f_pt, MIRROR_N, P2_MIRROR, MIRROR_N)) - dH0)*1000:+.1f}μm (放大!)")

    # ---- 第四部分：统一灵敏度值 ----
    print("\n" + "=" * 90)
    print("统一灵敏度值建议（一阶导数 |∂H/∂param|，μ m/单位）")
    print("=" * 90)
    print(f"{'参数':>8} {'单位':>3} | {'单次A一阶':>10} {'单次B一阶':>10} {'差值一阶':>10} | "
          "{'非对称度@1单位':>14} {'建议统一值':>12}")
    print("-" * 90)

    for r in asym_results:
        p = r["param"]
        u = r["unit"]
        # 差值一阶
        dH_first = (r["err_dH_plus_um"] - r["err_dH_minus_um"]) / 2
        dH_asym = abs(r["err_dH_plus_um"] + r["err_dH_minus_um"]) / 2
        # 统一值 = 一阶导数绝对值
        unified = abs(r["first_order_A_um_per_unit"])
        print(f"{p:>8} {u:>3} | {r['first_order_A_um_per_unit']:>10.1f} "
              f"{(r['err_B_plus_um']-r['err_B_minus_um'])/2:>10.1f} {dH_first:>10.1f} | "
              f"{r['asym_ratio_A']:>14.1%} {unified:>12.1f}")

    # ---- 输出 JSON ----
    out = {"true": {"H_A": H_A0, "H_B": H_B0, "dH": dH0},
           "asymmetry": asym_results}
    (Path(__file__).resolve().parent / "mc_asymmetry.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
