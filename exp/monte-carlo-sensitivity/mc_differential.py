"""差值测量灵敏度分析：两次测量的差值 ΔH = H_A − H_B 对参数误差的响应。

核心区别：
  - 单次灵敏度：∂H/∂param（参数变 → H 变多少）
  - 差值灵敏度：∂(H_A − H_B)/∂param = ∂H_A/∂param − ∂H_B/∂param
  - 如果两次测量灵敏度接近，差值灵敏度 ≈ 0，误差被抵消

场景：同一束激光（P1/P2 共享），两个不同位置的靶点 T_A、T_B，
关心 ΔH = H_A − H_B 的误差。

定义（与用户约定一致）：
  T  : 靶点（三维坐标）
  P1 : 1号点（前反射镜虚像点）
  P2 : 2号点（后反射镜虚像点）
  L  = P2 − P1（激光轴方向向量）
  w  = T − P1（或 T − P2，等价）
  H  = |L × w| / |L|（点到直线距离）

参数误差类型：
  - 镜面平移：P1/P2 各方向偏移（共享 → 两次测量相同误差）
  - 镜面旋转：法向偏转
  - 靶点平移：root 各方向 + 杆长（共享 → 两次测量相同误差）
"""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path

# ---- 基线几何（真值，取 config.toml）----
MIRROR_N = np.array([np.sqrt(2)/2, 0, np.sqrt(2)/2])
P1_TRUE = np.array([0.0, 0.0, 23.0])   # 前镜点（真值）
P2_TRUE = np.array([80.0, 0.0, 23.0])  # 后镜点（真值）

# 两个靶点位置（模拟两次测量）
# 从 CSV: 基线靶点 [41, 0, -718.051]，激光轴沿 x 方向
# 位置A：当前靶点位置
# 位置B：沿 x 轴移动 300mm（模拟探杆移到另一个位置）
T_A = np.array([41.0, 0.0, -718.051])
T_B = np.array([341.0, 0.0, -718.051])  # 沿 x 偏移 300mm

# 用 CSV 的实像点算虚像点（假设像素→设备系无误差）
import csv


def _load_real_points(csv_path: Path):
    """从 CSV 读一组前后实像点（取第一行作为基线光路输入）。"""
    with csv_path.open(newline="", encoding="utf-8") as f:
        r = next(csv.DictReader(f))
    fr = np.asarray(json.loads(r["front_real_point_device_mm"]))
    rr = np.asarray(json.loads(r["rear_real_point_device_mm"]))
    return fr, rr


def mirror(point: np.ndarray, mirror_pt: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """点关于镜面做镜像。"""
    n = normal / np.linalg.norm(normal)
    d = -float(n @ mirror_pt)
    signed = float(n @ point + d)
    return point - 2.0 * signed * n


def rot_x(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def point_to_line_distance(T: np.ndarray, P1: np.ndarray, P2: np.ndarray) -> float:
    """靶点 T 到过 P1、P2 的直线的距离: H = |L × w| / |L|。"""
    L = P2 - P1
    w = T - P1
    return float(np.linalg.norm(np.cross(L, w)) / np.linalg.norm(L))


def compute_H(T: np.ndarray, front_real: np.ndarray, rear_real: np.ndarray,
              f_pt: np.ndarray, f_n: np.ndarray,
              r_pt: np.ndarray, r_n: np.ndarray) -> float:
    """给定靶点和镜面参数，算点到激光轴的距离。

    步骤：实像点 → 镜像 → 虚像点 P1/P2 → 点到直线距离。
    实像点假设无误差（用户要求跳过像素→设备系这一步）。
    """
    P1 = mirror(front_real, f_pt, f_n)  # 前虚像点
    P2 = mirror(rear_real, r_pt, r_n)   # 后虚像点
    return point_to_line_distance(T, P1, P2)


def main():
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir.parent.parent / "dataset" / "samples" / "spot-measurements.csv"
    out_dir = base_dir

    front_real, rear_real = _load_real_points(csv_path)

    # ---- 真值基线 ----
    H_A_true = compute_H(T_A, front_real, rear_real, P1_TRUE, MIRROR_N, P2_TRUE, MIRROR_N)
    H_B_true = compute_H(T_B, front_real, rear_real, P1_TRUE, MIRROR_N, P2_TRUE, MIRROR_N)
    dH_true = H_A_true - H_B_true
    print(f"真值: H_A = {H_A_true:.4f} mm,  H_B = {H_B_true:.4f} mm,  ΔH = {dH_true:.4f} mm")
    print()

    # ---- 逐参数分析：单次灵敏度 vs 差值灵敏度 ----
    # 对每个参数施加 +1 单位误差（mm 或 deg），算：
    #   单次误差_A = H_A_pert - H_A_true
    #   单次误差_B = H_B_pert - H_B_true
    #   差值误差  = (H_A_pert - H_B_pert) - (H_A_true - H_B_true) = 误差_A - 误差_B
    #   抵消率    = 1 - |差值误差| / max(|误差_A|, |误差_B|)

    delta_mm = 1.0   # 1mm 误差
    delta_deg = 1.0   # 1° 误差

    perturbations = [
        # (名称, 单位, 误差量, 镜面/靶点, 哪个镜, 方向/参数)
        ("P1_dx", "mm", delta_mm, "mirror", "P1", "x"),
        ("P1_dy", "mm", delta_mm, "mirror", "P1", "y"),
        ("P1_dz", "mm", delta_mm, "mirror", "P1", "z"),
        ("P1_rx", "deg", delta_deg, "mirror_rot", "P1", "x"),
        ("P1_ry", "deg", delta_deg, "mirror_rot", "P1", "y"),
        ("P2_dx", "mm", delta_mm, "mirror", "P2", "x"),
        ("P2_dy", "mm", delta_mm, "mirror", "P2", "y"),
        ("P2_dz", "mm", delta_mm, "mirror", "P2", "z"),
        ("P2_rx", "deg", delta_deg, "mirror_rot", "P2", "x"),
        ("P2_ry", "deg", delta_deg, "mirror_rot", "P2", "y"),
        ("T_dx", "mm", delta_mm, "target", None, "x"),
        ("T_dy", "mm", delta_mm, "target", None, "y"),
        ("T_dz", "mm", delta_mm, "target", None, "z"),
        ("T_dL", "mm", delta_mm, "target_len", None, None),
    ]

    results = []
    print(f"{'参数':>8} {'单位':>4} {'误差A(μm)':>10} {'误差B(μm)':>10} {'差值误差(μm)':>12} {'抵消率':>8}")
    print("-" * 65)

    for name, unit, delta, kind, which, axis in perturbations:
        # 施加误差
        f_pt = P1_TRUE.copy(); f_n = MIRROR_N.copy()
        r_pt = P2_TRUE.copy(); r_n = MIRROR_N.copy()
        T_A_p = T_A.copy(); T_B_p = T_B.copy()

        if kind == "mirror":
            offset = np.zeros(3)
            offset["xyz".index(axis)] = delta
            if which == "P1": f_pt = P1_TRUE + offset
            else: r_pt = P2_TRUE + offset
        elif kind == "mirror_rot":
            if which == "P1":
                if axis == "x": f_n = rot_x(delta) @ MIRROR_N
                else: f_n = rot_y(delta) @ MIRROR_N
            else:
                if axis == "x": r_n = rot_x(delta) @ MIRROR_N
                else: r_n = rot_y(delta) @ MIRROR_N
        elif kind == "target":
            offset = np.zeros(3)
            offset["xyz".index(axis)] = delta
            T_A_p = T_A + offset
            T_B_p = T_B + offset  # 靶点误差对两次测量相同
        elif kind == "target_len":
            # 杆长误差 → 靶点沿 z 偏移 -delta
            T_A_p = T_A + np.array([0, 0, -delta])
            T_B_p = T_B + np.array([0, 0, -delta])

        H_A_p = compute_H(T_A_p, front_real, rear_real, f_pt, f_n, r_pt, r_n)
        H_B_p = compute_H(T_B_p, front_real, rear_real, f_pt, f_n, r_pt, r_n)

        err_A = (H_A_p - H_A_true) * 1000  # μm
        err_B = (H_B_p - H_B_true) * 1000
        dH_err = ((H_A_p - H_B_p) - dH_true) * 1000
        max_single = max(abs(err_A), abs(err_B))
        cancel = 1 - abs(dH_err) / max_single if max_single > 1e-9 else 1.0

        results.append({
            "param": name, "unit": unit, "delta": delta,
            "err_A_um": float(err_A), "err_B_um": float(err_B),
            "differential_err_um": float(dH_err),
            "cancellation": float(cancel),
        })
        print(f"{name:>8} {unit:>4} {err_A:>10.1f} {err_B:>10.1f} {dH_err:>12.1f} {cancel:>8.1%}")

    # ---- 随位置间距变化：差值灵敏度 vs 间距 ----
    print()
    print("=== 差值灵敏度随位置间距变化 ===")
    print(f"{'间距mm':>8} {'T_dx差值':>10} {'T_dy差值':>10} {'T_dz差值':>10} {'T_dL差值':>10} "
          f"{'P1_dx差值':>10} {'P1_dz差值':>10}")
    print("-" * 75)

    spacing_results = []
    for spacing in [0, 50, 100, 200, 300, 500, 800, 1200]:
        T_B_var = T_A + np.array([spacing, 0, 0])
        dH_true_var = H_A_true - compute_H(T_B_var, front_real, rear_real,
                                           P1_TRUE, MIRROR_N, P2_TRUE, MIRROR_N)
        row = {"spacing_mm": spacing, "dH_true_mm": float(dH_true_var)}

        for pname, apply_fn in [
            ("T_dx", lambda t: t + np.array([1, 0, 0])),
            ("T_dy", lambda t: t + np.array([0, 1, 0])),
            ("T_dz", lambda t: t + np.array([0, 0, 1])),
            ("T_dL", lambda t: t + np.array([0, 0, -1])),
            ("P1_dx", None),
            ("P1_dz", None),
        ]:
            if apply_fn is not None:
                # 靶点类误差：两次测量靶点都加同样误差
                H_A_p = compute_H(apply_fn(T_A), front_real, rear_real, P1_TRUE, MIRROR_N, P2_TRUE, MIRROR_N)
                H_B_p = compute_H(apply_fn(T_B_var), front_real, rear_real, P1_TRUE, MIRROR_N, P2_TRUE, MIRROR_N)
            else:
                # 镜面类误差
                if pname == "P1_dx":
                    f_pt = P1_TRUE + np.array([1, 0, 0])
                else:
                    f_pt = P1_TRUE + np.array([0, 0, 1])
                H_A_p = compute_H(T_A, front_real, rear_real, f_pt, MIRROR_N, P2_TRUE, MIRROR_N)
                H_B_p = compute_H(T_B_var, front_real, rear_real, f_pt, MIRROR_N, P2_TRUE, MIRROR_N)

            dH_p = H_A_p - H_B_p
            dH_err = (dH_p - dH_true_var) * 1000  # μm
            row[pname + "_dH_err_um"] = float(dH_err)

        spacing_results.append(row)
        print(f"{spacing:>8} {row['T_dx_dH_err_um']:>10.1f} {row['T_dy_dH_err_um']:>10.1f} "
              f"{row['T_dz_dH_err_um']:>10.1f} {row['T_dL_dH_err_um']:>10.1f} "
              f"{row['P1_dx_dH_err_um']:>10.1f} {row['P1_dz_dH_err_um']:>10.1f}")

    # ---- 蒙特卡罗：差值误差分布 ----
    print()
    print("=== 蒙特卡罗差值误差（参数联合扰动）===")
    rng = np.random.default_rng(42)
    N = 5000
    pos_range = 2.0   # mm
    ang_range = 1.0   # deg

    mc_samples = []
    for _ in range(N):
        # 镜面参数误差（共享）
        dP1 = rng.uniform(-pos_range, pos_range, 3)
        dP2 = rng.uniform(-pos_range, pos_range, 3)
        dP1_ang = rng.uniform(-ang_range, ang_range, 2)
        dP2_ang = rng.uniform(-ang_range, ang_range, 2)
        # 靶点误差（共享）
        dT = rng.uniform(-pos_range, pos_range, 3)
        dL = rng.uniform(-pos_range, pos_range)

        f_pt = P1_TRUE + dP1
        r_pt = P2_TRUE + dP2
        f_n = MIRROR_N.copy()
        r_n = MIRROR_N.copy()
        if dP1_ang[0] != 0: f_n = rot_x(dP1_ang[0]) @ f_n
        if dP1_ang[1] != 0: f_n = rot_y(dP1_ang[1]) @ f_n
        if dP2_ang[0] != 0: r_n = rot_x(dP2_ang[0]) @ r_n
        if dP2_ang[1] != 0: r_n = rot_y(dP2_ang[1]) @ r_n

        T_A_p = T_A + np.array([dT[0], dT[1], dT[2] - dL])
        T_B_p = T_B + np.array([dT[0], dT[1], dT[2] - dL])

        H_A_p = compute_H(T_A_p, front_real, rear_real, f_pt, f_n, r_pt, r_n)
        H_B_p = compute_H(T_B_p, front_real, rear_real, f_pt, f_n, r_pt, r_n)

        err_A = (H_A_p - H_A_true) * 1000
        err_B = (H_B_p - H_B_true) * 1000
        dH_err = ((H_A_p - H_B_p) - dH_true) * 1000

        mc_samples.append({
            "err_A_um": float(err_A),
            "err_B_um": float(err_B),
            "dH_err_um": float(dH_err),
        })

    mc_arr = np.array([[s["err_A_um"], s["err_B_um"], s["dH_err_um"]] for s in mc_samples])
    print(f"  N = {N}, 参数扰动范围 ±{pos_range}mm / ±{ang_range}°")
    print(f"  单次误差 A: abs_mean = {np.abs(mc_arr[:,0]).mean():.1f}μm, std = {mc_arr[:,0].std():.1f}μm, max = {np.abs(mc_arr[:,0]).max():.1f}μm")
    print(f"  单次误差 B: abs_mean = {np.abs(mc_arr[:,1]).mean():.1f}μm, std = {mc_arr[:,1].std():.1f}μm, max = {np.abs(mc_arr[:,1]).max():.1f}μm")
    print(f"  差值误差:   abs_mean = {np.abs(mc_arr[:,2]).mean():.1f}μm, std = {mc_arr[:,2].std():.1f}μm, max = {np.abs(mc_arr[:,2]).max():.1f}μm")
    print(f"  抵消率 (1 - |差值|/max(|A|,|B|)): {1 - np.abs(mc_arr[:,2]).mean() / max(np.abs(mc_arr[:,0]).mean(), np.abs(mc_arr[:,1]).mean()):.1%}")

    # ---- 输出 JSON ----
    out = {
        "true": {"H_A_mm": H_A_true, "H_B_mm": H_B_true, "dH_mm": dH_true,
                 "T_A": list(T_A), "T_B": list(T_B)},
        "single_param": results,
        "spacing_scan": spacing_results,
        "mc": {
            "N": N, "pos_range_mm": pos_range, "ang_range_deg": ang_range,
            "err_A_abs_mean_um": float(np.abs(mc_arr[:, 0]).mean()),
            "err_B_abs_mean_um": float(np.abs(mc_arr[:, 1]).mean()),
            "dH_err_abs_mean_um": float(np.abs(mc_arr[:, 2]).mean()),
            "dH_err_std_um": float(mc_arr[:, 2].std()),
            "dH_err_max_um": float(np.abs(mc_arr[:, 2]).max()),
            "cancellation": float(1 - np.abs(mc_arr[:, 2]).mean() / max(np.abs(mc_arr[:, 0]).mean(), np.abs(mc_arr[:, 1]).mean())),
        },
    }
    (out_dir / "mc_differential.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n输出: {out_dir / 'mc_differential.json'}")


if __name__ == "__main__":
    main()
