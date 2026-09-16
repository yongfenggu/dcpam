"""端到端蒙特卡罗灵敏度分析（三类影响因素）。

数据来源：dataset/samples/spot-measurements.csv（173 行真实采样）。
每一行包含一对前/后相机的像素光斑坐标 + 设备系实像点 + 靶点 + 距离。

三类影响因素：
  1. 光斑提取：像素中心 (u,v) 各方向偏移对最终距离的影响
  2. 反射镜安装：前/后反射镜 point(3) + normal(2 自由度) 扰动
  3. 探测杆（靶点）：root(3) + length_mm 在 XYZ 不同方向误差的影响，
     并且随 length 变化而变化 → 扫多个杆长

为聚焦这三类，从像素到设备系实像点这一步按用户要求假设无误差：
直接用 CSV 里已有的 front_real_point_device_mm / rear_real_point_device_mm
作为无误差输入，只对三类因素做扰动，重跑 镜像 → 距离 计算。

输出：
  results/mc_spot.json        光斑像素偏移灵敏度
  results/mc_mirror.json      反射镜安装灵敏度
  results/mc_probe_rod.json   探测杆 XYZ 方向灵敏度（多杆长扫描）
  results/summary.md          汇总表
"""
from __future__ import annotations

import csv
import json
import re
import statistics as st
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---- 基线几何（与 config.toml / defaults.py 对齐）----
FRONT_MIRROR_PT = np.array([0.0, 0.0, 23.0])
REAR_MIRROR_PT  = np.array([80.0, 0.0, 23.0])
MIRROR_NORMAL  = np.array([np.sqrt(2)/2, 0.0, np.sqrt(2)/2])  # 45°

PROBE_ROOT = np.array([41.0, 0.0, -132.0])
PROBE_LENGTH_BASE = 586.051  # defaults.py 里的基线杆长

# 光斑像素 → 设备系实像点 的近似雅可比（用反投影线性化估计）
# 从 back_projection.py：pixel → normalized → undistort → ray → plane intersect
# 实像面距相机约 31~38 mm，焦距约 3000 px，故 1 px ≈ 31/3000 mm ≈ 0.0103 mm
# 但这个雅可比与具体像素位置有关；这里用 CSV 数据直接数值估计（见 _estimate_pixel_jacobian）
PIXEL_TO_MM_FALLBACK = 0.0103  # mm/px，仅作 fallback


@dataclass
class Sample:
    """CSV 一行：一对前后相机像素坐标 + 设备系实像点 + 靶点 + 基线距离。"""
    name: str
    group: str
    spot_input: np.ndarray       # [fu, fv, ru, rv, length]
    front_real: np.ndarray       # (3,) 设备系前实像点
    rear_real: np.ndarray        # (3,) 设备系后实像点
    target: np.ndarray           # (3,) 靶点
    base_distance: float         # 基线距离 mm


def load_samples(csv_path: Path) -> list[Sample]:
    rows: list[Sample] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            g = re.match(r"(L\d+D\d+)", r["name"]).group(1)
            rows.append(Sample(
                name=r["name"], group=g,
                spot_input=np.asarray(json.loads(r["spot_input_vector"]), dtype=np.float64),
                front_real=np.asarray(json.loads(r["front_real_point_device_mm"]), dtype=np.float64),
                rear_real=np.asarray(json.loads(r["rear_real_point_device_mm"]), dtype=np.float64),
                target=np.asarray(json.loads(r["target_point_device_mm"]), dtype=np.float64),
                base_distance=float(r["distance_mm"]),
            ))
    return rows


# ---- 核心几何：镜像 + 点到直线距离（与 steps/mirror_transform.py、distance.py 一致）----
def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _mirror(point: np.ndarray, mirror_pt: np.ndarray, normal: np.ndarray) -> np.ndarray:
    n = _unit(normal)
    d = -float(n @ mirror_pt)
    signed = float(n @ point + d)
    return point - 2.0 * signed * n


def _point_to_line_distance(target: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    at = target - a
    cross = np.cross(ab, at)
    return float(np.linalg.norm(cross) / np.linalg.norm(ab))


def _distance_with_perturbation(
    s: Sample,
    f_mirror_pt: np.ndarray, f_mirror_n: np.ndarray,
    r_mirror_pt: np.ndarray, r_mirror_n: np.ndarray,
    target: np.ndarray,
) -> float:
    """给定扰动后的镜面参数和靶点，算一次距离。实像点用 CSV 基线值（假设无误差）。"""
    f_virt = _mirror(s.front_real, f_mirror_pt, f_mirror_n)
    r_virt = _mirror(s.rear_real, r_mirror_pt, r_mirror_n)
    return _point_to_line_distance(target, f_virt, r_virt)


# =====================================================================
# 1. 光斑提取：像素偏移灵敏度
# =====================================================================
# 用户要求：像素中心偏离几个像素，每偏 1 px 对最终精度多大影响。
# CSV 里有 spot_input（像素坐标），但没有保存从像素→设备系的中间量供我们直接扰动像素再重算。
# 做法：估计像素→设备系实像点的雅可比矩阵 J（4×6：fu,fv,ru,rv → fx,fy,fz,rx,ry,rz），
# 然后对每个像素方向加 Δpx 扰动，用 J 映射到设备系实像点偏移，再跑镜像+距离。

def _estimate_pixel_jacobian(samples: list[Sample], delta_px: float = 1.0) -> np.ndarray:
    """数值估计 4 维像素 → 6 维设备系实像点的雅可比。

    用各样本对像素坐标的差分与对应设备系实像点的差分做最小二乘拟合。
    返回 J shape (6, 4)：device_delta = J @ pixel_delta。
    """
    # 构造回归：对每对相邻样本（按 group 分组内）求像素差和设备系差
    groups: dict[str, list[Sample]] = {}
    for s in samples:
        groups.setdefault(s.group, []).append(s)

    X_rows = []  # 像素差
    Y_rows = []  # 设备系差
    for g, grp in groups.items():
        if len(grp) < 2:
            continue
        grp_sorted = sorted(grp, key=lambda x: x.name)
        for i in range(len(grp_sorted) - 1):
            a, b = grp_sorted[i], grp_sorted[i + 1]
            dpix = np.array([
                a.spot_input[0] - b.spot_input[0],
                a.spot_input[1] - b.spot_input[1],
                a.spot_input[2] - b.spot_input[2],
                a.spot_input[3] - b.spot_input[3],
            ])
            ddev = np.concatenate([a.front_real - b.front_real, a.rear_real - b.rear_real])
            if np.linalg.norm(dpix) < 1e-9:
                continue
            X_rows.append(dpix)
            Y_rows.append(ddev)

    X = np.asarray(X_rows)  # (N, 4)
    Y = np.asarray(Y_rows)  # (N, 6)
    # 最小二乘：Y = X @ J^T  →  J^T = pinv(X) @ Y → J = (pinv(X) @ Y).T
    Jt, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return Jt.T  # (6, 4)


def mc_spot_sensitivity(samples: list[Sample], pixel_deltas: list[float],
                        jacobian: np.ndarray, n_mc: int = 2000, seed: int = 42) -> dict:
    """每个像素方向单独偏移 Δpx，统计距离偏差。

    返回：每个像素方向（fu, fv, ru, rv）在每个 Δpx 下的距离误差均值/std，
    以及 4 维联合 MC 采样的分布。
    """
    rng = np.random.default_rng(seed)
    base_dist = np.array([s.base_distance for s in samples])
    results: dict[str, list[dict]] = {}

    # 单变量扫描：每 1 px
    pixel_names = ["front_u", "front_v", "rear_u", "rear_v"]
    for idx, name in enumerate(pixel_names):
        rows = []
        for dpx in pixel_deltas:
            errs = []
            for s in samples:
                # 像素偏移 → 设备系实像点偏移
                dpix = np.zeros(4)
                dpix[idx] = dpx
                ddev = jacobian @ dpix  # (6,)
                front_real_p = s.front_real + ddev[:3]
                rear_real_p = s.rear_real + ddev[3:]
                f_virt = _mirror(front_real_p, FRONT_MIRROR_PT, MIRROR_NORMAL)
                r_virt = _mirror(rear_real_p, REAR_MIRROR_PT, MIRROR_NORMAL)
                d = _point_to_line_distance(s.target, f_virt, r_virt)
                errs.append(d - s.base_distance)
            errs = np.asarray(errs)
            rows.append({
                "delta_px": dpx,
                "mean_err_um": float(np.mean(errs) * 1000),
                "std_err_um": float(np.std(errs) * 1000),
                "abs_mean_um": float(np.mean(np.abs(errs)) * 1000),
            })
        results[name] = {"unit": "px", "rows": rows}

    # 联合 MC：4 维同时随机偏移（-1~1 px 均匀）
    mc_samples = []
    for _ in range(n_mc):
        dpix = rng.uniform(-1.0, 1.0, 4)
        errs = []
        for s in samples:
            ddev = jacobian @ dpix
            front_real_p = s.front_real + ddev[:3]
            rear_real_p = s.rear_real + ddev[3:]
            f_virt = _mirror(front_real_p, FRONT_MIRROR_PT, MIRROR_NORMAL)
            r_virt = _mirror(rear_real_p, REAR_MIRROR_PT, MIRROR_NORMAL)
            d = _point_to_line_distance(s.target, f_virt, r_virt)
            errs.append(d - s.base_distance)
        errs = np.asarray(errs)
        mc_samples.append({
            "max_abs_px": float(np.max(np.abs(dpix))),
            "mean_abs_err_um": float(np.mean(np.abs(errs)) * 1000),
            "std_err_um": float(np.std(errs) * 1000),
        })

    return {"jacobian_shape": list(jacobian.shape), "single_variate": results,
            "mc_joint_1px": {"n": n_mc, "samples": mc_samples[:200]}}


# =====================================================================
# 2. 反射镜安装灵敏度
# =====================================================================
def mc_mirror_sensitivity(samples: list[Sample], n_mc: int = 5000,
                          pos_range_mm: float = 2.0, ang_range_deg: float = 1.0,
                          seed: int = 42) -> dict:
    """前/后反射镜 point(3) + normal(2 dof) 联合 MC。

    normal 的 2 自由度用绕 x / y 轴的小角度旋转扰动。
    """
    rng = np.random.default_rng(seed)

    def rot(axis: str, deg: float) -> np.ndarray:
        a = np.deg2rad(deg)
        c, s_ = np.cos(a), np.sin(a)
        if axis == "x":
            return np.array([[1, 0, 0], [0, c, -s_], [0, s_, c]])
        if axis == "y":
            return np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]])
        raise ValueError(axis)

    base_sigmas = _group_sigmas(samples, FRONT_MIRROR_PT, MIRROR_NORMAL,
                                REAR_MIRROR_PT, MIRROR_NORMAL)

    mc_samples = []
    for _ in range(n_mc):
        dfp = rng.uniform(-pos_range_mm, pos_range_mm, 3)
        drp = rng.uniform(-pos_range_mm, pos_range_mm, 3)
        dfang = rng.uniform(-ang_range_deg, ang_range_deg, 2)
        drang = rng.uniform(-ang_range_deg, ang_range_deg, 2)

        fpt = FRONT_MIRROR_PT + dfp
        rpt = REAR_MIRROR_PT + drp
        fn = MIRROR_NORMAL.copy()
        rn = MIRROR_NORMAL.copy()
        if dfang[0] != 0: fn = rot("x", dfang[0]) @ fn
        if dfang[1] != 0: fn = rot("y", dfang[1]) @ fn
        if drang[0] != 0: rn = rot("x", drang[0]) @ rn
        if drang[1] != 0: rn = rot("y", drang[1]) @ rn
        fn = _unit(fn); rn = _unit(rn)

        errs = []
        for s in samples:
            d = _distance_with_perturbation(s, fpt, fn, rpt, rn, s.target)
            errs.append(d - s.base_distance)
        errs = np.asarray(errs)
        mc_samples.append({
            "params": {
                "front_dx": float(dfp[0]), "front_dy": float(dfp[1]), "front_dz": float(dfp[2]),
                "front_rx": float(dfang[0]), "front_ry": float(dfang[1]),
                "rear_dx": float(drp[0]), "rear_dy": float(drp[1]), "rear_dz": float(drp[2]),
                "rear_rx": float(drang[0]), "rear_ry": float(drang[1]),
            },
            "mean_abs_err_um": float(np.mean(np.abs(errs)) * 1000),
            "std_err_um": float(np.std(errs) * 1000),
            "max_abs_err_um": float(np.max(np.abs(errs)) * 1000),
        })

    # 单变量扫描：10 个自由度各扫一遍
    single = _mirror_single_variate(samples)

    return {
        "baseline": {"sigma_um": base_sigmas},
        "ranges": {"pos_mm": pos_range_mm, "ang_deg": ang_range_deg},
        "mc": {"n": n_mc, "samples": mc_samples[:500]},
        "single_variate": single,
    }


def _group_sigmas(samples: list[Sample], fpt, fn, rpt, rn) -> dict[str, float]:
    groups: dict[str, list[float]] = {}
    for s in samples:
        d = _distance_with_perturbation(s, fpt, fn, rpt, rn, s.target)
        groups.setdefault(s.group, []).append(d)
    return {g: (st.stdev(v) * 1000 if len(v) >= 2 else 0.0) for g, v in groups.items()}


def _mirror_single_variate(samples: list[Sample]) -> dict:
    deltas_pos = [-2.0, -1.0, -0.5, -0.1, 0.1, 0.5, 1.0, 2.0]
    deltas_ang = [-1.0, -0.5, -0.1, 0.1, 0.5, 1.0]

    def rot(axis, deg):
        a = np.deg2rad(deg); c, s_ = np.cos(a), np.sin(a)
        if axis == "x": return np.array([[1,0,0],[0,c,-s_],[0,s_,c]])
        return np.array([[c,0,s_],[0,1,0],[-s_,0,c]])

    dofs = [
        ("front_dx", "front", "pos", 0), ("front_dy", "front", "pos", 1), ("front_dz", "front", "pos", 2),
        ("front_rx", "front", "rot", "x"), ("front_ry", "front", "rot", "y"),
        ("rear_dx", "rear", "pos", 0), ("rear_dy", "rear", "pos", 1), ("rear_dz", "rear", "pos", 2),
        ("rear_rx", "rear", "rot", "x"), ("rear_ry", "rear", "rot", "y"),
    ]
    out = {}
    for dof, side, kind, axis in dofs:
        deltas = deltas_pos if kind == "pos" else deltas_ang
        rows = []
        for d in deltas:
            fp, rp = FRONT_MIRROR_PT.copy(), REAR_MIRROR_PT.copy()
            fn, rn = MIRROR_NORMAL.copy(), MIRROR_NORMAL.copy()
            if side == "front":
                if kind == "pos": fp[axis] += d
                else: fn = _unit(rot(axis, d) @ fn)
            else:
                if kind == "pos": rp[axis] += d
                else: rn = _unit(rot(axis, d) @ rn)
            errs = []
            for s in samples:
                dist = _distance_with_perturbation(s, fp, fn, rp, rn, s.target)
                errs.append(dist - s.base_distance)
            errs = np.asarray(errs)
            rows.append({
                "delta": d,
                "unit": "mm" if kind == "pos" else "deg",
                "mean_err_um": float(np.mean(errs) * 1000),
                "abs_mean_um": float(np.mean(np.abs(errs)) * 1000),
            })
        out[dof] = rows
    return out


# =====================================================================
# 3. 探测杆（靶点）灵敏度：root XYZ 误差 + length 误差，随杆长变化
# =====================================================================
def mc_probe_rod_sensitivity(samples: list[Sample], n_mc: int = 3000,
                              root_err_mm: float = 2.0, length_err_mm: float = 2.0,
                              seed: int = 42) -> dict:
    """靶点 = root + [0, 0, -length]。

    分析：
      a) root 在 X/Y/Z 三个方向各有误差 Δ → 靶点同向偏移 Δ（直接叠加）
      b) length 误差 ΔL → 靶点沿 Z 偏移 -ΔL
      c) 随 length 变化：扫描 length = 100~1200 mm，看各方向误差对距离的灵敏度

    关键：靶点误差对最终距离的影响 = 靶点偏移在"垂直于激光轴"方向上的投影。
    激光轴方向随样本（前后实像点镜像后）变化，故不同样本的灵敏度不同。
    """
    rng = np.random.default_rng(seed)

    # ---- a) 单方向扫描：root X/Y/Z 和 length 各偏 1 mm ----
    directions = [
        ("root_x", np.array([1.0, 0, 0]), 0.0),
        ("root_y", np.array([0, 1.0, 0]), 0.0),
        ("root_z", np.array([0, 0, 1.0]), 0.0),
        ("length", np.array([0, 0, -1.0]), 0.0),  # ΔL=+1 → 靶点 Z 减 1
    ]
    single = {}
    for name, dir_vec, _ in directions:
        rows = []
        for delta_mm in [-2.0, -1.0, -0.5, -0.1, 0.1, 0.5, 1.0, 2.0]:
            errs = []
            for s in samples:
                target_p = s.target + dir_vec * delta_mm
                d = _distance_with_perturbation(s, FRONT_MIRROR_PT, MIRROR_NORMAL,
                                                 REAR_MIRROR_PT, MIRROR_NORMAL, target_p)
                errs.append(d - s.base_distance)
            errs = np.asarray(errs)
            rows.append({
                "delta_mm": delta_mm,
                "mean_err_um": float(np.mean(errs) * 1000),
                "abs_mean_um": float(np.mean(np.abs(errs)) * 1000),
                "std_um": float(np.std(errs) * 1000),
            })
        single[name] = rows

    # ---- b) 联合 MC：root XYZ + length 同时扰动 ----
    mc_samples = []
    for _ in range(n_mc):
        dr = rng.uniform(-root_err_mm, root_err_mm, 3)
        dL = rng.uniform(-length_err_mm, length_err_mm)
        errs = []
        for s in samples:
            target_p = s.target + np.array([dr[0], dr[1], dr[2] - dL])
            d = _distance_with_perturbation(s, FRONT_MIRROR_PT, MIRROR_NORMAL,
                                             REAR_MIRROR_PT, MIRROR_NORMAL, target_p)
            errs.append(d - s.base_distance)
        errs = np.asarray(errs)
        mc_samples.append({
            "dr_x": float(dr[0]), "dr_y": float(dr[1]), "dr_z": float(dr[2]), "dL": float(dL),
            "abs_mean_um": float(np.mean(np.abs(errs)) * 1000),
            "std_um": float(np.std(errs) * 1000),
            "max_um": float(np.max(np.abs(errs)) * 1000),
        })

    # ---- c) 随杆长变化：length = 100~1200 mm，每 100 mm 一档 ----
    length_scan = []
    for L in range(100, 1300, 100):
        # 用此 L 重新算靶点（root 不变），统计各方向 1mm 误差的灵敏度
        target_at_L = PROBE_ROOT + np.array([0, 0, -L])
        # 计算基线距离（用此 target）
        base_dists = []
        for s in samples:
            d = _distance_with_perturbation(s, FRONT_MIRROR_PT, MIRROR_NORMAL,
                                             REAR_MIRROR_PT, MIRROR_NORMAL, target_at_L)
            base_dists.append(d)
        base_dists = np.asarray(base_dists)

        dir_sens = {}
        for name, dir_vec, _ in directions:
            errs = []
            for s in samples:
                tp = target_at_L + dir_vec * 1.0  # +1 mm
                d = _distance_with_perturbation(s, FRONT_MIRROR_PT, MIRROR_NORMAL,
                                                 REAR_MIRROR_PT, MIRROR_NORMAL, tp)
                errs.append(d - base_dists[len(errs)])
            errs = np.asarray(errs)
            dir_sens[name] = float(np.mean(np.abs(errs)) * 1000)  # μm per mm
        length_scan.append({"length_mm": L, "sens_um_per_mm": dir_sens})

    return {
        "single_variate": single,
        "mc_joint": {"n": n_mc, "samples": mc_samples[:500]},
        "length_scan": length_scan,
    }


# =====================================================================
# main
# =====================================================================
def main() -> None:
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir.parent.parent / "dataset" / "samples" / "spot-measurements.csv"
    out_dir = base_dir  # 输出直接放分析目录下（results/ 被 gitignore 忽略）
    out_dir.mkdir(exist_ok=True)

    samples = load_samples(csv_path)
    print(f"读入 {len(samples)} 样本，{len({s.group for s in samples})} 组")

    # 基线校验
    base_errs = []
    for s in samples:
        d = _distance_with_perturbation(s, FRONT_MIRROR_PT, MIRROR_NORMAL,
                                         REAR_MIRROR_PT, MIRROR_NORMAL, s.target)
        base_errs.append(d - s.base_distance)
    print(f"基线复算残差: mean={np.mean(base_errs)*1000:.2f}μm std={np.std(base_errs)*1000:.2f}μm")

    # 1. 光斑
    print("\n[1/3] 光斑像素偏移灵敏度...")
    J = _estimate_pixel_jacobian(samples)
    print(f"  雅可比 shape={J.shape}")
    print(f"  雅可比范数(每列)={np.linalg.norm(J, axis=0)}")
    spot_res = mc_spot_sensitivity(samples, [-2, -1, -0.5, -0.1, 0.1, 0.5, 1, 2], J)
    (out_dir / "mc_spot.json").write_text(json.dumps(spot_res, ensure_ascii=False, indent=2))
    print(f"  单像素灵敏度(1px):")
    for name in ["front_u", "front_v", "rear_u", "rear_v"]:
        r1 = [r for r in spot_res["single_variate"][name]["rows"] if r["delta_px"] == 1.0][0]
        rm1 = [r for r in spot_res["single_variate"][name]["rows"] if r["delta_px"] == -1.0][0]
        print(f"    {name}: +1px → {r1['abs_mean_um']:.1f}μm  -1px → {rm1['abs_mean_um']:.1f}μm")

    # 2. 反射镜
    print("\n[2/3] 反射镜安装灵敏度...")
    mirror_res = mc_mirror_sensitivity(samples, n_mc=3000, pos_range_mm=2.0, ang_range_deg=1.0)
    (out_dir / "mc_mirror.json").write_text(json.dumps(mirror_res, ensure_ascii=False, indent=2))
    print("  单变量灵敏度(1mm/1deg):")
    for dof, rows in mirror_res["single_variate"].items():
        r1 = [r for r in rows if abs(r["delta"]) == 1.0][0]
        print(f"    {dof}: {r1['abs_mean_um']:.1f}μm")

    # 3. 探测杆
    print("\n[3/3] 探测杆灵敏度...")
    rod_res = mc_probe_rod_sensitivity(samples, n_mc=3000)
    (out_dir / "mc_probe_rod.json").write_text(json.dumps(rod_res, ensure_ascii=False, indent=2))
    print("  各方向 1mm 误差灵敏度:")
    for name in ["root_x", "root_y", "root_z", "length"]:
        r1 = [r for r in rod_res["single_variate"][name] if r["delta_mm"] == 1.0][0]
        print(f"    {name}: {r1['abs_mean_um']:.1f}μm/mm")
    print("  随杆长变化(μm/mm):")
    for entry in rod_res["length_scan"]:
        s = entry["sens_um_per_mm"]
        print(f"    L={entry['length_mm']:>4}mm: x={s['root_x']:.1f} y={s['root_y']:.1f} "
              f"z={s['root_z']:.1f} L={s['length']:.1f}")

    # 汇总 md
    _write_summary(out_dir / "summary.md", spot_res, mirror_res, rod_res, samples)
    print(f"\n输出目录: {out_dir}")


def _write_summary(path: Path, spot, mirror, rod, samples) -> None:
    lines = [
        "# 蒙特卡罗灵敏度分析汇总",
        "",
        f"数据: {len(samples)} 样本, {len({s.group for s in samples})} 组",
        "",
        "## 1. 光斑提取（像素偏移）",
        "",
        "| 方向 | +1px (μm) | -1px (μm) | +2px (μm) |",
        "|------|-----------|-----------|-----------|",
    ]
    for name in ["front_u", "front_v", "rear_u", "rear_v"]:
        rows = {r["delta_px"]: r for r in spot["single_variate"][name]["rows"]}
        lines.append(f"| {name} | {rows[1.0]['abs_mean_um']:.1f} | {rows[-1.0]['abs_mean_um']:.1f} | {rows[2.0]['abs_mean_um']:.1f} |")
    lines += ["", "## 2. 反射镜安装", "",
              "| 自由度 | 1mm/1deg (μm) |", "|--------|--------------|"]
    for dof, rows in mirror["single_variate"].items():
        r1 = [r for r in rows if abs(r["delta"]) == 1.0][0]
        lines.append(f"| {dof} | {r1['abs_mean_um']:.1f} |")
    lines += ["", "## 3. 探测杆（靶点）", "",
              "| 方向 | 1mm 误差 (μm) |", "|------|--------------|"]
    for name in ["root_x", "root_y", "root_z", "length"]:
        r1 = [r for r in rod["single_variate"][name] if r["delta_mm"] == 1.0][0]
        lines.append(f"| {name} | {r1['abs_mean_um']:.1f} |")
    lines += ["", "### 随杆长变化（μm per mm 误差）", "",
              "| 杆长 mm | root_x | root_y | root_z | length |",
              "|---------|--------|--------|--------|--------|"]
    for e in rod["length_scan"]:
        s = e["sens_um_per_mm"]
        lines.append(f"| {e['length_mm']} | {s['root_x']:.1f} | {s['root_y']:.1f} | {s['root_z']:.1f} | {s['length']:.1f} |")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
