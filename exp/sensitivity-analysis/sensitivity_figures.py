"""生成论文级灵敏度分析图（3 张）。

图1: 单次灵敏度 vs 差值灵敏度对比（水平柱状图）
图2: 杠杆放大机制示意图（几何图解）
图3: 非对称度随误差幅度变化
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Arc
from pathlib import Path

import sensitivity_theory as st

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)


def fig1_sensitivity_comparison():
    """图1: 13 个自由度的单次 vs 差值灵敏度对比。"""
    results, HA, HB, dH = st.compute_all_sensitivities()

    labels_map = {
        "P1_dx": r"$P_{1,\delta x}$", "P1_dz": r"$P_{1,\delta z}$",
        "P1_rx": r"$P_{1,\theta x}$", "P1_ry": r"$P_{1,\theta y}$", "P1_rz": r"$P_{1,\theta z}$",
        "P2_dx": r"$P_{2,\delta x}$", "P2_dz": r"$P_{2,\delta z}$",
        "P2_rx": r"$P_{2,\theta x}$", "P2_ry": r"$P_{2,\theta y}$", "P2_rz": r"$P_{2,\theta z}$",
        "T_dx": r"$T_{\delta x}$", "T_dy": r"$T_{\delta y}$", "T_dz": r"$T_{\delta z}$",
    }
    groups = [
        ("Target", ["T_dz", "T_dx", "T_dy"], "#2ca02c"),
        ("Front mirror", ["P1_dx", "P1_dz", "P1_ry", "P1_rx", "P1_rz"], "#1f77b4"),
        ("Rear mirror", ["P2_dx", "P2_dz", "P2_ry", "P2_rx", "P2_rz"], "#ff7f0e"),
    ]

    items = []
    for gname, params, color in groups:
        for p in params:
            r = next(x for x in results if x["param"] == p)
            items.append((labels_map[p], r["unified"], r["dH_first"], gname, color))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    y_pos = np.arange(len(items))
    bar_h = 0.6

    # 左图：单次测量灵敏度
    for i, (label, single, diff, gname, color) in enumerate(items):
        ax1.barh(i, single, height=bar_h, color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([it[0] for it in items])
    ax1.set_xlabel(r"Sensitivity $|\partial H/\partial p|$ ($\mu$m / unit)")
    ax1.set_title("(a) Single measurement")
    ax1.axvline(0, color="gray", linewidth=0.5)
    ax1.set_xscale("symlog", linthresh=1)
    ax1.set_xlim(0, 2000)
    ax1.grid(axis="x", alpha=0.3)

    # 右图：差值测量灵敏度
    for i, (label, single, diff, gname, color) in enumerate(items):
        ax2.barh(i, diff, height=bar_h, color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax2.set_xlabel(r"Differential sensitivity $|\partial \Delta H/\partial p|$ ($\mu$m / unit)")
    ax2.set_title("(b) Differential measurement")
    ax2.axvline(0, color="gray", linewidth=0.5)
    ax2.set_xscale("symlog", linthresh=1)
    ax2.set_xlim(0, 10000)
    ax2.grid(axis="x", alpha=0.3)

    # 图例
    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor=g[2], label=g[0]) for g in groups]
    ax2.legend(handles=legend_elems, loc="lower right", framealpha=0.9)

    fig.suptitle("Sensitivity: Single vs. Differential Measurement", y=1.02, fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_sensitivity_comparison.pdf")
    plt.savefig(FIG_DIR / "fig1_sensitivity_comparison.png")
    plt.close()
    print("  fig1 saved")


def fig2_lever_mechanism():
    """图2: 杠杆放大机制示意图（不按比例，分上下两面板）。

    上板: 镜面区域特写——偏移 δM → 虚像点偏 [1,0,1] → 激光轴偏转 θ
    下板: 靶点区域——偏转轴对近/远靶点垂距变化不同 → 差值放大
    """
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7, 6.5),
        gridspec_kw={"height_ratios": [1, 1.2], "hspace": 0.35},
    )

    # ====== 上板：镜面区域特写（x-z 平面，不按比例）======
    # 真值激光轴：P1=(0,0,23) → P2=(80,0,23.−2.7) → 简化为水平
    P1 = np.array([0.0, 23.0])
    P2 = np.array([80.0, 23.0])
    # 偏移后 P1'（放大 30 倍便于可视化）
    scale = 30
    dV = np.array([1, 1]) * scale  # [1,0,1] 偏移，放大
    P1p = P1 + dV

    # 真值轴
    ax_top.plot([P1[0], P2[0]], [P1[1], P2[1]], "k-", linewidth=2.5, zorder=5)
    ax_top.plot(*P1, "ks", markersize=8, zorder=6)
    ax_top.plot(*P2, "ks", markersize=8, zorder=6)
    ax_top.annotate(r"$P_1$", (P1[0], P1[1]), textcoords="offset points",
                    xytext=(-15, -8), fontsize=12)
    ax_top.annotate(r"$P_2$", (P2[0], P2[1]), textcoords="offset points",
                    xytext=(5, -8), fontsize=12)

    # 偏移后轴
    ax_top.plot([P1p[0], P2[0]], [P1p[1], P2[1]], "r--", linewidth=2, alpha=0.85, zorder=4)
    ax_top.plot(*P1p, "r^", markersize=8, zorder=5)
    ax_top.annotate(r"$P_1'$", (P1p[0], P1p[1]), textcoords="offset points",
                    xytext=(-20, 5), fontsize=11, color="red")

    # 偏移箭头
    ax_top.annotate("", xy=(P1p[0], P1p[1]), xytext=(P1[0], P1[1]),
                     arrowprops=dict(arrowstyle="->", color="red", lw=2))
    ax_top.annotate(r"$\delta M_x \rightarrow \Delta V = 2(\mathbf{n}\cdot\delta)\,\mathbf{n}$",
                    xy=(P1[0] + dV[0] * 0.5, P1[1] + dV[1] * 0.5),
                    textcoords="offset points", xytext=(10, 8), fontsize=9, color="red")

    # 偏转角 θ
    theta_vis = np.arctan2(dV[1], 80 - dV[0])  # 可视化角度
    arc_r = 18
    arc = Arc((P2[0], P2[1]), 2 * arc_r, 2 * arc_r, angle=0,
              theta2=180, theta1=180 - np.rad2deg(theta_vis) * 10,
              color="blue", linewidth=1.8)
    ax_top.add_patch(arc)
    ax_top.annotate(r"$\theta$", xy=(P2[0] - arc_r * 0.6, P2[1] + arc_r * 0.4),
                    fontsize=12, color="blue")

    # 反射镜示意（45° 斜线）
    mirror_x = np.array([-8, 8])
    mirror_z = P1[1] + mirror_x * 0  # 镜面在 P1 处
    ax_top.plot([-5, 5], [P1[1] - 5, P1[1] + 5], "gray", linewidth=3, alpha=0.5, zorder=1)
    ax_top.annotate("mirror", xy=(5, P1[1] + 5), textcoords="offset points",
                    xytext=(5, 0), fontsize=8, color="gray")

    ax_top.set_xlim(-15, 95)
    ax_top.set_ylim(10, 65)
    ax_top.set_xlabel("x (mm)")
    ax_top.set_ylabel("z (mm)")
    ax_top.set_title("(a) Mirror offset rotates the laser axis", fontsize=11)
    ax_top.set_aspect("equal")
    ax_top.grid(alpha=0.15)

    # ====== 下板：靶点区域（示意，不按比例）======
    # 支点在 x=6（对应 P2 端），T_A 在左侧（近支点），T_B 在右侧（远支点）
    x_axis_z = 0
    TA_x = 3
    TB_x = 11
    pivot_x = 6
    angle_vis = 0.06  # rad

    # 真值激光轴
    ax_bot.plot([-1, 13], [x_axis_z, x_axis_z], "k-", linewidth=2.5, zorder=5, label="True axis")
    # 偏转后轴（绕支点旋转）
    z_at = lambda x: (x - pivot_x) * np.tan(angle_vis)
    ax_bot.plot([-1, 13], [z_at(-1), z_at(13)], "r--",
                linewidth=2, alpha=0.85, zorder=4, label="Perturbed axis")

    # 支点标记
    ax_bot.plot(pivot_x, x_axis_z, "k|", markersize=12, markeredgewidth=2, zorder=6)
    ax_bot.annotate("pivot\n($P_2$)", (pivot_x, x_axis_z), textcoords="offset points",
                    xytext=(0, -18), fontsize=8, ha="center", color="gray")

    # 靶点 A（近支点，左侧）
    ax_bot.plot(TA_x, x_axis_z, "go", markersize=10, zorder=7)
    ax_bot.annotate(r"$T_A$", (TA_x, x_axis_z), textcoords="offset points",
                    xytext=(-5, -15), fontsize=12)
    # 靶点 B（远支点，右侧）
    ax_bot.plot(TB_x, x_axis_z, "go", markersize=10, zorder=7)
    ax_bot.annotate(r"$T_B$", (TB_x, x_axis_z), textcoords="offset points",
                    xytext=(-5, -15), fontsize=12)

    # 真值 H_A, H_B
    HA_vis, HB_vis = 1.5, 1.5
    ax_bot.plot([TA_x, TA_x], [x_axis_z, x_axis_z + HA_vis], "g-", linewidth=2)
    ax_bot.annotate(r"$H_A$", (TA_x, x_axis_z + HA_vis / 2),
                    textcoords="offset points", xytext=(-22, 0), fontsize=10, color="green")
    ax_bot.plot([TB_x, TB_x], [x_axis_z, x_axis_z + HB_vis], "g-", linewidth=2)
    ax_bot.annotate(r"$H_B$", (TB_x, x_axis_z + HB_vis / 2),
                    textcoords="offset points", xytext=(-22, 0), fontsize=10, color="green")

    # 偏转后 H_A'（A 在支点左侧，轴下移 → H 增大）, H_B'（B 在支点右侧，轴上移 → H 减小）
    dHA = abs(z_at(TA_x))  # 轴在 A 处下移量
    dHB = z_at(TB_x)       # 轴在 B 处上移量
    HA_new = HA_vis + dHA   # A: H 增大
    HB_new = HB_vis - dHB   # B: H 减小

    ax_bot.plot([TA_x, TA_x], [x_axis_z, x_axis_z + HA_new], "r--", linewidth=1.8, alpha=0.7)
    ax_bot.annotate(r"$H_A'$", (TA_x, x_axis_z + HA_new),
                    textcoords="offset points", xytext=(8, 0), fontsize=10, color="red")
    ax_bot.plot([TB_x, TB_x], [x_axis_z, x_axis_z + HB_new], "r--", linewidth=1.8, alpha=0.7)
    ax_bot.annotate(r"$H_B'$", (TB_x, x_axis_z + HB_new),
                    textcoords="offset points", xytext=(8, 0), fontsize=10, color="red")

    # 变化量标注
    ax_bot.annotate("", xy=(TA_x + 1.8, x_axis_z + HA_new),
                     xytext=(TA_x + 1.8, x_axis_z + HA_vis),
                     arrowprops=dict(arrowstyle="<->", color="purple", lw=1.5))
    ax_bot.annotate(r"$+\Delta H_A$", xy=(TA_x + 2.1, x_axis_z + (HA_vis + HA_new) / 2),
                    fontsize=9, color="purple")
    ax_bot.annotate("", xy=(TB_x + 1.8, x_axis_z + HB_vis),
                     xytext=(TB_x + 1.8, x_axis_z + HB_new),
                     arrowprops=dict(arrowstyle="<->", color="purple", lw=1.5))
    ax_bot.annotate(r"$-\Delta H_B$", xy=(TB_x + 2.1, x_axis_z + (HB_vis + HB_new) / 2),
                    fontsize=9, color="purple")

    # 间距 D
    ax_bot.annotate("", xy=(TA_x, -2.5), xytext=(TB_x, -2.5),
                     arrowprops=dict(arrowstyle="<->", color="darkgreen", lw=1.5))
    ax_bot.annotate(r"$D$ (spacing)", xy=((TA_x + TB_x) / 2, -2.5),
                    textcoords="offset points", xytext=(0, -12), fontsize=10,
                    color="darkgreen", ha="center")

    # 公式
    ax_bot.text(0.98, 0.02,
                r"$\Delta(\Delta H) = \Delta H_A - \Delta H_B \approx D \times \theta$"
                r" (opposite signs $\Rightarrow$ amplified)",
                transform=ax_bot.transAxes, fontsize=10, va="bottom", ha="right",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

    ax_bot.set_xlim(-1.5, 13.5)
    ax_bot.set_ylim(-4, 4)
    ax_bot.set_xlabel("x (mm, not to scale)")
    ax_bot.set_ylabel("z (mm)")
    ax_bot.set_title("(b) Axis rotation amplifies differential error", fontsize=11)
    ax_bot.set_aspect("equal")
    ax_bot.grid(alpha=0.15)
    ax_bot.legend(loc="upper left", framealpha=0.9)

    plt.savefig(FIG_DIR / "fig2_lever_mechanism.pdf")
    plt.savefig(FIG_DIR / "fig2_lever_mechanism.png")
    plt.close()
    print("  fig2 saved")


def fig3_asymmetry_and_amplification():
    """图3: (a) 非对称度随误差幅度变化; (b) 差值放大倍数随间距变化。"""
    results, HA, HB, dH = st.compute_all_sensitivities()
    fr, rr = st.load_real_points()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # ---- (a) 非对称度随 δ 变化 ----
    deltas = np.array([0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0])
    for name, color, marker in [("P1_dx", "#1f77b4", "o"), ("P1_ry", "#ff7f0e", "s"),
                                 ("P2_dx", "#2ca02c", "^")]:
        asyms = []
        for dv in deltas:
            fp, fn, rp, rn, TA_p = st.perturb(name, dv, +1, st.T_A)
            HA_p = st.H_of(TA_p, fr, rr, fp, fn, rp, rn)
            fp, fn, rp, rn, TA_m = st.perturb(name, dv, -1, st.T_A)
            HA_m = st.H_of(TA_m, fr, rr, fp, fn, rp, rn)
            a = (HA_p - HA_m) / (2 * dv) * 1000
            b = (HA_p + HA_m - 2 * HA) / (dv**2) * 1000
            asyms.append(abs(b * dv / (a + 1e-12)))
        ax1.plot(deltas, np.array(asyms) * 100, color=color, marker=marker, markersize=5,
                 linewidth=1.5, label={"P1_dx": r"$P_{1,\delta x}$", "P1_ry": r"$P_{1,\theta y}$",
                                       "P2_dx": r"$P_{2,\delta x}$"}[name])

    ax1.set_xlabel(r"Error magnitude $\delta$ (mm or $^\circ$)")
    ax1.set_ylabel(r"Asymmetry ratio $|b\delta/a|$ (%)")
    ax1.set_title("(a) Nonlinearity grows with error magnitude")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.legend()
    ax1.grid(alpha=0.3, which="both")
    ax1.axhline(10, color="gray", linestyle="--", linewidth=0.8)
    ax1.text(0.02, 11, "10%", fontsize=8, color="gray", transform=ax1.get_yaxis_transform())

    # ---- (b) 放大倍数随间距变化 ----
    spacings = np.array([50, 100, 200, 300, 500, 800, 1200])
    for name, color, marker in [("P1_dx", "#1f77b4", "o"), ("P1_ry", "#ff7f0e", "s")]:
        amplifications = []
        for sp in spacings:
            T_B_var = st.T_A + np.array([sp, 0, 0])
            dH_true_var = HA - st.H_of(T_B_var, fr, rr, st.P1_M, st.MIRROR_N, st.P2_M, st.MIRROR_N)

            fp, fn, rp, rn, TA_p = st.perturb(name, 1.0, +1, st.T_A)
            TB_p = st.perturb(name, 1.0, +1, T_B_var)[4]
            HA_p = st.H_of(TA_p, fr, rr, fp, fn, rp, rn)
            HB_p = st.H_of(TB_p, fr, rr, fp, fn, rp, rn)

            dH_err = abs(((HA_p - HB_p) - dH_true_var) * 1000)
            single = abs((HA_p - HA) * 1000)
            amplifications.append(dH_err / (single + 1e-12))

        ax2.plot(spacings, amplifications, color=color, marker=marker, markersize=5,
                 linewidth=1.5, label={"P1_dx": r"$P_{1,\delta x}$", "P1_ry": r"$P_{1,\theta y}$"}[name])

    # 理论线：D / |L| ≈ spacing / 80
    ax2.plot(spacings, spacings / 80, "k--", linewidth=1, alpha=0.6, label=r"$D/|L|$ theory")

    ax2.set_xlabel(r"Spacing $D$ (mm)")
    ax2.set_ylabel(r"Amplification factor $\Delta H_{\rm diff}/\Delta H_{\rm single}$")
    ax2.set_title("(b) Differential amplification grows with spacing")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle("Nonlinearity and spacing-dependent amplification", y=1.02, fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_asymmetry_amplification.pdf")
    plt.savefig(FIG_DIR / "fig3_asymmetry_amplification.png")
    plt.close()
    print("  fig3 saved")


if __name__ == "__main__":
    print("Generating figures...")
    fig1_sensitivity_comparison()
    fig2_lever_mechanism()
    fig3_asymmetry_and_amplification()
    print(f"All figures saved to {FIG_DIR}")
