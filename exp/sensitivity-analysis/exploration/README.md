# 探索中间过程

这里是灵敏度分析在对话中逐步探索的完整过程记录，按时间顺序保留。

## 演进路线

```
mc_sensitivity.py    → 最初端到端三类因素（光斑/反射镜/探测杆），仅单次测量，6 镜面自由度
       ↓
mc_differential.py   → 引入差值测量，发现靶点抵消、镜面放大（固定间距 300mm）
       ↓
mc_asymmetry.py      → 专门分析正负不对称，一阶/二阶分解
       ↓
mc_full_dofs.py      → 补齐完整 13 自由度（加绕x、绕z旋转）
       ↓
（最终整理 → ../sensitivity_theory.py + ../sensitivity_figures.py）
```

## 各脚本说明

| 脚本 | 内容 | 独有价值 |
|------|------|---------|
| `mc_sensitivity.py` | 最初三类因素端到端分析 | **光斑像素偏移灵敏度**（前相机 10.6μm/px、后相机 3.3μm/px），最终版未纳入 |
| `mc_differential.py` | 差值测量引入 + 蒙特卡罗联合扰动 | 首次发现靶点抵消、镜面放大 |
| `mc_asymmetry.py` | 正负不对称分析 + 非对称度随 δ 变化 | 回答"为什么 +δ 和 −δ 不一样" |
| `mc_full_dofs.py` | 完整 13 自由度 | 补齐绕x、绕z旋转 |

## 与最终版的关系

最终版 `../sensitivity_theory.py` 从 `mc_full_dofs.py` 精炼而来，统一了函数定义。探索脚本里每个都重新定义了几何函数，有大量重复——这是迭代探索的正常状态，保留原样。

光斑像素灵敏度（`mc_sensitivity.py` 独有）尚未整合进最终版，如需要可从 `mc_spot.json` 取数据。
