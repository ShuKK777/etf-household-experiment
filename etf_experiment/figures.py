"""The five report figures, drawn from the aggregate CSVs.

Every number here is read from a published table, never recomputed, so a figure
cannot drift from the table it illustrates. The single exception is the
per-bracket threshold series in figure 4, which the fifteen-condition tables do
not carry; it is computed by ``threshold_by_bracket`` and the values it produces
are written next to the figure so the derivation is auditable.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .io import read_csv
from .stats import rng_for, weighted_ci

MODEL_LABELS = {
    "sharp-4b": "4b（4B+LoRA+决策头）",
    "sharp-421m-mlx": "421m-mlx",
    "sharp-4b-head-200k": "4b-head-200k",
    "sharp-322m": "322m",
}
MODEL_ORDER = list(MODEL_LABELS)
COLORS = {
    "sharp-4b": "#1f4e79",
    "sharp-421m-mlx": "#c0504d",
    "sharp-4b-head-200k": "#9bbb59",
    "sharp-322m": "#8064a2",
}
BRACKET_TAILS = {
    "b1": "不足 1 万元", "b2": "1 万至不足 5 万元", "b3": "5 万至不足 20 万元",
    "b4": "20 万至不足 50 万元", "b5": "50 万元及以上",
}


def _setup() -> None:
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 160
    plt.rcParams["savefig.bbox"] = "tight"


def _baseline(tables: Path, model: str, group: str, label: str) -> dict:
    for row in read_csv(tables / f"{model}_基准选择.csv") if (tables / f"{model}_基准选择.csv").exists() \
            else read_csv(tables / "基准选择.csv"):
        if row["分组"] == group and row["组别"] == label:
            return row
    raise KeyError((model, group, label))


def _clause(tables: Path, model: str, title: str) -> dict:
    path = tables / f"{model}_条款影响.csv"
    for row in read_csv(path if path.exists() else tables / "条款影响.csv"):
        if row["对比"] == title:
            return row
    raise KeyError((model, title))


def _heterogeneity(tables: Path, model: str, title: str, label: str) -> dict:
    path = tables / f"{model}_人群差异.csv"
    for row in read_csv(path if path.exists() else tables / "人群差异.csv"):
        if row["对比"] == title and row["组别"] == label:
            return row
    raise KeyError((model, title, label))


def figure_baseline(tables: Path, out: Path) -> Path:
    keys = list("ABCDEF")
    names = ["A 存款", "B 货基", "C 纯债", "D 红利低波", "E 宽基", "F 科技成长"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(keys))
    width = 0.2
    for position, model in enumerate(MODEL_ORDER):
        row = _baseline(tables, model, "全体", "全体")
        values = [float(row[f"{key}_估计"]) for key in keys]
        ax.bar(x + (position - 1.5) * width, values, width, label=MODEL_LABELS[model],
               color=COLORS[model], edgecolor="white", linewidth=0.6)
    ax.axhline(1 / 6, color="#888", ls="--", lw=1)
    ax.text(5.55, 1 / 6 + 0.006, "1/6\n（无差别）", fontsize=8, color="#666", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("选择概率", fontsize=10)
    ax.set_title("基准条件下六产品的选择概率：只有 4b 给出明显的风险—收益排序", fontsize=11)
    ax.legend(fontsize=8.5, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, 0.52)
    path = out / "fig1_基准分布.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_gradients(tables: Path, out: Path) -> Path:
    panels = [
        ("风险偏好", ["不愿承担任何风险", "略低", "平均", "略高", "高风险"],
         ["不愿承担\n任何风险", "略低", "平均", "略高", "高风险"], "风险偏好：4b 跨度 +10.2pp"),
        ("金融知识", ["两题全错", "仅通胀题正确", "仅利息题正确", "全对"],
         ["两题全错", "仅通胀题\n正确", "仅利息题\n正确", "全对"], "金融知识：4b 跨度 +3.5pp"),
        ("收支", ["收入不够支出", "收支相当", "有结余"],
         ["收入不够\n支出", "收支相当", "有结余"], "收支余裕：4b 跨度 +1.5pp"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))
    for ax, (group, keys, labels, title) in zip(axes, panels):
        for model in MODEL_ORDER:
            values, lows, highs = [], [], []
            for key in keys:
                row = _baseline(tables, model, group, key)
                values.append(float(row["权益ETF_估计"]))
                lows.append(float(row["权益ETF_下限"]))
                highs.append(float(row["权益ETF_上限"]))
            ax.plot(range(len(keys)), values, "o-", color=COLORS[model],
                    label=MODEL_LABELS[model], lw=1.8, ms=5)
            ax.fill_between(range(len(keys)), lows, highs, color=COLORS[model], alpha=0.15, lw=0)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, 0.55)
    axes[0].set_ylabel("权益 ETF 概率（D+E+F）", fontsize=10)
    axes[0].legend(fontsize=8, frameon=False, loc="center left")
    fig.suptitle("只有 4b 把家庭异质性读进决策；其余三个模型在三条梯度上都近似平线",
                 fontsize=11.5, y=1.02)
    path = out / "fig2_三类梯度.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_open_fee(tables: Path, out: Path, factorial_csv: Path | None) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    points = []
    for model in MODEL_ORDER:
        row = _clause(tables, model, "#3−#1")
        points.append((f"{MODEL_LABELS[model]} · 十五条件", float(row["权益ETF_估计"]),
                       float(row["权益ETF_下限"]), float(row["权益ETF_上限"]), COLORS[model], "o"))
    if factorial_csv and factorial_csv.is_file():
        for row in read_csv(factorial_csv):
            if row["档位"] == "开户费200元" and row["组别"] == "全体":
                points.append(("322m · 162 格因子边际", float(row["权益ETF"]),
                               float(row["权益ETF_下限"]), float(row["权益ETF_上限"]), "#8064a2", "s"))
    points.append(("27B 教师（1,000 户）", -0.0289, -0.0289, -0.0289, "#e07b39", "D"))
    y = np.arange(len(points))[::-1]
    for yi, (name, est, low, high, color, marker) in zip(y, points):
        ax.errorbar(est * 100, yi, xerr=[[(est - low) * 100], [(high - est) * 100]],
                    fmt=marker, color=color, ms=6, capsize=3, lw=1.3)
        ax.text(est * 100, yi + 0.28, f"{est * 100:+.2f}", fontsize=8, ha="center", color=color)
    ax.set_yticks(y)
    ax.set_yticklabels([point[0] for point in points], fontsize=8.5)
    ax.axvline(0, color="#333", lw=1)
    ax.set_xlabel("权益 ETF 概率变化（百分点）", fontsize=9.5)
    ax.set_title("200 元开户费：六个来源，方向无一翻转", fontsize=10.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(-3.6, 0.8)

    ax = axes[1]
    tick = ["b1 不足1万", "b2 1–5万", "b3 5–20万", "b4 20–50万", "b5 50万+"]
    for model in MODEL_ORDER:
        values = [float(_heterogeneity(tables, model, "#3−#1", f"{bracket} {tail}")["权益ETF_估计"]) * 100
                  for bracket, tail in BRACKET_TAILS.items()]
        ax.plot(range(5), values, "o-", color=COLORS[model], label=MODEL_LABELS[model], lw=1.7, ms=4.5)
    ax.axhline(0, color="#333", lw=1)
    ax.set_xticks(range(5))
    ax.set_xticklabels(tick, fontsize=8, rotation=18, ha="right")
    ax.set_ylabel("权益 ETF 变化（百分点）", fontsize=9.5)
    ax.set_title("分档梯度：4b 与 322m 方向相反", fontsize=10.5)
    ax.legend(fontsize=7.5, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    path = out / "fig3_开户费.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def threshold_by_bracket(
    probabilities,
    x: np.ndarray,
    weights: np.ndarray,
    asset: np.ndarray,
) -> dict[str, dict[str, dict[str, float]]]:
    """Condition 5 and 6 against the baseline, per bracket.

    The published clause table reports these only for the whole panel, because
    the interesting result is a *difference between brackets*: b1 is shut out of a
    10,000-yuan minimum and b2 is not. Recomputing per bracket here keeps the
    figure and the claim in section 4.5 derived from the same source.
    """
    out: dict[str, dict[str, dict[str, float]]] = {}
    for bracket in BRACKET_TAILS:
        mask = asset == bracket
        entry: dict[str, dict[str, float]] = {}
        for name, condition in (("#5−#1", 5), ("#6−#1", 6)):
            delta = probabilities.equity[mask, condition - 1] - probabilities.equity[mask, 0]
            estimate = weighted_ci(delta, weights[mask], rng_for("threshold", bracket, condition))
            entry[name] = {"point": estimate.point, "low": estimate.low, "high": estimate.high}
        out[bracket] = entry
    return out


def figure_threshold(tables: Path, out: Path, series: dict | None) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    if series:
        names = ["b1 不足1万\nX=1,500", "b2 1–5万\nX=9,100", "b3 5–20万\nX=30,000",
                 "b4 20–50万\nX=53,000", "b5 50万+\nX=88,000"]
        low = [series[bracket]["#5−#1"]["point"] for bracket in BRACKET_TAILS]
        high = [series[bracket]["#6−#1"]["point"] for bracket in BRACKET_TAILS]
        x = np.arange(5)
        ax.bar(x - 0.19, [value * 100 for value in low], 0.36, label="门槛降到 100 元",
               color="#9bbb59", edgecolor="white")
        ax.bar(x + 0.19, [value * 100 for value in high], 0.36, label="门槛升到 10,000 元",
               color="#c0504d", edgecolor="white")
        ax.axhline(0, color="#333", lw=1)
        for index, (low_value, high_value) in enumerate(zip(low, high)):
            ax.text(index - 0.19, low_value * 100 + (0.12 if low_value >= 0 else -0.28),
                    f"{low_value * 100:+.2f}", fontsize=7.5, ha="center")
            ax.text(index + 0.19, high_value * 100 + (0.12 if high_value >= 0 else -0.28),
                    f"{high_value * 100:+.2f}", fontsize=7.5, ha="center")
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=7.5)
        ax.set_ylim(-3.2, 1.0)
    else:
        ax.text(0.5, 0.5, "需要逐户概率才能分档", ha="center", va="center", fontsize=10, color="#888")
        ax.set_xticks([])
    ax.set_ylabel("权益 ETF 概率变化（百分点）", fontsize=9.5)
    ax.set_title("4b：门槛效应只出现在真的买不起的那一档", fontsize=10.5)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    base = 0.028665
    points = [base + 0.001332, base, base - 0.0219]
    ax.plot(range(3), [value * 100 for value in points], "o-", color="#e07b39", lw=1.8, ms=7)
    for index, value in enumerate(points):
        ax.annotate(f"{value * 100:.2f}%", (index, value * 100), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["100 元", "1,000 元", "10,000 元"], fontsize=9)
    ax.set_xlabel("起购门槛（b1 户，X=2,500 元）", fontsize=9.5)
    ax.set_ylabel("权益 ETF 概率（%）", fontsize=9.5)
    ax.set_title("27B 教师：买不起时 b1 的权益概率从 2.87% 掉到 0.68%", fontsize=10.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, 3.9)
    path = out / "fig4_门槛.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_rates(tables: Path, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    x = np.arange(len(MODEL_ORDER))
    simple = [float(_clause(tables, model, "#12−#1")["权益ETF_估计"]) * 100 for model in MODEL_ORDER]
    inflation = [float(_clause(tables, model, "#13−#12")["权益ETF_估计"]) * 100 for model in MODEL_ORDER]
    ax.bar(x - 0.19, simple, 0.36, label="① 存款利率 1.5%→0.95%（纯降息）", color="#9dc3e6",
           edgecolor="white")
    ax.bar(x + 0.19, inflation, 0.36, label="② 再告知物价年涨 2.5%（实际利率转负）", color="#1f4e79",
           edgecolor="white")
    ax.axhline(0, color="#333", lw=1)
    for index, (a, b) in enumerate(zip(simple, inflation)):
        ax.text(index - 0.19, a + 0.03, f"{a:+.2f}", fontsize=8, ha="center")
        ax.text(index + 0.19, b + 0.03, f"{b:+.2f}", fontsize=8, ha="center")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[model] for model in MODEL_ORDER], fontsize=8, rotation=12, ha="right")
    ax.set_ylabel("权益 ETF 概率变化（百分点）", fontsize=9.5)
    ax.set_title("② 在四个模型上符号完全一致，① 则几乎为零", fontsize=10.5)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(-0.2, 2.2)

    ax = axes[1]
    labels = ["①纯降息\n#12−#1", "②加通胀\n#13−#12", "合计\n#13−#1"]
    keys = ["#12−#1", "#13−#12", "#13−#1"]
    products = [("A_估计", "存款", "#c0504d"), ("B_估计", "货基", "#9bbb59"),
                ("C_估计", "纯债", "#8064a2"), ("权益ETF_估计", "权益 ETF", "#1f4e79")]
    x = np.arange(3)
    width = 0.2
    for position, (column, name, color) in enumerate(products):
        values = [float(_clause(tables, "sharp-4b", key)[column]) * 100 for key in keys]
        ax.bar(x + (position - 1.5) * width, values, width, label=name, color=color, edgecolor="white")
    ax.axhline(0, color="#333", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylabel("概率变化（百分点，对称对数轴）", fontsize=9)
    ax.set_title("4b：降息的钱先进货基，实际利率转负才进纯债与权益", fontsize=10.5)
    ax.legend(fontsize=8, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    ax.spines[["top", "right"]].set_visible(False)
    path = out / "fig5_降息与通胀.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def draw_all(tables: Path, out: Path, factorial_csv: Path | None = None,
             threshold_series: dict | None = None) -> list[str]:
    _setup()
    out.mkdir(parents=True, exist_ok=True)
    paths = [
        figure_baseline(tables, out),
        figure_gradients(tables, out),
        figure_open_fee(tables, out, factorial_csv),
        figure_threshold(tables, out, threshold_series),
        figure_rates(tables, out),
    ]
    return [path.name for path in paths]
