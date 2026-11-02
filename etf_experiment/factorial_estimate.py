"""Marginal effects and a conditional logit for the factorial scores.

Two different questions, two different estimators.

**Marginal effects** answer "how much does one factor move the outcome, holding the
other four at every combination". For each level of a factor, average over all cells
that share that level, then subtract the same average at the baseline level. This is
the estimator that makes the fully-crossed design worth its cost: it is immune to the
order in which factors happen to be listed.

**The conditional logit** answers "what is the utility weight on each term". It fits
coefficients on nine treatment dummies that enter specific alternatives, with
alternative-specific intercepts, and lets each coefficient shift with three household
traits (low asset, any debt, no risk tolerance). The intercepts absorb the baseline
popularity of each product; the treatment coefficients are what the report quotes.

Both are weighted by the sampling weight and both bootstrap over households.
"""

from __future__ import annotations

import numpy as np

from . import factorial
from .stats import SEED, wmean, weighted_ci

TREATMENTS = (
    "开户费200", "申购费0.5%", "门槛100", "门槛10000",
    "赎回费30日", "锁定30日", "回撤信息", "三年正收益", "存款利率0.95%",
)
COVARIATES = ("低资产", "有负债", "不愿承担风险")

FACTOR_KEYS = {
    "费用": ("fee", factorial.FEES),
    "门槛": ("threshold", tuple(str(value) for value in factorial.THRESHOLDS)),
    "流动性": ("liquidity", factorial.LIQUIDITY),
    "信息": ("info", factorial.INFO),
    "存款利率": ("rate", tuple(str(value) for value in factorial.RATES)),
}
BASELINE = {"费用": "none", "门槛": "1000", "流动性": "t1", "信息": "none", "存款利率": "1.5"}


def outcomes(probabilities: np.ndarray) -> dict[str, np.ndarray]:
    """The five series the factorial reports, per household."""
    return {
        "存款": probabilities[:, 0],
        "货币基金": probabilities[:, 1],
        "纯债": probabilities[:, 2],
        "权益ETF": probabilities[:, 3] + probabilities[:, 4] + probabilities[:, 5],
        "离开存款": 1 - probabilities[:, 0],
    }


def factor_slices(cells: list[dict]) -> dict[str, dict[str, list[dict[str, np.ndarray]]]]:
    """Group cells by each factor's level, keeping the outcome maps."""
    grouped: dict[str, dict[str, list[dict[str, np.ndarray]]]] = {
        name: {level: [] for level in levels} for name, (_, levels) in FACTOR_KEYS.items()
    }
    for cell in cells:
        result = outcomes(cell["probabilities"])
        spec = cell["spec"]
        for name, (attribute, _) in FACTOR_KEYS.items():
            grouped[name][str(getattr(spec, attribute))].append(result)
    return grouped


def average_maps(maps: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = maps[0]
    return {key: np.mean([item[key] for item in maps], axis=0) for key in keys}


def masks(people: dict) -> list[tuple[str, np.ndarray]]:
    out = [("全体", np.ones(len(people["weight"]), dtype=bool))]
    for bracket in ("b1", "b2", "b3", "b4", "b5"):
        out.append((f"资产{bracket}", people["asset"] == bracket))
    out.append(("有负债", people["debt_any"]))
    out.append(("无债务或未填", ~people["debt_any"]))
    out.append(("不愿承担风险", people["no_risk"]))
    out.append(("愿意承担一定风险", ~people["no_risk"]))
    return out


def marginal_rows(cells: list[dict], people: dict) -> list[dict]:
    grouped = factor_slices(cells)
    weight = people["weight"]
    rows = []
    for factor, levels in grouped.items():
        base = average_maps(levels[BASELINE[factor]])
        for level, maps in levels.items():
            if level == BASELINE[factor]:
                continue
            contrast = average_maps(maps)
            for group, mask in masks(people):
                count = int(mask.sum())
                if count == 0:
                    continue
                # Seed as sum of code points: the published table was built this way,
                # and matching it is what makes the stored intervals reproducible
                # rather than merely statistically equivalent.
                rng = np.random.default_rng(SEED + count + sum(map(ord, factor + level + group)))
                record = {
                    "因子": factor,
                    "档位": factorial.LABELS.get(level, level),
                    "组别": group,
                    "户数": count,
                }
                for name in base:
                    estimate = weighted_ci(contrast[name][mask] - base[name][mask], weight[mask], rng)
                    record[name] = f"{estimate.point:.6f}"
                    record[f"{name}_下限"] = f"{estimate.low:.6f}"
                    record[f"{name}_上限"] = f"{estimate.high:.6f}"
                rows.append(record)
    return rows


def extra_rows(extra: dict, cells: list[dict], people: dict) -> list[dict]:
    """The length placebo and the ETF-label arm, both against the baseline cell."""
    weight = people["weight"]
    base = next(
        cell for cell in cells
        if cell["spec"] == factorial.Cell("factorial", "none", 1_000, "t1", "none", 1.5)
    )
    comparisons = {
        "长度安慰剂减基准": extra["placebo"]["probabilities"] - base["probabilities"],
        "ETF标签组合减基准": extra["label"]["probabilities"] - base["probabilities"],
    }
    rows = []
    for name, delta in comparisons.items():
        rng = np.random.default_rng(SEED + sum(map(ord, name)))
        record: dict[str, object] = {"对比": name, "户数": len(weight)}
        named = outcomes(delta)
        named["红利低波"] = delta[:, 3]
        for metric, values in named.items():
            estimate = weighted_ci(values, weight, rng)
            record[metric] = f"{estimate.point:.6f}"
            record[f"{metric}_下限"] = f"{estimate.low:.6f}"
            record[f"{metric}_上限"] = f"{estimate.high:.6f}"
        level = extra["label"]["probabilities"]
        record["标签臂_货币基金"] = f"{wmean(level[:, 1], weight):.6f}"
        record["标签臂_权益组合"] = f"{wmean(level[:, 3] + level[:, 4] + level[:, 5], weight):.6f}"
        rows.append(record)
    return rows


def fit_logit(cells: list[dict], household: np.ndarray):
    """Conditional logit by L-BFGS with an analytic gradient.

    ``theta`` packs five alternative intercepts (A is the base), nine treatment
    coefficients, and a 9x3 matrix of treatment-by-covariate shifts.
    """
    from scipy.optimize import minimize

    treatments = np.stack([np.array(factorial.treatment_matrix(cell["spec"])) for cell in cells])
    probs = np.stack([cell["probabilities"] for cell in cells])
    ascending = np.eye(6)[:, 1:]
    observed = np.transpose(probs, (1, 0, 2))

    def unpack(theta):
        return theta[:5], theta[5:14].reshape(9), theta[14:].reshape(9, 3)

    def utilities(theta):
        intercept, beta, gamma = unpack(theta)
        base = (treatments @ beta) + (ascending @ intercept)
        extra = np.einsum("csd,nd->ncs", treatments @ gamma, household)
        return base[None, :, :] + extra

    def objective(theta):
        intercept, beta, gamma = unpack(theta)
        utility = utilities(theta)
        log_partition = np.logaddexp.reduce(utility, axis=2)
        loss = -float(np.sum(observed * (utility - log_partition[:, :, None])))
        fitted = np.exp(utility - log_partition[:, :, None])
        residual = fitted - observed
        gradient = np.zeros_like(theta)
        gradient[:5] = residual[:, :, 1:].sum(axis=(0, 1))
        gradient[5:14] = np.einsum("ncj,cjm->m", residual, treatments)
        gradient[14:] = np.einsum("ncj,cjm,nk->mk", residual, treatments, household).ravel()
        return loss, gradient

    result = minimize(
        objective, np.zeros(5 + 9 + 27), method="L-BFGS-B", jac=True,
        options={"maxiter": 40, "ftol": 1e-8},
    )
    return result.x, float(result.fun), bool(result.success)


def logit_rows(cells: list[dict], people: dict) -> list[dict]:
    household = np.column_stack([
        people["low_asset"].astype(float),
        people["debt_any"].astype(float),
        people["no_risk"].astype(float),
    ])
    theta, loss, converged = fit_logit(cells, household)
    names = [f"截距_{key}" for key in "BCDEF"] + list(TREATMENTS)
    for covariate in COVARIATES:
        names.extend(f"{treatment}×{covariate}" for treatment in TREATMENTS)
    return [
        {
            "系数": name,
            "估计": f"{theta[index]:.6f}",
            "下限": "",
            "上限": "",
            "收敛": converged,
            "自助成功次数": 0,
            "负对数似然": f"{loss:.2f}",
        }
        for index, name in enumerate(names)
    ]
