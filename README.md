# Household ETF adoption: controlled product-condition experiments

A study of **what makes Chinese households move money out of deposits into
money-market funds, bond funds and equity ETFs**, and how much they would move
under each product design.

One fixed panel of **10,000 non-investing household personas**, stratified by
financial assets and risk preference, is put through fifteen product conditions.
For each condition a dedicated decision model scores all six candidate products
for every household, and the resulting probabilities are compared within
household, so household characteristics are held constant and only the clause
that changed can explain a difference.

The practical question is not "is 8% better than 1.5%". It is which frictions —
account-opening fees, minimum purchase amounts, lock-ups, drawdown disclosure,
framing — actually move money, for which households, and by how much.

---

## What this repository contains

| Path | Contents |
|---|---|
| `etf_experiment/` | the experiment code: conditions, persona rendering, task building, the engine launcher, the Qwen scorer, the three tables, the figures |
| `docs/调用链路.md` | **how the experiment calls the MAS engine**, and which modules are ours |
| `docs/重算一致性.md` | **can the published tables be rebuilt?** Byte-level comparison, and the two 1e-6 gaps explained |
| `docs/脚本对照.md` | **where every original script went**, and the one file deliberately left in the experiment package |
| `docs/设计文档/` | **how each of the six rounds is designed** — what changed, what it is compared against, what it identifies |
| `报告/实验汇总.md` | full report: design, results, economic interpretation, limits |
| `报告/实验汇总.pdf` | the same report as a print-ready PDF with figures |
| `报告/figures/` | the five figures, regenerable from the aggregate CSVs |
| `data/` | the frozen 10,000-household panel with weights and display orders |
| `results/` | aggregate tables for all four models, plus the placebo and two-tick results |
| `tests/` | regression tests, including a byte-level check of the condition templates |

Model checkpoints and the per-household raw scores are **not** in the repository;
see [Data and reproducibility](#data-and-reproducibility).

## Results in one page

Three findings survive cross-model replication (all four decision models plus a
larger 27B teacher model agree in sign):

| Finding | Numbers |
|---|---|
| **A one-off account-opening fee cuts equity participation, and pushes clients to the fee-free substitute** | 200 CNY → equity −1.42pp; of the 1.42pp leaving equity, money funds gain 0.62, bonds 1.15, and **deposits also lose 0.36** |
| **Minimum purchase amounts only bind households that genuinely cannot afford them, and blocking is far stronger than relaxing** | raising to 10,000 CNY: b1 equity −2.47pp, b2 (short by only 900 CNY) −0.01pp; lowering to 100 CNY recovers just +0.08pp |
| **A negative *real* rate changes asset class; a nominal rate cut only substitutes within class** | cut of 55bp: money funds +0.59pp, equity +0.17pp; adding 2.5% inflation: equity +1.01pp more, money funds turns −0.20pp |

Three findings are deliberately **not** claimed, because their sign flips across
models: the net effect of a proportional subscription fee, the effect of drawdown
disclosure, and which asset bracket a fixed fee hurts most.

Two robust **negative** results matter as much:

- A written 30-day lock-up or redemption fee does not reliably reduce equity
  allocation across five models (−0.33 to +1.83pp, signs disagree).
- Telling a household "your last order did not go through" *raises* the
  probability of the unaffordable product by 1.6–2.4pp relative to a
  length-matched neutral sentence. The constraint is not internalised.

Read `报告/实验汇总.md` for the full account, including the places where an
earlier analysis was wrong and why.

## How the engine is called

The experiment is built on the MAS engine's `persona_choice` path. One decision:

```
our task jsonl  →  run_batches  →  World (state, SQLite commit store)
                                →  Brain (policy=decision, memory off)
                                →  POST /v1/systemone
                                →  scorer: laya.serve  or  our Qwen head adapter
                                ←  six option probabilities
```

`docs/调用链路.md` has the full diagrams and a table of exactly which parts are
engine and which are ours. In short:

- **Engine, unmodified** (paths relative to the engine checkout): `run_batches`,
  `examples/persona_choice/app.py`, `mas_engine.service.server`,
  `src/mas_engine/brain/decision_model.py`, `laya.serve`.
- **Ours**: the fifteen condition templates, the 36-field persona rendering, the
  task-row construction, the launcher that stands up eight lanes, the Qwen
  decision-head adapter, the statistics, the tables and the figures.

## Install

```bash
python -m pip install -r requirements.txt
```

Python 3.10+. The only heavy dependency for the analysis path is NumPy; plots need
Matplotlib, the PDF needs WeasyPrint, and PDFs need CJK fonts (Noto Sans CJK).

Running the models additionally needs the MAS engine checkout and a CUDA build of
torch. Nothing is hardcoded: every location is read from the environment, and the
defaults are relative, so the common layout — the engine beside this checkout —
needs no configuration at all.

| Variable | Default | Meaning |
|---|---|---|
| `ETF_ROOT` | this checkout | where `data/`, `results/`, `tasks/` live |
| `MAS_ENGINE` | `$ETF_ROOT/../MAS_Engine_Microservice` | the engine checkout |
| `ETF_MODELS` | `$ETF_ROOT/models` | decision-model packages |
| `ETF_DATA` | `$ETF_ROOT/data` | the household panel |
| `ETF_INPUTS` | `$ETF_ROOT/inputs` | rendered prompt inputs |
| `ETF_TASKS` | `$ETF_ROOT/tasks` | engine task rows |
| `ETF_RESULTS` | `$ETF_ROOT/results` | per-household results |
| `ETF_LOGS`, `ETF_FIGURES` | `$ETF_ROOT/...` | run logs, report figures |
| `ETF_ENGINE_PYTHON` | `$MAS_ENGINE/.venv/bin/python` | engine interpreter |
| `ETF_GPU_PYTHON` | current interpreter | torch+CUDA, for `laya.serve` |
| `ETF_QWEN_PYTHON` | current interpreter | torch+peft, for the Qwen head |

The two scorer interpreters fall back to the running interpreter rather than to a
guessed path, so the analysis stages work on a machine with no GPU while a scoring
stage fails at import with a message that says which environment is missing.

## Use

```bash
# sanity check the panel (10,000 rows, weights sum to the count, orders permute A-F)
python -m etf_experiment.cli check

# render the panel and all fifteen conditions into engine task rows
python -m etf_experiment.cli tasks --model sharp-322m

# run one model through the engine on eight lanes
python -m etf_experiment.cli run --model sharp-322m --expected-files 150

# rebuild the three tables from archived per-household results
python -m etf_experiment.cli tables \
    --results results/sharp-322mNew \
    --out out/tables --schedule new

# redraw the figures
python -m etf_experiment.cli figures --tables out/tables

# the 162-cell factorial: score, then build its three tables
python -m etf_experiment.cli factorial --gpu 0
python -m etf_experiment.cli factorial-tables --cells results/factorial --out out/factorial

# the text probes and the teacher contrast
python -m etf_experiment.cli probes --gpu 0
python -m etf_experiment.cli probe-tables --probes results/probes --out out/probes
python -m etf_experiment.cli teacher-table \
    --teacher results/probes/teacher27b --cells results/factorial --out out/probes

# render the report to PDF
python -m etf_experiment.cli pdf

# tests
python -m pytest tests -q
```

## Design, in brief

**Panel.** From a synthetic population of 10 million household records built on
CFPS / CHFS / CGSS / CLDS, take the 9,130,025 with no investment activity and
stratify by financial-asset bracket x risk preference (30 strata). Draw 10,000
without replacement, seed `20261021`. Each household carries 36 fields.

**Idle cash.** X is the CFPS-2018 calibrated median discretionary balance by
bracket: **1,500 / 9,100 / 30,000 / 53,000 / 88,000 CNY**. The earlier schedule
(2,500 / 15,000 / 62,500 / 175,000 / 312,500) is kept because the first rounds ran
on it and the two are compared. The choice matters: on the new schedule a
10,000 CNY minimum shuts out both b1 and b2, which is what makes the threshold
result visible.

**Menu.** Deposit 1.5%, money fund 2.0%, bond fund 3.5% / −1.5% stress, and three
equity ETFs at 6/8/12% with −12/−25/−40% stress. A–F always denote the same
product; each household's display order is randomised once and fixed across all
conditions, and results are mapped back to A–F.

**Conditions.** Baseline plus fourteen single-clause deviations, each compared
against the baseline within household.

**Estimation.** Weighted mean of within-household differences, with a
household-clustered bootstrap (1,000 draws, seed `20261021`). The clustering
matters: a household contributes fifteen correlated conditions, so resampling
rows would badly understate the intervals.

**What the numbers are.** Model probabilities over options, not observed
purchases. They describe how the decision surface moves with each clause, which is
what product screening and test prioritisation need; they are not a forecast of
real client conversion.

## Data and reproducibility

Everything needed to reproduce the published tables is here:

- `data/` — the panel, weights, display orders, and sampling record.
- `results/` — aggregate CSVs per model: baseline, clause impact, heterogeneity;
  plus the placebo and two-tick tables.
- `报告/figures/` — regenerable from those CSVs.

What is here is the part that cannot be regenerated from code: **the design of every
round** (`docs/设计文档/`) and **the aggregate tables** (`results/`). The per-household
scores and the weights are deliberately excluded — together they are several hundred
megabytes, and both follow from the code plus the panel.

Not included, and why:

- **Model checkpoints.** Several GB each. The encoder and Qwen decision-head
  releases are on Hugging Face under `ya9919/` (`FinL-bilingual`, `FinL-mlx`,
  `FinQ`). Paths are configured through `ETF_MODELS`.
- **Per-household raw scores.** Tens of GB of jsonl. The aggregate tables plus the
  code that produced them are sufficient to audit every published number; the raw
  files can be regenerated by running `tasks` then `run`.

Known caveat, documented rather than hidden: one intermediate table produced
during the original work had conditions 12 and 13 identical row for row, because a
re-scoring pass wrote the same probabilities twice. It is not in this repository. On
a fresh machine, `etf-experiment tables` rebuilds from the per-household files and
refuses to run if any household-condition is missing, so the failure cannot recur
silently.

## Licence and citation

Code: MIT (see `LICENSE`). The household panel is derived from synthetic personas
built on public survey structures and is redistributed for research use.

If you use this work, please cite the report in `报告/实验汇总.md`.
