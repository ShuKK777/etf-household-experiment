#!/usr/bin/env python3
"""按第一步 ETF 方案抽取固定 10,000 户；只准备数据，不调用模型。"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ASSET = "extended.financial_asset_bracket"
RISK = "extended.risk_preference"
INVESTMENT = "extended.investment_activity"
QUOTAS = {"b1": 3000, "b2": 2600, "b3": 2400, "b4": 1200, "b5": 800}
M_VALUES = {"b1": 5000, "b2": 30000, "b3": 125000, "b4": 350000, "b5": 625000}
MAIN_SEED, FIRST_BATCH_SEED, ORDER_SEED = 20261021, 20261022, 20261023
EXPECTED_POOL_N = 9130025
EXPECTED_SOURCE_SHA = "231bc5651f8da612bfa9854ba73038438142c1c376ffe57a1bf3b9040f4be12b"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def largest_remainder(counts, quota):
    """精确整数运算；余数相同时按原编码表顺序。"""
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    numerators = counts * quota
    result = numerators // total
    remaining = int(quota - result.sum())
    order = sorted(range(len(counts)), key=lambda j: (-int(numerators[j] % total), j))
    result[order[:remaining]] += 1
    assert int(result.sum()) == quota
    assert np.all(result <= counts)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root, out = args.project.resolve(), args.output.resolve()
    source = root / "10M_Household/unique_10m.parquet"
    source_manifest = root / "10M_Household/manifest.json"
    brackets_path = root / "10M_Household/brackets.json"
    design = root / "etf实验/实验方案_20261021.md"
    out.mkdir(parents=True, exist_ok=True)
    # 禁止覆盖已经生成的名单；重现时另指定空目录。
    if any(p.name != "draw_sample.py" for p in out.iterdir()):
        raise RuntimeError("输出目录已有数据，请用 --output 指定新的空目录。")

    source_sha = sha256(source)
    assert source_sha == EXPECTED_SOURCE_SHA, "源文件与方案核对版本不同"
    metadata = json.loads(source_manifest.read_text())
    codebooks = metadata["codebooks"]
    pf = pq.ParquetFile(source)
    assert pf.metadata.num_rows == 10000000
    assert len(pf.schema_arrow.names) == 36
    assert all(t == pa.uint16() for t in pf.schema_arrow.types)
    assert codebooks[ASSET][:5] == list(QUOTAS)
    assert len(codebooks[RISK]) == 6
    print("源文件已核对；读取筛选与分组字段。", flush=True)

    frame = pq.read_table(source, columns=[INVESTMENT, ASSET, RISK])
    arrays = {c: frame[c].combine_chunks().to_numpy() for c in frame.column_names}
    eligible = arrays[INVESTMENT] == codebooks[INVESTMENT].index("none")
    pool_n = int(eligible.sum())
    assert pool_n == EXPECTED_POOL_N
    assert np.all(arrays[ASSET][eligible] < 5)
    assert np.all(arrays[RISK][eligible] < 6)

    rng = np.random.Generator(np.random.PCG64(MAIN_SEED))
    first_rng = np.random.Generator(np.random.PCG64(FIRST_BATCH_SEED))
    order_rng = np.random.Generator(np.random.PCG64(ORDER_SEED))
    records, strata = [], []
    for asset in codebooks[ASSET][:5]:
        asset_code = codebooks[ASSET].index(asset)
        in_asset = eligible & (arrays[ASSET] == asset_code)
        counts = np.bincount(arrays[RISK][in_asset], minlength=6)
        allocation = largest_remainder(counts, QUOTAS[asset])
        first_allocation = largest_remainder(counts, QUOTAS[asset] * 3 // 10)
        for risk_code, risk in enumerate(codebooks[RISK]):
            candidates = np.flatnonzero(in_asset & (arrays[RISK] == risk_code))
            N_h, n_h = int(counts[risk_code]), int(allocation[risk_code])
            n_first = int(first_allocation[risk_code])
            assert n_h > 0 and n_first <= n_h
            chosen = np.sort(rng.choice(candidates, size=n_h, replace=False))
            first_rows = set(map(int, first_rng.choice(chosen, size=n_first, replace=False)))
            inclusion = n_h / N_h
            weight = (N_h / pool_n) / (n_h / 10000)
            cell = {
                "stratum_id": asset + "|" + risk, "asset": asset, "risk": risk,
                "pool_n": N_h, "sample_n": n_h, "first_batch_n": n_first,
                "inclusion_probability": inclusion, "expansion_weight": 1 / inclusion,
                "normalized_weight": weight,
            }
            strata.append(cell)
            for row in chosen:
                row = int(row)
                x_cents = M_VALUES[asset] * 50
                # P=X/1.015，存款金额向上取到分，I 为余额。
                p_cents = (x_cents * 1000 + 1014) // 1015
                records.append({
                    "household_id": "sha256:" + source_sha + ":row:" + str(row),
                    "source_row_index": row,
                    "stratum_id": cell["stratum_id"],
                    "asset_bracket": asset, "risk_preference": risk,
                    "pool_stratum_n": N_h, "sample_stratum_n": n_h,
                    "inclusion_probability": inclusion,
                    "expansion_weight": 1 / inclusion, "normalized_weight": weight,
                    "assigned_asset_M_cny": M_VALUES[asset],
                    "assigned_idle_X_cents": x_cents,
                    "assigned_idle_X_cny": x_cents / 100,
                    "combo_deposit_P_cents": p_cents, "combo_etf_I_cents": x_cents - p_cents,
                    "batch": 1 if row in first_rows else 2,
                })
    records.sort(key=lambda r: r["source_row_index"])
    for i, record in enumerate(records):
        record["sample_index"] = i
        record["display_order"] = order_rng.permutation(list("ABCDEF")).tolist()
    selected = np.array([r["source_row_index"] for r in records], dtype=np.int64)
    assert len(selected) == len(np.unique(selected)) == 10000
    assert sum(r["batch"] == 1 for r in records) == 3000
    assert np.isclose(sum(r["normalized_weight"] for r in records), 10000)
    assert sum(s["pool_n"] for s in strata) == pool_n
    print("已抽取 10,000 个不同源行；读取完整 36 字段。", flush=True)

    pieces, offset = [], 0
    for batch in pf.iter_batches(batch_size=131072):
        end = offset + batch.num_rows
        lo, hi = np.searchsorted(selected, [offset, end])
        if hi > lo:
            positions = pa.array(selected[lo:hi] - offset)
            pieces.append(pa.Table.from_batches([batch]).take(positions))
        offset = end
    coded = pa.concat_tables(pieces)
    assert coded.num_rows == 10000 and coded.num_columns == 36
    assert coded.schema.equals(pf.schema_arrow, check_metadata=True)
    assert all(coded[c].null_count == 0 for c in coded.column_names)
    coded_arrays = {c: coded[c].combine_chunks().to_numpy() for c in coded.column_names}
    decoded_columns = {}
    for c in coded.column_names:
        assert int(coded_arrays[c].max()) < len(codebooks[c])
        decoded_columns[c] = pa.array([codebooks[c][int(v)] for v in coded_arrays[c]])
    decoded = pa.table(decoded_columns)
    assert set(decoded[INVESTMENT].to_pylist()) == {"none"}
    index = pa.Table.from_pylist(records)
    pq.write_table(coded, out / "households_coded.parquet", compression="zstd")
    pq.write_table(decoded, out / "households_decoded.parquet", compression="zstd")
    pq.write_table(index, out / "sample_index.parquet", compression="zstd")
    with (out / "households.jsonl").open("w") as f:
        for record, profile in zip(records, decoded.to_pylist()):
            f.write(json.dumps({**record, "profile": profile}, ensure_ascii=False) + "\n")
    save_json(out / "strata.json", strata)
    shutil.copyfile(source_manifest, out / "source_manifest.json")
    shutil.copyfile(brackets_path, out / "brackets.json")
    shutil.copyfile(design, out / "design_at_sampling.md")
    if out != Path(__file__).resolve().parent:
        shutil.copyfile(__file__, out / "draw_sample.py")

    # 重新打开产物，检查保存后的数据、行顺序和编码解码关系。
    assert pq.read_table(out / "households_coded.parquet").equals(coded)
    assert pq.read_table(out / "households_decoded.parquet").equals(decoded)
    assert pq.read_table(out / "sample_index.parquet").equals(index)
    jsonl_count = 0
    with (out / "households.jsonl").open() as f:
        for row, line in enumerate(f):
            obj = json.loads(line)
            assert obj["source_row_index"] == int(selected[row])
            assert len(obj["profile"]) == 36
            assert sorted(obj["display_order"]) == list("ABCDEF")
            jsonl_count += 1
    assert jsonl_count == 10000
    checks = {
        "passed": True, "source_sha256_matches": True,
        "pool_n": pool_n, "sample_n": 10000, "unique_source_rows": 10000,
        "original_fields_retained": 36, "all_investment_activity_none": True,
        "asset_quota": QUOTAS, "risk_strata": 30,
        "first_batch_n": 3000, "remaining_n": 7000,
        "weights_sum": sum(r["normalized_weight"] for r in records),
        "output_roundtrip_matches": True, "jsonl_records": jsonl_count,
        "model_calls_run": 0,
    }
    save_json(out / "validation.json", checks)

    risk_zh = dict(zip(codebooks[RISK], [
        "高风险高收益", "略高风险", "平均风险", "略低风险", "不愿承担任何风险", "未填写"
    ]))
    readme = [
        "# 第一步实验：固定 10,000 户样本", "",
        "已按方案抽取完成。全部 15 个条件共用这份名单，模型尚未运行。", "",
        "## 先看哪个文件", "",
        "- households.jsonl：每行一户，含完整 36 字段解码画像、源行号、权重、闲置金额、选项顺序和运行批次。",
        "- households_coded.parquet：源文件原样保留的 36 个 uint16 字段，共 10,000 行。",
        "- households_decoded.parquet：同样的 36 字段，数值编码已换成编码表中的标签。",
        "- sample_index.parquet：名单及实验赋值。三个 Parquet 文件的第 i 行都对应同一户。",
        "- strata.json：30 个资产与风险组合的池内人数、抽取人数和权重。",
        "- manifest.json、validation.json：来源、抽样方法、文件校验值和核对结果。",
        "- source_manifest.json、brackets.json：源编码表和金额区间说明。",
        "- draw_sample.py：本次实际使用的抽样脚本；design_at_sampling.md 保存抽样时的方案。", "",
        "## 抽样方式", "",
        "从当前无投资活动的 9,130,025 户中整行抽取。按资产五档分配人数，档内按风险偏好实际占比用最大余数法分配，再等概率、不放回抽取。",
        "同余数按原风险编码顺序决定。主抽样种子 20261021，随机算法 NumPy Generator(PCG64)。",
        "源行号从 0 开始，按原 Parquet 行序计数；输出按源行号升序排列。household_id 由源文件 SHA-256 和源行号组成。",
        "首批 3,000 户使用单独种子 20261022，在主样本各组内抽取；选项顺序使用单独种子 20261023，每户只生成一次。",
        "缺失值保留 unknown。M、X、组合资金 P/I 是方案给定的情景金额；原资产和收支区间保留在 profile 中。",
        "各户 normalized_weight 用于回到原无投资活动池的平均结果；expansion_weight 表示抽中一户对应的池内人数。",
        "households.jsonl 的 display_order 为展示位置从前到后对应的固定产品 ID，例如 C,A,F,B,D,E；15 个条件共用。", "",
        "| 资产档 | 户数 | 首批 | 闲置金额 X（元） |",
        "| --- | ---: | ---: | ---: |",
    ]
    for asset in QUOTAS:
        readme.append(f"| {asset} | {QUOTAS[asset]:,} | {QUOTAS[asset]*3//10:,} | {M_VALUES[asset]/2:,.0f} |")
    readme += [
        "| 合计 | 10,000 | 3,000 | — |", "",
        "## 前五户预览", "",
        "年龄和收支仍为原区间；以下金额 X 为实验赋值。", "",
        "| 样本序号（从 0 开始） | 源行号 | 年龄档 | 家庭人数 | 资产档 | 风险偏好 | X（元） |",
        "| ---: | ---: | --- | --- | --- | --- | ---: |",
    ]
    profiles = decoded.to_pylist()
    for record, profile in list(zip(records, profiles))[:5]:
        readme.append(
            f"| {record['sample_index']} | {record['source_row_index']} | "
            f"{profile['decision_maker.age_group']} | {profile['household.household_size']} | "
            f"{record['asset_bracket']} | {risk_zh[record['risk_preference']]} | "
            f"{record['assigned_idle_X_cny']:,.0f} |"
        )
    readme += [
        "", "## 重新生成", "",
        "使用同一源文件、同一方案及记录的 NumPy 版本，执行：", "",
        "python3 draw_sample.py --project <项目根目录> --output <另一个空目录>", "",
        "脚本拒绝覆盖已有名单。模型和 15 个条件的推理结果不包含在本目录中。", "",
    ]
    (out / "README.md").write_text("\n".join(readme))
    manifest = {
        "dataset": "ETF first-step fixed 10000-household panel",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"path": str(source), "sha256": source_sha, "rows": 10000000, "fields": 36},
        "source_manifest_sha256": sha256(source_manifest),
        "source_brackets_sha256": sha256(brackets_path),
        "design_snapshot_sha256": sha256(out / "design_at_sampling.md"),
        "filter": {INVESTMENT: "none"}, "pool_n": pool_n,
        "sample_n": 10000, "stratification": [ASSET, RISK],
        "asset_quota": QUOTAS,
        "asset_order": codebooks[ASSET][:5], "risk_order": codebooks[RISK],
        "allocation": "Within asset: integer largest remainder; ties use risk codebook order.",
        "selection": "Ascending eligible source rows per cell; Generator(PCG64).choice(replace=False); sorted output.",
        "seeds": {"main": MAIN_SEED, "first_batch": FIRST_BATCH_SEED, "display_order": ORDER_SEED},
        "first_batch": "3000 nested households; 30 percent per asset; risk allocation uses source-cell proportions.",
        "source_row_index_base": 0, "output_sort": "source_row_index ascending",
        "normalized_weight": "(pool_stratum_n / pool_n) / (sample_stratum_n / 10000)",
        "expansion_weight": "pool_stratum_n / sample_stratum_n",
        "assignments": {"M_cny": M_VALUES, "X": "M/2", "P_cents": "ceil(X_cents/1.015)", "I_cents": "X_cents-P_cents"},
        "planned_conditions": list(range(1, 16)), "planned_choices": 150000,
        "sample_drawn": True, "model_calls_run": 0,
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pyarrow": pa.__version__},
        "files": {
            p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
            for p in sorted(out.iterdir()) if p.is_file()
        },
    }
    save_json(out / "manifest.json", manifest)
    print(json.dumps({"output": str(out), **checks}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

