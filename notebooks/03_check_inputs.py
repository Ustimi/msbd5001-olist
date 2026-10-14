# notebooks/03_check_inputs.py
from pathlib import Path
import json
import re
import pandas as pd

ROOT = Path.cwd()          # 在仓库根目录运行
PROC = ROOT / "data/processed"

with open(PROC / "02_features.json", encoding="utf-8") as f:
    spec = json.load(f)

print("label =", spec["label"])
print("split =", spec["split"])
print("datasets =", spec["datasets"])

EXPECTED_N = 87514
EXPECTED_SPLIT = {"train": 61260, "val": 13127, "test": 13127}
STAGES = ["E1", "E2", "E3"]

def resolve_path(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


# ============================================================
# 1. 三表 order_id 集合一致、唯一、行数 87514
# ============================================================
order_sets = {}

for stage in STAGES:
    path = resolve_path(spec["datasets"][stage])
    ids = pd.read_parquet(path, columns=["order_id"])

    n = len(ids)
    nunique = ids["order_id"].nunique()
    is_unique = ids["order_id"].is_unique
    null_n = ids["order_id"].isna().sum()

    print(f"{stage}: {path.name} rows={n}, unique={nunique}, "
          f"is_unique={is_unique}, null={null_n}")

    assert n == EXPECTED_N, f"{stage} 行数 {n} != {EXPECTED_N}"
    assert is_unique, f"{stage} order_id 不唯一"
    assert null_n == 0, f"{stage} order_id 有空值"

    order_sets[stage] = set(ids["order_id"])

base_ids = order_sets["E1"]

for stage in ["E2", "E3"]:
    only_base = base_ids - order_sets[stage]
    only_stage = order_sets[stage] - base_ids

    print(f"{stage} vs E1: missing={len(only_base)}, extra={len(only_stage)}")

    assert not only_base and not only_stage, \
        f"{stage} 与 E1 order_id 集合不一致"


# ============================================================
# 2. split 不重新划分，且与 split_ids.csv 对齐
# ============================================================
split_ids = pd.read_csv(PROC / "split_ids.csv")
print("split_ids columns:", split_ids.columns.tolist())

assert {"order_id", "split"} <= set(split_ids.columns), \
    "split_ids.csv 缺少 order_id 或 split"

split_ids = split_ids[["order_id", "split"]].rename(
    columns={"split": "split_ref"}
)

assert split_ids["order_id"].is_unique, "split_ids.csv order_id 不唯一"
assert len(split_ids) == EXPECTED_N, \
    f"split_ids.csv 行数 {len(split_ids)} != {EXPECTED_N}"
assert set(split_ids["order_id"]) == base_ids, \
    "split_ids.csv 与 E1 order_id 集合不一致"

print("split_ids counts:", split_ids["split_ref"].value_counts().to_dict())

assert split_ids["split_ref"].value_counts().to_dict() == EXPECTED_SPLIT, \
    "split 计数不等于 61260 / 13127 / 13127"

for stage in STAGES:
    path = resolve_path(spec["datasets"][stage])
    df_split = pd.read_parquet(path, columns=["order_id", spec["split"]])

    if spec["split"] != "split":
        df_split = df_split.rename(columns={spec["split"]: "split"})

    merged = df_split.merge(
        split_ids,
        on="order_id",
        how="left",
        validate="one_to_one"
    )

    assert merged["split_ref"].notna().all(), \
        f"{stage} 有订单不在 split_ids.csv"

    mismatch = merged["split"] != merged["split_ref"]

    print(f"{stage} split 不一致数: {mismatch.sum()}")
    assert not mismatch.any(), \
        f"{stage} 的 split 与 split_ids.csv 不一致"

    print(f"{stage} split 计数:", merged["split"].value_counts().to_dict())


# ============================================================
# 3. 特征白名单与泄漏检查
# ============================================================
FORBIDDEN_EXACT = {
    "order_id", "customer_id", "customer_unique_id", "seller_id", "product_id",
    "split", "low_rating",
    "review_id", "review_score", "review_comment_title", "review_comment_message",
    "review_creation_date", "review_answer_timestamp",
    "order_purchase_timestamp", "order_approved_at",
    "order_delivered_carrier_date", "order_delivered_customer_date",
    "order_estimated_delivery_date",
    "customer_zip_code_prefix", "seller_zip_code_prefix",
    "geolocation_zip_code_prefix",
}

SUSPICIOUS_PATTERNS = [
    re.compile(r"(^|_)id($|_)"),      # order_id / customer_id / review_id 等
    re.compile(r"review"),            # 评价字段
    re.compile(r"timestamp"),         # 原始时间戳
    re.compile(r"zip|postal|prefix"), # 邮编
    re.compile(r"^low_rating$|^split$"),
]

for stage in STAGES:
    cols = spec[stage]

    print(f"\n===== {stage} 特征白名单 =====")
    print("特征数:", len(cols))
    print("特征:", cols)

    assert len(cols) == len(set(cols)), f"{stage} 特征重复"

    bad_exact = set(cols) & FORBIDDEN_EXACT
    assert not bad_exact, f"{stage} 含禁止列: {bad_exact}"

    suspicious = [
        c for c in cols
        if any(p.search(c) for p in SUSPICIOUS_PATTERNS)
    ]
    print("可疑列:", suspicious)
    assert not suspicious, f"{stage} 有可疑泄漏列: {suspicious}"

    path = resolve_path(spec["datasets"][stage])
    df = pd.read_parquet(path)

    missing = set(cols) - set(df.columns)
    assert not missing, f"{stage} 缺列: {missing}"

    X = df[cols]

    assert list(X.columns) == cols, \
        f"{stage} X 列顺序不等于白名单"

    assert spec["label"] not in X.columns, \
        f"{stage} X 含标签 {spec['label']}"
    assert spec["split"] not in X.columns, \
        f"{stage} X 含 split"
    assert "order_id" not in X.columns, \
        f"{stage} X 含 order_id"

    print("X shape:", X.shape)
    print("X 缺失:", X.isna().sum()[X.isna().sum() > 0].to_dict())


# ============================================================
# 4. E1 / E2 / E3 累计关系检查
# ============================================================
assert set(spec["E1"]) <= set(spec["E2"]) <= set(spec["E3"]), \
    "E1/E2/E3 不是累计特征组"

print("\nE2-E1:", sorted(set(spec["E2"]) - set(spec["E1"])))
print("E3-E2:", sorted(set(spec["E3"]) - set(spec["E2"])))

print("\n全部输入检查通过。")

# ============================================================
# 5. 输出检查结果
# ============================================================
check_results = []

def record_check(name, ok):
    check_results.append({"check": name, "ok": bool(ok)})
    if not ok:
        raise AssertionError(f"检查失败: {name}")

# 三表 order_id 集合一致
record_check("三表 order_id 集合一致", order_sets["E1"] == order_sets["E2"] == order_sets["E3"])

# split 与 split_ids.csv 一致
record_check("split 与 split_ids.csv 一致", not mismatch.any())

# 白名单无禁止列
record_check("E1/E2/E3 白名单无禁止列", not bad_exact and not suspicious)

# 累计特征组
record_check("E1 ⊆ E2 ⊆ E3", set(spec["E1"]) <= set(spec["E2"]) <= set(spec["E3"]))

# 最后统一落盘
pd.DataFrame(check_results).to_csv(PROC / "03_input_checks.csv", index=False)