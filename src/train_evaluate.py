# src/train_evaluate.py
from pathlib import Path
import json
import hashlib
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline                               # type: ignore
from sklearn.compose import ColumnTransformer                       # type: ignore
from sklearn.impute import SimpleImputer                            # type: ignore
from sklearn.preprocessing import StandardScaler, OneHotEncoder     # type: ignore
from sklearn.dummy import DummyClassifier                           # type: ignore
from sklearn.linear_model import LogisticRegression                 # type: ignore
from sklearn.ensemble import RandomForestClassifier                 # type: ignore
from sklearn.metrics import average_precision_score                 # type: ignore    
from sklearn.metrics import f1_score                                # type: ignore  
from sklearn.metrics import precision_score                         # type: ignore 
from sklearn.metrics import recall_score                            # type: ignore 
from sklearn.metrics import roc_auc_score                           # type: ignore 



# ============================================================
# 0. 路径与配置
# ============================================================
# __file__ = <project>/src/train_evaluate.py
# parents[0] = src, parents[1] = 项目根
ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT / "outputs" / "experiment_config.json"
FEATURE_SPEC_PATH = ROOT / "data/processed/02_features.json"
SPLIT_IDS_PATH = ROOT / "data/processed/split_ids.csv"

# 读配置
with open(CONFIG_PATH, encoding="utf-8") as f:
    config = json.load(f)

config_hash = hashlib.md5(CONFIG_PATH.read_bytes()).hexdigest()

# 读特征规格
with open(FEATURE_SPEC_PATH, encoding="utf-8") as f:
    spec = json.load(f)

SEED = config["seed"]
LABEL = config["label"]
SPLIT = config["split"]
ID = config["id"]

EXPECTED_N = 87514
EXPECTED_SPLIT_COUNTS = {"train": 61260, "val": 13127, "test": 13127}


CATEGORICAL_COLS = {
    "customer_state",
    "customer_city",
    "purchase_month",
    "purchase_hour",
    "purchase_dayofweek",
}

BINARY_COLS = {
    "has_credit_card",
    "has_voucher",
    "late_delivery",
    "multi_seller",
    "invalid_price",
}

_ABLATION_DROP = {
    "E3_no_delivery": {
        "delivery_duration_days",
        "delivery_delay_days",
        "late_delivery",
    },
    "E3_no_payment": {
        "payment_total", "payment_count",
        "max_installments", "payment_type_count",
        "has_credit_card", "has_voucher",
    },
    "E3_no_product": {
        "order_item_count", "seller_count",
        "total_price", "total_freight",
        "avg_product_weight_g", "multi_seller",
    },
    "E3_no_geo": {
        "customer_state", "customer_city",
        "same_state_share",
    },
}

ABLATION_COLS = {
    name: [c for c in spec["E3"] if c not in drop]
    for name, drop in _ABLATION_DROP.items()
}

# ============================================================
# 1. 读一个 stage 的数据，并做核心断言
# ============================================================
def load_stage(stage: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    读取指定 stage 的数据，返回 (X, y, split)。
    同时执行：
      - order_id 唯一
      - order_id 集合与 split_ids.csv 完全一致
      - split 计数严格等于 61260 / 13127 / 13127
      - split 与 split_ids.csv 逐行一致
      - X 列严格等于 spec[stage] 白名单
    """
    assert stage in ("E1", "E2", "E3"), f"未知 stage: {stage}"

    # spec['datasets'][stage] 是相对路径，相对项目根解析
    data_path = ROOT / spec["datasets"][stage]
    assert data_path.exists(), f"找不到数据文件: {data_path}"

    df = pd.read_parquet(data_path)

    # --- 1.1 order_id 唯一且非空 ---
    assert ID in df.columns, f"{data_path.name} 缺 {ID} 列"
    assert df[ID].is_unique, f"{stage} 的 {ID} 不唯一"
    assert df[ID].notna().all(), f"{stage} 的 {ID} 有空值"

    # --- 1.2 行数检查 ---
    assert len(df) == EXPECTED_N, \
        f"{stage} 行数 {len(df)} != {EXPECTED_N}"

    # --- 1.3 与 split_ids.csv 对齐 ---
    split_ids = pd.read_csv(SPLIT_IDS_PATH)
    assert {ID, SPLIT} <= set(split_ids.columns), \
        f"split_ids.csv 缺 {ID} 或 {SPLIT}"

    split_ids = split_ids[[ID, SPLIT]].rename(columns={SPLIT: "split_ref"})
    assert split_ids[ID].is_unique, "split_ids.csv 的 order_id 不唯一"
    assert len(split_ids) == EXPECTED_N, \
        f"split_ids.csv 行数 {len(split_ids)} != {EXPECTED_N}"

    # 集合必须完全一致，用对称差检查，报错信息更清楚
    ids_data = set(df[ID])
    ids_ref = set(split_ids[ID])
    only_data = ids_data - ids_ref
    only_ref = ids_ref - ids_data
    assert not only_data and not only_ref, (
        f"{stage} 与 split_ids.csv 订单集合不一致: "
        f"仅数据有 {len(only_data)}, 仅参考有 {len(only_ref)}"
    )

    # --- 1.4 split 计数 ---
    split_counts = df[SPLIT].value_counts().to_dict()
    assert split_counts == EXPECTED_SPLIT_COUNTS, (
        f"{stage} split 计数 {split_counts} != {EXPECTED_SPLIT_COUNTS}"
    )

    # --- 1.5 逐行 split 与 split_ids.csv 一致 ---
    merged = df[[ID, SPLIT]].merge(
        split_ids, on=ID, how="left", validate="one_to_one"
    )
    assert merged["split_ref"].notna().all(), \
        f"{stage} 有订单在 split_ids.csv 中找不到"
    mismatch = (merged[SPLIT] != merged["split_ref"]).sum()
    assert mismatch == 0, \
        f"{stage} 与 split_ids.csv 有 {mismatch} 行 split 不一致"

    # --- 1.6 特征白名单 ---
    cols = spec[stage]
    assert isinstance(cols, list) and len(cols) > 0, \
        f"spec['{stage}'] 不是非空列表"

    missing_cols = set(cols) - set(df.columns)
    assert not missing_cols, f"{stage} 缺列: {missing_cols}"

    # 白名单里不能含标签、split、ID
    forbidden = {LABEL, SPLIT, ID}
    bad = forbidden & set(cols)
    assert not bad, f"{stage} 白名单含禁止列: {bad}"

    X = df[cols].copy()
    # 类别列统一转为 pandas string 类型
    # 否则 int32 的 purchase_month/hour/dayofweek 无法被 SimpleImputer 填 'Missing'
    cat_present = [c for c in cols if c in CATEGORICAL_COLS]
    for c in cat_present:
        X[c] = X[c].astype("string")

    y = df[LABEL].copy()
    split = df[SPLIT].copy()

    # --- 1.7 y 只能是 0/1，无缺失 ---
    assert set(y.dropna().unique()) <= {0, 1}, \
        f"{stage} 的 {LABEL} 取值不是 0/1"
    assert y.notna().all(), f"{stage} 的 {LABEL} 有缺失"

    return X, y, split

def load_ablation_stage(variant: str):
    assert variant in ABLATION_COLS, f"未知消融: {variant}"

    data_path = ROOT / spec["datasets"]["E3"]
    df = pd.read_parquet(data_path)

    assert df[ID].is_unique and df[ID].notna().all()
    assert len(df) == EXPECTED_N

    split_ids = pd.read_csv(SPLIT_IDS_PATH)
    split_ids = split_ids[[ID, SPLIT]].rename(columns={SPLIT: "split_ref"})
    assert split_ids[ID].is_unique
    assert set(df[ID]) == set(split_ids[ID])

    split_counts = df[SPLIT].value_counts().to_dict()
    assert split_counts == EXPECTED_SPLIT_COUNTS

    merged = df[[ID, SPLIT]].merge(
        split_ids, on=ID, how="left", validate="one_to_one"
    )
    assert merged["split_ref"].notna().all()
    assert (merged[SPLIT] == merged["split_ref"]).all()

    cols = ABLATION_COLS[variant]
    missing = set(cols) - set(df.columns)
    assert not missing, f"{variant} 缺列: {missing}"
    forbidden = {LABEL, SPLIT, ID}
    assert not (forbidden & set(cols))

    X = df[cols].copy()
    for c in cols:
        if c in CATEGORICAL_COLS:
            X[c] = X[c].astype("string")

    y = df[LABEL].copy()
    split = df[SPLIT].copy()
    assert set(y.dropna().unique()) <= {0, 1}

    return X, y, split

# ============================================================
# 2. 构建 train / val / test mask
# ============================================================
def build_masks(split: pd.Series) -> dict:
    masks = {
        "train": split.eq("train"),
        "val": split.eq("val"),
        "test": split.eq("test"),
    }
    for name, m in masks.items():
        assert m.sum() == EXPECTED_SPLIT_COUNTS[name], \
            f"{name} mask 数量 {m.sum()} != {EXPECTED_SPLIT_COUNTS[name]}"
    # 三个 mask 互斥
    assert (masks["train"] & masks["val"]).sum() == 0
    assert (masks["train"] & masks["test"]).sum() == 0
    assert (masks["val"] & masks["test"]).sum() == 0
    return masks

# ============================================================
# 3. 列分类：把特征分成数值 / 类别 / 二元
# ============================================================

def split_feature_columns(stage: str):
    if stage in ABLATION_COLS:
        cols = ABLATION_COLS[stage]
    else:
        cols = spec[stage]
    num_cols = [c for c in cols if c not in CATEGORICAL_COLS | BINARY_COLS]
    cat_cols = [c for c in cols if c in CATEGORICAL_COLS]
    bin_cols = [c for c in cols if c in BINARY_COLS]
    assert len(num_cols) + len(cat_cols) + len(bin_cols) == len(cols)
    return num_cols, cat_cols, bin_cols

# ============================================================
# 4. 预处理 Pipeline
# ============================================================

def make_preprocessor(stage: str, model_name: str) -> ColumnTransformer:
    num_cols, cat_cols, bin_cols = split_feature_columns(stage)

    # 数值管道：插补 + (LogReg 才加) 标准化
    num_steps = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ]
    if model_name == "logreg":
        num_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(num_steps)

    # 类别管道：填 Missing + 独热
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    transformers = [
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ]
    if bin_cols:
        transformers.append(("bin", "passthrough", bin_cols))

    return ColumnTransformer(transformers)

# ============================================================
# 5. 模型工厂
# ============================================================

def make_model(model_name, class_weight=None, params=None):
    params = params or {}
    if model_name == "dummy":
        return DummyClassifier(strategy="most_frequent", random_state=SEED)
    if model_name == "logreg":
        return LogisticRegression(
            C=params.get("C", 1.0),
            max_iter=config["logreg_params"]["max_iter"],
            solver=config["logreg_params"]["solver"],
            class_weight=class_weight,
            random_state=SEED,
        )
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 300),
            max_depth=params.get("max_depth", None),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            class_weight=class_weight,
            random_state=SEED,
            n_jobs=-1,
        )
    raise ValueError(f"未知模型: {model_name}")

def choose_threshold(y_val, p_val, grid) -> tuple[float, float]:
    best_t, best_f1 = None, -1.0
    for t in grid:
        f1 = f1_score(y_val, (p_val >= t).astype(int), pos_label=1)
        if f1 > best_f1:
            best_f1, best_t = f1, t
        elif f1 == best_f1 and (best_t is None or t > best_t):
            best_t = t
    assert best_t is not None
    return best_t, best_f1

def compute_metrics(y_true, y_prob, threshold) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "AP": average_precision_score(y_true, y_prob),
        "F1": f1_score(y_true, y_pred, pos_label=1),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, y_prob),
        "threshold": float(threshold),
    }

def paired_bootstrap(y_true, p_a, p_b, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    n_samples = len(y_true)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, n_samples, n_samples)
        ap_a = average_precision_score(y_true[idx], p_a[idx])
        ap_b = average_precision_score(y_true[idx], p_b[idx])
        diffs.append(ap_a - ap_b)
    diffs = np.array(diffs)
    return diffs.mean(), np.percentile(diffs, [2.5, 97.5])

def append_metrics(metrics_path: Path, record: dict) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame([record])

    if metrics_path.exists():
        df_old = pd.read_csv(metrics_path)
        if "config_hash" not in df_old.columns:
            df_old["config_hash"] = "legacy_no_hash"

        # 同一 config + 同一 experiment + 同一 model + 同一 notes 视为重跑，删旧留新
        key_cols = ["config_hash", "experiment", "model", "notes"]
        for c in key_cols:
            if c not in df_old.columns:
                df_old[c] = None
        mask = True
        for c in key_cols:
            mask = mask & (df_old[c].astype(str) == str(record.get(c)))
        df_old = df_old[~mask]

        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    cols = list(df_all.columns)
    if "config_hash" in cols:
        cols.insert(0, cols.pop(cols.index("config_hash")))
        df_all = df_all[cols]

    df_all.to_csv(metrics_path, index=False)

def run_experiment(stage, model_name, class_weight=None, params=None, tag=""):
    X, y, split = load_stage(stage)
    masks = build_masks(split)

    X_tr = X.loc[masks["train"]]
    X_va = X.loc[masks["val"]]
    X_te = X.loc[masks["test"]]
    y_tr = y.loc[masks["train"]]
    y_va = y.loc[masks["val"]]
    y_te = y.loc[masks["test"]]

    pre = make_preprocessor(stage, model_name)
    model = make_model(model_name, class_weight, params)

    X_tr_t = pre.fit_transform(X_tr)
    X_va_t = pre.transform(X_va)
    X_te_t = pre.transform(X_te)

    model.fit(X_tr_t, y_tr)
    p_va = model.predict_proba(X_va_t)[:, 1]
    p_te = model.predict_proba(X_te_t)[:, 1]

    if model_name == "dummy":
        threshold = 0.5
    else:
        threshold, _ = choose_threshold(y_va, p_va, config["threshold_grid"])

    metrics_va = compute_metrics(y_va, p_va, threshold)
    metrics_te = compute_metrics(y_te, p_te, threshold)

    record = {
        "experiment": stage,
        "model": model_name,
        "class_weight": class_weight,
        "params": str(params),
        "n_train": int(masks["train"].sum()),
        "n_val": int(masks["val"].sum()),
        "n_test": int(masks["test"].sum()),
        "AP": metrics_te["AP"],
        "F1": metrics_te["F1"],
        "precision": metrics_te["precision"],
        "recall": metrics_te["recall"],
        "ROC_AUC": metrics_te["ROC_AUC"],
        "threshold": metrics_te["threshold"],
        "val_AP": metrics_va["AP"],
        "val_F1": metrics_va["F1"],
        "seed": SEED,
        "config_hash": config_hash,
        "notes": tag,
    }
    append_metrics(ROOT / "outputs" / "metrics.csv", record)

    # 保存 test 预测概率，供 bootstrap 使用
    # 从主表按 split 列筛选，取真正的 hex order_id，不用 X 的位置索引
    test_df = pd.read_parquet(ROOT / spec["datasets"][stage]).reset_index(drop=True)
    test_order_ids = test_df.loc[test_df[SPLIT].eq("test"), "order_id"].values

    pred_path = ROOT / "outputs" / "test_predictions.csv"
    pred_row = pd.DataFrame({
        "order_id": test_order_ids,
        "y_true": y_te.values,
        "y_prob": p_te,
        "experiment": stage,
        "model": model_name,
        "tag": tag,
    })
    if pred_path.exists():
        pred_old = pd.read_csv(pred_path)
        pred_row = pd.concat([pred_old, pred_row], ignore_index=True)
    pred_row.to_csv(pred_path, index=False)

    print(f"[{stage}/{model_name}] "
          f"val AP={metrics_va['AP']:.4f} F1={metrics_va['F1']:.4f} t={threshold:.2f} | "
          f"test AP={metrics_te['AP']:.4f} F1={metrics_te['F1']:.4f}")

    return record, p_va, p_te, y_va, y_te

def run_ablation(variant, model_name, class_weight=None, params=None, tag=""):
    X, y, split = load_ablation_stage(variant)
    masks = build_masks(split)

    X_tr = X.loc[masks["train"]]
    X_va = X.loc[masks["val"]]
    X_te = X.loc[masks["test"]]
    y_tr = y.loc[masks["train"]]
    y_va = y.loc[masks["val"]]
    y_te = y.loc[masks["test"]]

    pre = make_preprocessor(variant, model_name)
    model = make_model(model_name, class_weight, params)

    X_tr_t = pre.fit_transform(X_tr)
    X_va_t = pre.transform(X_va)
    X_te_t = pre.transform(X_te)

    model.fit(X_tr_t, y_tr)
    p_va = model.predict_proba(X_va_t)[:, 1]
    p_te = model.predict_proba(X_te_t)[:, 1]

    threshold, _ = choose_threshold(y_va, p_va, config["threshold_grid"])
    metrics_va = compute_metrics(y_va, p_va, threshold)
    metrics_te = compute_metrics(y_te, p_te, threshold)

    record = {
        "experiment": variant,
        "model": model_name,
        "class_weight": class_weight,
        "params": str(params),
        "n_train": int(masks["train"].sum()),
        "n_val": int(masks["val"].sum()),
        "n_test": int(masks["test"].sum()),
        "AP": metrics_te["AP"],
        "F1": metrics_te["F1"],
        "precision": metrics_te["precision"],
        "recall": metrics_te["recall"],
        "ROC_AUC": metrics_te["ROC_AUC"],
        "threshold": metrics_te["threshold"],
        "val_AP": metrics_va["AP"],
        "val_F1": metrics_va["F1"],
        "seed": SEED,
        "config_hash": config_hash,
        "notes": tag,
    }
    append_metrics(ROOT / "outputs" / "metrics.csv", record)

    # 保存 test 预测，从 E3 主表按 split 列筛选取 hex order_id
    test_df = pd.read_parquet(ROOT / spec["datasets"]["E3"]).reset_index(drop=True)
    test_order_ids = test_df.loc[test_df[SPLIT].eq("test"), "order_id"].values

    pred_path = ROOT / "outputs" / "test_predictions.csv"
    pred_row = pd.DataFrame({
        "order_id": test_order_ids,
        "y_true": y_te.values,
        "y_prob": p_te,
        "experiment": variant,
        "model": model_name,
        "tag": tag,
    })
    if pred_path.exists():
        pred_old = pd.read_csv(pred_path)
        pred_row = pd.concat([pred_old, pred_row], ignore_index=True)
    pred_row.to_csv(pred_path, index=False)

    print(f"[{variant}/{model_name}] "
          f"val AP={metrics_va['AP']:.4f} F1={metrics_va['F1']:.4f} t={threshold:.2f} | "
          f"test AP={metrics_te['AP']:.4f} F1={metrics_te['F1']:.4f}")

    return record, p_va, p_te, y_va, y_te

def run_feasibility():
    print("===== E0 Dummy =====")
    run_experiment("E1", "dummy", tag="E0 baseline")

    print("===== E1 + LogReg =====")
    run_experiment("E1", "logreg", class_weight=None, tag="E1 feasibility")

    print("===== E1 + RF =====")
    run_experiment("E1", "rf", class_weight=None, tag="E1 feasibility")

    print("完成。metrics 见 outputs/metrics.csv")

def tune_params(stage, model_name, class_weight=None):
    """
    在验证集上用小网格选参数，指标用 average_precision。
    返回 (best_params, best_val_ap)。
    不写 metrics.csv，只负责选参。
    """
    X, y, split = load_stage(stage)
    masks = build_masks(split)

    X_tr = X.loc[masks["train"]]
    X_va = X.loc[masks["val"]]
    y_tr = y.loc[masks["train"]]
    y_va = y.loc[masks["val"]]

    pre = make_preprocessor(stage, model_name)
    X_tr_t = pre.fit_transform(X_tr)
    X_va_t = pre.transform(X_va)

    if model_name == "logreg":
        grid = [(C, cw)
                for C in config["logreg_params"]["C"]
                for cw in ([class_weight] if class_weight is not None
                           else config["class_weight_options"])]
    elif model_name == "rf":
        grid = [(n, d, l, cw)
                for n in config["rf_params"]["n_estimators"]
                for d in config["rf_params"]["max_depth"]
                for l in config["rf_params"]["min_samples_leaf"]
                for cw in ([class_weight] if class_weight is not None
                           else config["class_weight_options"])]
    else:
        raise ValueError(f"调参不支持模型: {model_name}")

    best_ap, best_params = -1.0, None
    for params in grid:
        if model_name == "logreg":
            C, cw = params
            m = LogisticRegression(
                C=C,
                max_iter=config["logreg_params"]["max_iter"],
                solver=config["logreg_params"]["solver"],
                class_weight=cw,
                random_state=SEED,
            )
        else:
            n, d, l, cw = params
            m = RandomForestClassifier(
                n_estimators=n, max_depth=d, min_samples_leaf=l,
                class_weight=cw, random_state=SEED, n_jobs=-1,
            )
        m.fit(X_tr_t, y_tr)
        p_va = m.predict_proba(X_va_t)[:, 1]
        ap = average_precision_score(y_va, p_va)
        if ap > best_ap:
            best_ap, best_params = ap, params

    print(f"[tune] {stage}/{model_name}: best val AP={best_ap:.4f}, params={best_params}")
    return best_params, best_ap

# ============================================================
# 6. 主流程：先跑通读取与断言
# ============================================================
def main():
    print("ROOT =", ROOT)
    print("config_hash =", config_hash)

    # 输入检查（你已有）
    for stage in ["E1", "E2", "E3"]:
        X, y, split = load_stage(stage)
        build_masks(split)
    print("输入检查通过。")

    for name, cols in ABLATION_COLS.items():
        assert cols, f"{name} 为空"
        assert len(cols) == len(set(cols)), f"{name} 有重复"
        assert set(cols) <= set(spec["E3"]), f"{name} 不是 E3 子集"
        assert set(spec["E3"]) - set(cols) == _ABLATION_DROP[name], \
            f"{name} 去掉的列与声明不符"

    print("消融清单:")
    for name, cols in ABLATION_COLS.items():
        print(f"  {name}: {len(cols)} 列")


if __name__ == "__main__":
    main()