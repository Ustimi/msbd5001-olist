# scripts/make_experiment_config.py
from pathlib import Path
import json
import hashlib

# 脚本在 <project>/scripts/make_experiment_config.py
# parents[0] = scripts, parents[1] = 项目根 msbd5001-olist
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "experiment_config.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

config = {
    "seed": 42,
    "label": "low_rating",
    "split": "split",
    "id": "order_id",
    "feature_spec": "data/processed/02_features.json",
    "stages": ["E1", "E2", "E3"],
    "models": ["logreg", "rf"],
    "baseline_model": "dummy",

    "threshold_rule": "验证集 F1 最大；同分取较高阈值",
    "threshold_grid": [
        round(x, 2) for x in [i / 100 for i in range(5, 96, 5)]
    ],

    "tune_metric": "average_precision",
    "tune_on": "val",
    "final_eval_on": "test",

    "class_weight_options": [None, "balanced"],
    "use_smote": False,

    "logreg_params": {
        "C": [0.1, 1.0, 10.0],
        "max_iter": 2000,
        "solver": "lbfgs"
    },
    "rf_params": {
        "n_estimators": [300, 500],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 5]
    },

    "bootstrap_n": 1000,
    "bootstrap_unit": "test_orders_paired",

    "missing_indicator": True,
    "clip_quantile": None,
    "low_freq_category_threshold": None,

    "notes": [
        "阈值只在验证集选择，绝不在 test 上挑阈值。",
        "SMOTE 不使用。",
        "E1/E2/E3 使用同一批 order_id 与同一 split。",
        "purchase_month / purchase_hour / purchase_dayofweek 的归属：暂按 B 的 02_features.json 版本归入 E1，并累计进入 E2/E3。",
        "此项为待 A 和 C 组会确认的暂定决策；若组会改为归 E3，必须同步修改 02_features.json 并重跑所有实验。",
        "旧 metrics 保留旧 config_hash；新实验只使用最新 config_hash。"
    ]
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=2)
    f.write("\n")  # 末尾换行，保证文件字节稳定

file_hash = hashlib.md5(OUT.read_bytes()).hexdigest()
print("已写入:", OUT)
print("config_hash =", file_hash)