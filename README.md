# MSBD5001 Group Project — Olist Low-Rating Prediction

研究问题：在固定模型、样本与评估协议的前提下，多表数据融合与特征工程能多大程度提升低评分（评分 ≤3）预测。完整方案见 `docs/Olist_Project_Proposal.docx`。

## 仓库结构

```
.
├── README.md                                  本文件：协作说明与交接索引
├── requirements.txt                            pip 直接依赖清单（主版本与 A 实测环境一致）
├── environment.yml                             conda 环境配方（python 3.11 + pip）
├── .gitignore                                  Python 临时文件忽略规则
│
├── data/
│   ├── raw/                                    原始数据（只读）
│   │   ├── olist_orders_dataset.csv            订单主表（99,441 行）
│   │   ├── olist_order_reviews_dataset.csv     评价表（99,224 行）
│   │   ├── olist_order_items_dataset.csv       订单商品明细（112,650 行）
│   │   ├── olist_order_payments_dataset.csv    支付记录（103,886 行）
│   │   ├── olist_products_dataset.csv          商品维度表（32,951 行）
│   │   ├── olist_customers_dataset.csv         客户表（99,441 行）
│   │   ├── olist_sellers_dataset.csv           卖家表（3,095 行）
│   │   ├── olist_geolocation_dataset.csv       地理坐标（1,000,163 行，可选，核心不使用）
│   │   ├── product_category_name_translation.csv  类别英文翻译（71 行）
│   │   └── checksums.md5                       原始数据 MD5 校验和
│   └── processed/                              处理产物（parquet 权威 + 同名 CSV 兜底）
│       ├── audit_summary.csv                   9 表审计概况
│       ├── audit_findings.csv                  25 条专项发现（质量坑点与处理口径）
│       ├── keys_for_join.csv                   表连接关系（B 的开工依据）
│       ├── labels.parquet / labels.csv         一单一标签（98,643 单）
│       ├── sample_flow.csv                     筛选漏斗逐步记录
│       ├── split_ids.csv                       按送达时间 70/15/15 划分
│       ├── split_checks.csv                    划分检查（含真实客户跨段重叠率）
│       ├── base_orders.parquet / base_orders.csv  建模主表（87,514 × 16）
│       └── eda_train_summary.csv               train/val/test EDA 汇总
│
├── notebooks/                                  数据管线脚本
│   ├── 01_audit.py                             数据审计
│   ├── 01_labels_sample.py                     标签构造 + 样本筛选
│   ├── 01_split.py                             固定划分
│   └── 01_base.py                              主表组装 + EDA
│
├── outputs/figures/                            EDA 图（5 张）
│   ├── fig_class_distribution.png              训练集类别分布
│   ├── fig_missingness.png                     特征缺失率
│   ├── fig_feature_histograms.png              连续特征直方图
│   ├── fig_split_timeline.png                  按月 × split 订单量
│   └── fig_lowrate_drift.png                   低评率时间漂移 + 划分边界
│
└── docs/
    ├── Olist_Project_Proposal.docx             Proposal 方案（英文）
    ├── Olist_Team_Execution_Plan.docx          组内执行计划与分工
    ├── 课程要求.txt                             MSBD5001 课程项目要求
    └── A_work_log.md                           A 的工作记录（报告/PPT 文本依据）
```

## 环境搭建

Python 3.11（conda）。虚拟环境与机器路径绑定，需各自重建，不可拷贝：

```bash
conda env create -f environment.yml
conda activate olist
pip install -r requirements.txt          # 安装缓慢可加清华镜像：-i https://pypi.tuna.tsinghua.edu.cn/simple
python -c "import pandas, sklearn, pyarrow; print(pandas.__version__, sklearn.__version__, pyarrow.__version__)"
```

依赖主版本预期：pandas 3.x / scikit-learn 1.9 / pyarrow 26.x。`requirements.txt` 只列直接依赖，间接依赖由 pip 自动解析。

## 数据

- 来源：[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)（Kaggle），下载日期 2026-10-09
- 数据已随仓库提交；若经共享盘分发，解压到 `data/raw/` 后校验：`md5sum -c data/raw/checksums.md5`
- 原始 CSV 只读，处理结果一律写入 `data/processed/`

## 数据管线与复现

四个脚本互相独立、均从 `data/raw` 重建并带断言，按序运行即可复现全部产物：

| 脚本 | 产出 |
|---|---|
| `notebooks/01_audit.py` | `audit_summary.csv`、`audit_findings.csv`、`keys_for_join.csv` |
| `notebooks/01_labels_sample.py` | `labels`（98,643 单）、`sample_flow.csv`（99,441 → 87,514） |
| `notebooks/01_split.py` | `split_ids.csv`、`split_checks.csv` |
| `notebooks/01_base.py` | `base_orders`（87,514 × 16）、`eda_train_summary.csv`、EDA 图 |

## B/C 交接指引

**B（融合与特征）**：开工前先读 `data/processed/keys_for_join.csv`（每张表的连接关系与聚合要求），再读 `audit_findings.csv` 中 table 为 items / payments / products 的条目（已知坑点）。输入为 `base_orders.parquet`；产出写入 `notebooks/02_*.py`，交付 `master_orders` 与 `features.json`。

**C（模型与评估）**：读 `audit_findings.csv` 中 table 为 reviews / orders 的条目（标签与样本口径）。输入为 `base_orders.parquet` + `split_ids.csv`。注意两点：`split_ids.csv` 的送达日期列仅用于切分审计，不进入模型特征；低评率随月份漂移（见 `split_checks.csv` 与 `fig_lowrate_drift.png`），验证集所选阈值应用到测试集时需留意。产出写入 `notebooks/03_*.py`。

## 其他

- 以 `order_id` 为交接索引；不删除订单、不重新随机划分
- `data/processed/` 下凡 parquet 必有同名 CSV（parquet 为权威工作格式，CSV 为通用副本）
