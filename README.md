# Olist 低评分预测 · 组内协作说明

> 面向 B、C 组员。回答三件事：**环境怎么配、数据怎么拿、文件怎么看**。Proposal 与期末报告承担对外内容，本文件只做内部协作用。

## 1. 环境重建（不要拷贝环境文件夹）

虚拟环境与机器路径绑定，直接拷贝无法使用。请在各自机器上执行：

```bash
conda env create -f environment.yml   # ① 创建 olist 环境（python 3.11）
conda activate olist                  # ② 激活
pip install -r requirements.txt       # ③ 安装直接依赖（间接依赖由 pip 自动解析）
python -c "import pandas, sklearn, pyarrow; print(pandas.__version__, sklearn.__version__, pyarrow.__version__)"
```

预期输出的主版本应与 A 的一致（pandas 3.x / scikit-learn 1.9 / pyarrow 26.x）。安装缓慢可加清华镜像：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 2. 原始数据获取与校验

原始数据通过共享盘分发（Kaggle 需登录，A 已下载并分发）。解压到 `data/raw/` 后**必须校验 MD5**：

```bash
md5sum -c data/raw/checksums.md5   # Linux / Git Bash
```

Windows PowerShell 可用 `Get-FileHash` 与 `checksums.md5` 中的值逐个比对。

- 来源：https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce（下载日期 2026-10-09）
- 9 个 CSV：orders 99,441 / reviews 99,224 / items 112,650 / payments 103,886 / products 32,951 / customers 99,441 / sellers 3,095 / geolocation 1,000,163 / translation 71（行）
- **原始 CSV 只读**，任何处理结果写入 `data/processed/`

## 3. 目录约定

```
data/raw/        原始 CSV（只读）
data/processed/  清洗后主表、划分、审计文件（交接用）
notebooks/       01 A 数据准备 / 02 B 融合特征 / 03 C 建模
src/             可重复运行的脚本
outputs/         metrics、figures
docs/            proposal、报告、A 的工作记录
```

## 4. 交接规则

- 以 `order_id` 为交接索引；B 不得悄悄删除订单，C 不得重新随机划分
- 任何样本改动需全组同意，并更新数据版本、划分与全部实验，不把不同版本结果放同一张表
- 格式组规：`data/processed/` 下凡 parquet 必有同名 CSV——parquet 为**权威工作格式**（pipeline 用它，类型不丢、读写快），CSV 为**通用副本**（应急/抽查/Excel 快看），两者由同一脚本生成，天然同步
- A 的交付：`audit_summary.csv`、`audit_findings.csv`、`keys_for_join.csv`、`sample_flow.csv`、`base_orders.parquet/csv`、`split_ids.csv`、`split_checks.csv`

## 5. 给 B/C 的阅读指路（审计结论已全部落盘，无需运行任何脚本）

- **给 B**：先看 `data/processed/keys_for_join.csv`（每张表怎么连、要不要先聚合），再看 `audit_findings.csv` 中 table 为 items / payments / products 的行（坑在哪）
- **给 C**：看 `audit_findings.csv` 中 table 为 reviews / orders 的行（标签与样本怎么定的）；主表与划分直接读 `base_orders.parquet` + `split_ids.csv`
- 注意：`split_ids.csv` 的 `order_delivered_customer_date` 列**仅用于切分审计**（验证时间顺序），禁作特征；C 建模以 `base_orders.parquet` + `split` 列为准

## 6. 复现 A 的产出

四个脚本互相独立、都从 `data/raw` 重建并带断言，按序运行即可复现全部产物：

```bash
python notebooks/01_audit.py           # → audit_summary / audit_findings / keys_for_join
python notebooks/01_labels_sample.py   # → labels（双格式）、sample_flow
python notebooks/01_split.py           # → split_ids、split_checks
python notebooks/01_base.py            # → base_orders（双格式）、EDA 图
```
