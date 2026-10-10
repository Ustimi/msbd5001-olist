# A 的工作记录（A_work_log）

> A 负责数据侧全部工作。本文件记录每一步的**任务、动作、产出、结论与评审修正**，
> 作为 Proposal、期末报告、PPT 的文本依据；引用具体数字时以 `data/processed/` 中落盘文件为准。
> 最后更新：2026-10-10

## 0. 时间线概览

| 日期 | 事项 |
|---|---|
| 10-09 | 下载数据（9 CSV，来源 Kaggle，MD5 已存档）；创建 olist 环境 |
| 10-10 | Step 1–5 全部完成；经多轮评审修正后交付闭环 |

## 1. 环境与共享资产

- 创建 `olist` conda 环境（python 3.11.17）：pandas 3.0.6 / numpy 2.4.6 / scikit-learn 1.9.1 / pyarrow 26.0.0 / matplotlib 3.11.2 / seaborn 0.13.2
- 交付：`requirements.txt`（9 行直接依赖 + 实测版本下限）、`environment.yml`、`data/raw/checksums.md5`、`README.md`
- 修正记录：初版 requirements 为 pip freeze 全量导出（114 行），含本地路径 `packaging @ file://...` 与 Windows 专属 `pywinpty` 两个"必炸"项 → 按评审改为直接依赖精简版

## 2. Step 1 数据审计

- 脚本：`notebooks/01_audit.py`（三轮迭代：初版 → 键检查修正 → 落盘完备性修正）
- 产出：`audit_summary.csv`（9 表体检总表）、`audit_findings.csv`（25 条专项发现）、`keys_for_join.csv`（7 表施工图纸）
- 关键结论：
  - 9 表行数与公开数据一致；无负金额；除 geolocation 外无完全重复行
  - items/payments 对 order_id 一对多（每单 1~21 行 / 1~29 条）→ 必须先聚合再合并
  - 775 单无 items、1 单无支付记录 → 保留缺失标记，不删订单
  - review_id 重复 814 行（无影响：按订单内处理，仅作同刻同分 tiebreak）
  - 官方字段拼写错误 `product_name_lenght`（用列名时勿"纠正"）
  - 8 单 delivered 缺送达日期（筛选漏斗第一级排除）
  - 评分分布 11424 / 3151 / 8179 / 19142 / 57328，低评(≤3)占 22.93%
- 修正记录：items 误用 order_item_id 做全局唯一性检查（它是订单内序号，无意义）→ 改为每单分布；时间缺失字典字符串 → 拆成一行一字段；print-only 结论 → 全部落盘

## 3. Step 2 标签处理 + Step 3 样本筛选

- 脚本：`notebooks/01_labels_sample.py`
- 产出：`labels.parquet/csv`（98,643 × 6，双格式）、`sample_flow.csv`
- 标签规则：删完全重复（0 行）→ 仅接受 1–5 分（0 行被删）→ 每单按 (order_id, 创建时间, review_id) 取首评 → 最早时刻评分冲突 30 单整体排除并计数 → `low_rating = (score ≤ 3)`
- 筛选漏斗（sample_flow.csv）：99,441 → 96,478（非 delivered 2,963）→ 95,802（无标签/冲突 676）→ 95,794（缺送达日期 8）→ **87,514**（首评不晚于送达 8,280）
- 保留率 88.0%（相对全量）；全标签低评率 22.9%，最终样本低评率 17.5%
- 修正记录：曾误报冲突数 22 vs 30 不一致，排查对账后确认无 bug（预检与脚本对"已送达"的定义差 8 个缺送达日期订单所致，两条路径自洽）

## 4. Step 4 固定划分

- 脚本：`notebooks/01_split.py`
- 产出：`split_ids.csv`（87,514 行）、`split_checks.csv`（10 条检查）
- 规则：按送达时间排序，组起始累计位置 70/15/15，同一送达时刻整体同侧
- 结果：train 61,260 / val 13,127 / test 13,127；日期区间按秒级严格递增、互不重叠
- 低评率漂移：18.22% → 16.42% → 15.24%（时间划分的正常现象，报告需披露；提示 C：验证集所选阈值应用到测试集时需留意）
- 真实客户跨段重叠（customer_unique_id 口径）：0.7% 客户跨多段；val/test 订单中客户出现在 train 的比例 2.3% / 1.6% → 写进 Limitations（bootstrap 区间可能偏乐观的依据）
- 修正记录：① 初版用 customer_id 查跨段得 0% 是恒真结论（一单一号，识别不了真实客户）→ 改用 customer_unique_id（96,096 个唯一值，2,997 个回头客）；② split_checks.csv 前三行值顺序张冠李戴（groupby 字母序 test/train/val）→ reindex 修复

## 5. Step 5 主表 + EDA

- 脚本：`notebooks/01_base.py`
- 产出：`base_orders.parquet/csv`（87,514 × 16，双格式）、`eda_train_summary.csv`、5 张图（outputs/figures/）
- 列构成：order_id/customer_id（追踪）、split、low_rating（y）、review 追踪 4 列、时间锚 3 列（追踪）、E1 候选特征 5 列（approved_wait_days、estimated_delivery_days、purchase_month/hour/dayofweek）
- EDA：训练集低评率 18.22%；测试集特征零缺失、低评率 15.24%；EDA 只用训练集，测试集仅做完整性检查
- 图：类别分布 / 特征缺失率 / 特征直方图 / 按月×split 时间线 / 低评率漂移+划分边界
- 修正记录：双格式组规（parquet 权威 + CSV 兜底）；timeline 图例顺序 reindex；新增漂移图回应"时间漂移怎么办"

## 6. 报告/PPT 引用速查

| 要写什么 | 去哪拿 |
|---|---|
| 数据集规模与质量 | `audit_summary.csv` + `audit_findings.csv` |
| 筛选漏斗数字 | `sample_flow.csv`（99,441 → 87,514） |
| 标签定义与低评率 | labels 22.9%；base_orders 训练集 18.2% |
| 划分方案与漂移 | `split_checks.csv`；`fig_lowrate_drift.png` |
| Limitations（客户重叠） | `split_checks.csv`（0.7% / 2.3% / 1.6%） |
| 给 B 的 join 依据 | `keys_for_join.csv` |
| 过程诚实叙事（列名错误、恒真检查返工、顺序 bug） | 本文件各节"修正记录" |

## 7. 待办

- [ ] Step 6：与 B 互审交接（B 检查键与筛选逻辑）；反查 C 的划分与样本一致性
- [ ] Step 7：Proposal 第 1–3 页（A 讲：问题、价值、数据与标签，共 3 分 15 秒）+ 统一文稿
- [ ] 组会对齐：低评率漂移的披露口径；E1/E3 特征边界（建议 weekday/月份/时段归 E3，由 features.json 定死）
