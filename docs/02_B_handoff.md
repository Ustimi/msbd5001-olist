# B → C 建模交接

特征规格为暂定版本：下单月份、小时、星期暂归 E1，并累计纳入 E2/E3，待成员 A 和 C 确认。见 02_features.json 的 pending_decisions。
预测时点为顾客实际送达时、首评前；实际配送特征不适用于下单时预测。

## 文件命名兼容性

B 文件统一使用 02_ 前缀。README、A 的脚本注释和工作记录仍使用旧交付名称；遵循协作边界，未修改这些文件。
C 或外部代码需要更新旧路径。旧名不保留副本或软链接；A 的 base_orders、labels、split_ids 等名称保持不变。
完整改名映射：

- `data/processed/items_agg.parquet` → `data/processed/02_items_agg.parquet`
- `data/processed/items_agg.csv` → `data/processed/02_items_agg.csv`
- `data/processed/payments_agg.parquet` → `data/processed/02_payments_agg.parquet`
- `data/processed/payments_agg.csv` → `data/processed/02_payments_agg.csv`
- `data/processed/customers_lookup.parquet` → `data/processed/02_customers_lookup.parquet`
- `data/processed/customers_lookup.csv` → `data/processed/02_customers_lookup.csv`
- `data/processed/master_orders.parquet` → `data/processed/02_master_orders.parquet`
- `data/processed/master_orders.csv` → `data/processed/02_master_orders.csv`
- `data/processed/master_orders_e3.parquet` → `data/processed/02_master_orders_e3.parquet`
- `data/processed/master_orders_e3.csv` → `data/processed/02_master_orders_e3.csv`
- `data/processed/join_audit.csv` → `data/processed/02_join_audit.csv`
- `data/processed/e3_feature_checks.csv` → `data/processed/02_e3_feature_checks.csv`
- `data/processed/features.json` → `data/processed/02_features.json`
- `data/processed/features_e3.json` → `data/processed/02_features_e3.json`
- `data/processed/feature_dictionary.csv` → `data/processed/02_feature_dictionary.csv`
- `data/processed/feature_spec_checks.csv` → `data/processed/02_feature_spec_checks.csv`
- `docs/B_handoff.md` → `docs/02_B_handoff.md`

## 输入与实验

E1 使用 data/processed/base_orders.parquet；E2 使用 02_master_orders.parquet；E3 使用 02_master_orders_e3.parquet。
parquet 为权威格式，同名 CSV 为副本。02_features.json 是显式白名单，不可把整表或“全部非标签列”直接作为 X。
三表均 87,514 单，按 order_id 对齐。y = low_rating（评分 ≤3）；split = split（train/val/test）。
固定划分为 61,260 / 13,127 / 13,127，不重新划分。统一模型与评估协议，比较累计特征组。

C 可按规格选择列，不要从数据表自动推断预测字段：

```python
import json
import pandas as pd

with open('data/processed/02_features.json', encoding='utf-8') as handle:
    spec = json.load(handle)
stage = 'E2'  # E1 / E2 / E3
df = pd.read_parquet(spec['datasets'][stage])
X = df[spec[stage]]
y = df[spec['label']]
train_mask = df[spec['split']].eq('train')
val_mask = df[spec['split']].eq('val')
test_mask = df[spec['split']].eq('test')
# C 的预处理 Pipeline 只在 X.loc[train_mask] 上 fit。
```

### E1（5 个，累计清单）

`order_approved_wait_days`, `estimated_delivery_days`, `purchase_month`, `purchase_hour`, `purchase_dayofweek`

### E2（18 个，累计清单）

`order_approved_wait_days`, `estimated_delivery_days`, `purchase_month`, `purchase_hour`, `purchase_dayofweek`, `order_item_count`, `seller_count`, `total_price`, `total_freight`, `avg_product_weight_g`, `payment_total`, `payment_count`, `max_installments`, `payment_type_count`, `has_credit_card`, `has_voucher`, `customer_state`, `customer_city`

### E3（25 个，累计清单）

`order_approved_wait_days`, `estimated_delivery_days`, `purchase_month`, `purchase_hour`, `purchase_dayofweek`, `order_item_count`, `seller_count`, `total_price`, `total_freight`, `avg_product_weight_g`, `payment_total`, `payment_count`, `max_installments`, `payment_type_count`, `has_credit_card`, `has_voucher`, `customer_state`, `customer_city`, `delivery_duration_days`, `delivery_delay_days`, `late_delivery`, `freight_ratio`, `invalid_price`, `multi_seller`, `same_state_share`

## 待团队确认

2026-10-14，成员 B 暂定将 purchase_month、purchase_hour、purchase_dayofweek 归入 E1，因三者仅由原始订单下单时间戳派生。
因此三者均已纳入 E1、E2、E3；这是待成员 A 和 C 确认的暂定团队决策，不代表 A/C 已批准。
A 的代码将三者列为 E1 候选，而 A_work_log 第 7 节曾建议归 E3，保留此差异供团队确认。
本次仅更新特征规格与交接材料，没有改变数据集或重新训练模型；若团队调整归属，应同步更新生成逻辑和交接规格。
invalid_price 当前全为 0，但按 E3 缺失规则保留在清单；所有常量列移除如需实施应采用一致的训练集规则。

## 类型、缺失与编码

customer_state/customer_city 是无序类别，建议训练集拟合 OneHotEncoder(handle_unknown='ignore')；城市较多，
若采用低频合并，阈值和词表只能从 train 获得。保留原始城市拼写；不要使用全样本目标编码。
purchase_month/hour/dayofweek 是日历类别，建议训练集拟合独热编码；如采用周期编码，应在统一建模协议中说明。
has_credit_card/has_voucher、late_delivery/multi_seller/invalid_price 是二元特征，直接映射为 0/1，无需当作连续量缩放。
其余数值类型见 02_feature_dictionary.csv。order_approved_wait_days 缺失 14 个，avg_product_weight_g 缺失 16 个，
其余正式特征当前无缺失。数值可采用训练集中位数插补，必要时增加训练流程中的缺失标记；类别可填固定 Missing 类别。
将所有插补、缩放、编码和数据驱动异常阈值置于只在 train 拟合的 Pipeline；val 用于调参和阈值选择，test 仅最终评估。
不要删除缺失订单；不要在整个数据集上 fit。

## 泄漏防护

ID、邮编、标签、split、原始时间戳和所有评价字段均不在白名单。review_creation_date 仅用于时间验证。
全样本已验证 purchase 和 actual delivery 严格早于首评。预计送达是原先承诺日期：84,801 单预计日期不早于首评，
不代表使用首评之后发生的实际事件。源数据是静态快照，商品/客户属性历史版本不可追溯，是预测时点可用性的限制。
没有按标签生成任何特征，没有训练统计编码；白名单与公式检查通过不等于消除所有数据源层面的历史可用性风险。

## 已知质量问题与限制

三张 lookup 对固定样本匹配率均 100%；缺失重量没有填零。
原始支付有 2 条零分期、3 条 not_defined，均保留；这些是原始表口径，不等于固定样本计数。
延迟保留负数，最小约 -146.016 天，源于预计日期较远；实际配送时长最大约 70.214 天。
2,738 单运费比值 >1，最大约 21.447（商品金额 0.85，运费 18.23），未截断。
各 E3 特征范围和分位数见 02_e3_feature_checks.csv；这些全样本质量报告不得用于测试集驱动的调参或特征选择。
低评率随时间漂移 18.22% → 16.42% → 15.24%；A 报告真实客户跨段重叠约 0.7%，应披露，
且 bootstrap 区间可能偏乐观。样本仅覆盖 delivered 且首评晚于送达的订单，存在适用人群限制。

## 复现

在仓库根目录、A 的 base_orders 已交付后依次执行：

```bash
conda run -n olist python notebooks/02_integration.py --stage items
conda run -n olist python notebooks/02_integration.py --stage payments
conda run -n olist python notebooks/02_integration.py --stage customers
conda run -n olist python notebooks/02_integration.py --stage master
conda run -n olist python notebooks/02_integration.py --stage e3
conda run -n olist python notebooks/02_integration.py --stage specs
```

已有 E2/E3 时只运行 specs，可生成交接文件和检查而不改数据。02_join_audit.csv、02_e3_feature_checks.csv、
02_feature_spec_checks.csv 为验证记录；02_features_e3.json 为 E3 公式来源，02_feature_dictionary.csv 汇总正式与待决特征。
