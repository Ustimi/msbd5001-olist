# -*- coding: utf-8 -*-
"""
01_split — 固定划分（A 负责，Step 4）
规则：按送达时间升序；同一送达时刻的订单整体放同一侧；按组起始累计位置划分
      start_cum < 70% -> train；< 85% -> val；其余 test。
检查：各组行数/低评率/日期范围；客户跨段重复（两种口径：
      customer_id=订单级 ID（一单一号，恒为 0，不识别真实客户）；
      customer_unique_id=真实客户，用于披露与 Limitations）。
产出：data/processed/split_ids.csv、data/processed/split_checks.csv
"""
import pandas as pd
from pathlib import Path

RAW = Path('data/raw')
PROC = Path('data/processed')
PROC.mkdir(parents=True, exist_ok=True)

labels = pd.read_parquet(PROC / 'labels.parquet')
orders = pd.read_csv(RAW / 'olist_orders_dataset.csv')
orders['order_delivered_customer_date'] = pd.to_datetime(
    orders['order_delivered_customer_date'], errors='coerce')

# 重建合格样本（与 01_labels_sample 同一筛选链，保证可复现）
delivered = orders[orders['order_status'] == 'delivered'].copy()
m = delivered.merge(labels, on='order_id', how='inner')
m = m[m['order_delivered_customer_date'].notna()]
m = m[m['review_creation_date'] > m['order_delivered_customer_date']]
assert len(m) == 87514, f'合格样本数不一致: {len(m)}'

# 挂真实客户 ID（跨段重复检查用，仅追踪，不进模型）
cust = pd.read_csv(RAW / 'olist_customers_dataset.csv')[['customer_id', 'customer_unique_id']]
m = m.merge(cust, on='customer_id', how='left', validate='many_to_one')
assert m['customer_unique_id'].notna().all(), 'customer_unique_id 存在缺失'

# 同一送达时刻整体同侧：按组起始累计位置划分
m = m.sort_values('order_delivered_customer_date', kind='stable')
g_size = m.groupby('order_delivered_customer_date').size()
g_start = g_size.cumsum() - g_size
n = len(m)

def assign(start_cum):
    if start_cum < 0.70 * n:
        return 'train'
    if start_cum < 0.85 * n:
        return 'val'
    return 'test'

m['split'] = m['order_delivered_customer_date'].map(g_start).map(assign)

# 检查 1：各组规模、低评率、日期范围（日期区间应严格递增且互不重叠）
per = m.groupby('split').agg(
    n_orders=('order_id', 'nunique'),
    low_rate=('low_rating', 'mean'),
    date_min=('order_delivered_customer_date', 'min'),
    date_max=('order_delivered_customer_date', 'max'),
)
# groupby 默认字母序（test/train/val），强制回 train/val/test 顺序，避免 tolist 张冠李戴
per = per.reindex(['train', 'val', 'test'])
per['share'] = per['n_orders'] / n
print('=== 各组概况 ===')
print(per.assign(
    low_rate=per['low_rate'].round(4), share=per['share'].round(4)).to_string())

# 检查 2a：订单级 ID（customer_id）跨段 —— 一单一号，恒为 0，仅作口径说明
cid_nsplit = m.groupby('customer_id')['split'].nunique()
cid_overlap = (cid_nsplit > 1).mean()
print(f'\n订单级 customer_id 跨段重复率（一单一号，预期恒为 0）: {cid_overlap:.1%}')

# 检查 2b：真实客户（customer_unique_id）跨段 —— 报告 Limitations 的依据
uid_nsplit = m.groupby('customer_unique_id')['split'].nunique()
uid_overlap = (uid_nsplit > 1).mean()
print(f'真实客户（customer_unique_id）跨段重复率: {uid_overlap:.1%}')
train_uids = set(m.loc[m['split'] == 'train', 'customer_unique_id'])
for s in ['val', 'test']:
    su = set(m.loc[m['split'] == s, 'customer_unique_id'])
    share = len(su & train_uids) / len(su)
    print(f'{s} 订单中，其真实客户也出现在 train 的比例: {share:.1%}')

# 输出 1：split_ids.csv（送达日期列仅用于切分审计，禁作特征）
out = m[['order_id', 'split', 'order_delivered_customer_date']]
assert out['order_id'].is_unique and len(out) == 87514
out.to_csv(PROC / 'split_ids.csv', index=False, encoding='utf-8-sig')
print(f'\n[out] data/processed/split_ids.csv 已保存（{len(out)} 行，order_id 唯一，互不重叠）')

# 输出 2：split_checks.csv —— Step 4 全部检查结果落盘
checks = [
    ('各组行数 train/val/test', ' / '.join(map(str, per['n_orders'].tolist())),
     '按组起始累计位置划分，占比如 share 行'),
    ('各组占比 train/val/test', ' / '.join(f'{x:.3%}' for x in per['share']),
     '目标 70/15/15，微调源于同刻同侧规则'),
    ('各组低评率 train/val/test', ' / '.join(f'{x:.4f}' for x in per['low_rate']),
     '随时间漂移：验证集所选阈值应用到测试集时需留意'),
    ('日期范围 train', f"{per.loc['train','date_min']} ~ {per.loc['train','date_max']}",
     '三组按秒级时刻严格递增、互不重叠；同日分属两侧属正常（不同时刻）'),
    ('日期范围 val', f"{per.loc['val','date_min']} ~ {per.loc['val','date_max']}", ''),
    ('日期范围 test', f"{per.loc['test','date_min']} ~ {per.loc['test','date_max']}", ''),
    ('customer_id 跨段重复率', f'{cid_overlap:.4f}',
     '订单级 ID 一单一号，恒为 0，不识别真实客户（口径说明，非泄漏结论）'),
    ('customer_unique_id 跨段重复率', f'{uid_overlap:.4f}',
     '真实客户口径：>0 时写进报告 Limitations（bootstrap 区间可能偏乐观）'),
    ('val 订单的真实客户出现在 train 的比例',
     f"{len(set(m.loc[m['split']=='val','customer_unique_id']) & train_uids) / m.loc[m['split']=='val','customer_unique_id'].nunique():.4f}", ''),
    ('test 订单的真实客户出现在 train 的比例',
     f"{len(set(m.loc[m['split']=='test','customer_unique_id']) & train_uids) / m.loc[m['split']=='test','customer_unique_id'].nunique():.4f}", ''),
]
checks_df = pd.DataFrame(checks, columns=['check', 'value', 'note'])
checks_df.to_csv(PROC / 'split_checks.csv', index=False, encoding='utf-8-sig')
print(f'[out] data/processed/split_checks.csv 已保存（{len(checks_df)} 条检查）')
