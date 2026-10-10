# -*- coding: utf-8 -*-
"""
01_labels_sample — 标签处理 + 样本筛选（A 负责）
Step 2 标签：删除完全重复评价 → 每单保留首次评价（时间平局按规则处理）→ 生成 low_rating。
Step 3 样本：逐步筛选（delivered → 有标签 → 时间有效 → 首评严格晚于送达），记录每步排除数。
产出：data/processed/labels.parquet、data/processed/sample_flow.csv
"""
import pandas as pd
from pathlib import Path

RAW = Path('data/raw')
PROC = Path('data/processed')
PROC.mkdir(parents=True, exist_ok=True)

# %% Step 2 标签处理
rev = pd.read_csv(RAW / 'olist_order_reviews_dataset.csv')
n0 = len(rev)
print(f'[标签] 原始评价行数: {n0}')

# 2.1 删除完全重复的评价行（审计发现为 0，规则仍然执行以可复现）
rev = rev.drop_duplicates(keep='first')
print(f'[标签] 删除完全重复行: {n0 - len(rev)}')

# 2.2 只接受 1–5 的有效评分
invalid = ~rev['review_score'].isin([1, 2, 3, 4, 5])
rev = rev[~invalid].copy()
print(f'[标签] 评分不在 1–5 范围: {int(invalid.sum())}')

# 2.3 解析评价创建时间
rev['review_creation_date'] = pd.to_datetime(rev['review_creation_date'], errors='coerce')
print(f'[标签] 评价创建时间无法解析: {int(rev["review_creation_date"].isna().sum())}')

# 2.4 每单保留首次评价；最早时刻评分冲突的订单整体排除
rev = rev.sort_values(['order_id', 'review_creation_date', 'review_id'])
n_rev_per_order = rev.groupby('order_id').size()
first = rev.groupby('order_id').head(1).copy()
# 检查最早时刻是否唯一（存在同一订单多条评价且时间并列）
tie = rev.groupby('order_id').apply(
    lambda g: g['review_creation_date'].nunique() < len(g), include_groups=False)
tie_orders = set(tie[tie].index)
# 时间平局的订单里，最早时刻评分是否冲突
conflict_orders = set()
for oid in tie_orders:
    g = rev[rev['order_id'] == oid]
    t0 = g['review_creation_date'].min()
    if g.loc[g['review_creation_date'] == t0, 'review_score'].nunique() > 1:
        conflict_orders.add(oid)
print(f'[标签] 最早时刻存在多条评价的订单: {len(tie_orders)}')
print(f'[标签] 最早时刻评分冲突（整体排除）: {len(conflict_orders)}')
first = first[~first['order_id'].isin(conflict_orders)].copy()

# 2.5 生成标签
first['low_rating'] = (first['review_score'] <= 3).astype(int)
first['n_reviews'] = first['order_id'].map(n_rev_per_order)
labels = first[['order_id', 'review_id', 'review_score',
                'review_creation_date', 'low_rating', 'n_reviews']]
low_rate = labels['low_rating'].mean()
print(f'[标签] 得到标签的订单数: {len(labels)} | 低评分占比: {low_rate:.3%}')
labels.to_parquet(PROC / 'labels.parquet', index=False)
labels.to_csv(PROC / 'labels.csv', index=False, encoding='utf-8-sig')

# %% Step 3 样本筛选（逐步记录排除数）
orders = pd.read_csv(RAW / 'olist_orders_dataset.csv')
flow = []  # (step, n_before, n_excluded, reason, n_after)
n = len(orders)
flow.append(('start', n, 0, '全量订单', n))

# 3.1 只保留已送达订单
is_delivered = orders['order_status'] == 'delivered'
n_ex = int((~is_delivered).sum())
n = int(is_delivered.sum())
flow.append(('status=delivered', n, n_ex, '非 delivered 状态', n))
delivered = orders[is_delivered]

# 3.2 有合格标签（首评无冲突）
labeled = delivered.merge(labels, on='order_id', how='left', indicator=True)
n_ex = int((labeled['_merge'] != 'both').sum())
n = int((labeled['_merge'] == 'both').sum())
flow.append(('has_label', n, n_ex, '无评价/评分无效/首评冲突', n))
labeled = labeled[labeled['_merge'] == 'both'].drop(columns='_merge')

# 3.3 送达时间有效
labeled['order_delivered_customer_date'] = pd.to_datetime(
    labeled['order_delivered_customer_date'], errors='coerce')
n_ex = int(labeled['order_delivered_customer_date'].isna().sum())
n = int(labeled['order_delivered_customer_date'].notna().sum())
flow.append(('valid_delivery_time', n, n_ex, '送达时间缺失/无法解析', n))
labeled = labeled[labeled['order_delivered_customer_date'].notna()]

# 3.4 首评严格晚于送达（预测时点要求）
after = labeled['review_creation_date'] > labeled['order_delivered_customer_date']
n_ex = int((~after).sum())
n = int(after.sum())
flow.append(('review_after_delivery', n, n_ex, '首评不晚于送达（含同时刻）', n))
sample = labeled[after]

final_low_rate = sample['low_rating'].mean()
print(f'[样本] 最终合格样本: {len(sample)} 单 | 低评分占比: {final_low_rate:.3%}')
print(f'[样本] 保留率（相对全量）: {len(sample)/len(orders):.1%}')

flow_df = pd.DataFrame(flow, columns=['step', 'n_after', 'n_excluded', 'reason', 'n_after_check'])
flow_df.to_csv(PROC / 'sample_flow.csv', index=False, encoding='utf-8-sig')
print(f'\n[out] data/processed/labels.parquet、sample_flow.csv 已保存')
print(flow_df.to_string(index=False))
