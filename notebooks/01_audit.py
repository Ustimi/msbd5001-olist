# -*- coding: utf-8 -*-
"""
01_audit — Olist 数据审计（A 负责）
目的：在决定标签、样本、特征之前，摸清 9 张原始表的数据质量。
产出：data/processed/audit_summary.csv（每表概况：shape/缺失/重复/键）
      data/processed/audit_findings.csv（专项发现，落盘供报告引用）
原始 CSV 只读；本脚本只写 data/processed。

数据来源：Kaggle Brazilian E-Commerce Public Dataset by Olist
下载日期：2026-10-09
"""
import pandas as pd
import numpy as np
from pathlib import Path

pd.set_option('display.width', 200)

RAW = Path('data/raw')
PROC = Path('data/processed')
PROC.mkdir(parents=True, exist_ok=True)

TABLES = {
    'orders': 'olist_orders_dataset.csv',
    'reviews': 'olist_order_reviews_dataset.csv',
    'items': 'olist_order_items_dataset.csv',
    'payments': 'olist_order_payments_dataset.csv',
    'products': 'olist_products_dataset.csv',
    'customers': 'olist_customers_dataset.csv',
    'sellers': 'olist_sellers_dataset.csv',
    'geolocation': 'olist_geolocation_dataset.csv',
    'translation': 'product_category_name_translation.csv',
}

findings = []  # 专项发现，最终写入 audit_findings.csv

def note(table, check, value, comment=''):
    findings.append({'table': table, 'check': check, 'value': str(value), 'note': comment})

# %% 1. 逐表读取 + 通用检查
summary = []
tables = {}
for name, fname in TABLES.items():
    df = pd.read_csv(RAW / fname)
    tables[name] = df
    n_rows, n_cols = df.shape
    n_dup = int(df.duplicated().sum())
    missing = df.isna().mean()
    max_missing = float(missing.max()) if len(missing) else 0.0
    missing_cols = missing[missing > 0].round(4).to_dict()
    summary.append({
        'table': name, 'file': fname, 'n_rows': n_rows, 'n_cols': n_cols,
        'n_dup_rows': n_dup, 'max_col_missing_pct': round(max_missing, 4),
        'cols_with_missing': missing_cols or '',
    })
    print(f'[load] {name:12s} rows={n_rows:>7d} cols={n_cols} dup={n_dup} '
          f'max_missing={max_missing:.2%} missing_cols={missing_cols or "无"}')

# %% 2. 键唯一性（B 做 join 前必须知道的结论）
# 注意：items 不进此表——order_item_id 是订单内序号（1..N），检查其全局唯一性没有意义，
# 该表以 order_id 为 join 键（预期一对多），底数见 items 专项。
key_checks = {
    'orders': ('order_id', None, ''),
    'reviews': ('review_id', None,
                '（重复 814 行；无影响：标签按订单内处理，review_id 仅作同刻同分 tiebreak）'),
    'payments': ('order_id', 'payment_sequential', ''),
    'products': ('product_id', None, ''),
    'customers': ('customer_id', None, ''),
    'sellers': ('seller_id', None, ''),
    'geolocation': ('geolocation_zip_code_prefix', None,
                    '（expected：一个邮编对应多行坐标，非问题）'),
    'translation': ('product_category_name', None, ''),
}
print('\n=== 键唯一性 ===')
for name, (key, key2, anno) in key_checks.items():
    df = tables[name]
    n_rows = len(df)
    if key2:
        n_uniq = df.drop_duplicates([key, key2]).shape[0]
        msg = f'{key}+{key2} 组合唯一' if n_uniq == n_rows else f'{key}+{key2} 组合不唯一！'
    else:
        n_uniq = int(df[key].nunique())
        msg = f'{key} 唯一' if n_uniq == n_rows else f'{key} 不唯一！'
    if anno:
        msg += ' ' + anno
    print(f'{name:12s} {msg}  ({n_uniq}/{n_rows})')

# %% 3. orders 专项：状态、缺失送达日期、时间顺序
print('\n=== orders 专项 ===')
o = tables['orders']
print('order_status 分布:')
print(o['order_status'].value_counts(dropna=False).to_string())
date_cols = ['order_purchase_timestamp', 'order_approved_at',
             'order_delivered_carrier_date', 'order_delivered_customer_date',
             'order_estimated_delivery_date']
print('\n时间字段缺失数:')
print(o[date_cols].isna().sum().to_string())
# 送达缺失但状态为 delivered 的订单数
mask_del = (o['order_status'] == 'delivered') & o['order_delivered_customer_date'].isna()
n_missing_deliv = int(mask_del.sum())
print(f'状态为 delivered 但缺送达日期: {n_missing_deliv}')
note('orders', '状态为 delivered 的订单数', int((o['order_status'] == 'delivered').sum()),
     '筛选漏斗第一级：其余订单为非 delivered 状态，全部排除')
note('orders', '状态为 delivered 但缺送达日期', n_missing_deliv,
     'Plan A 可行性的核心输入：这些订单将在筛选漏斗中排除')
for c in date_cols:
    note('orders', f'时间字段缺失数: {c}', int(o[c].isna().sum()), '')
# 时间顺序违规: 下单 > 批准
t = o.copy()
for c in date_cols:
    t[c] = pd.to_datetime(t[c], errors='coerce')
bad_purchase = (t['order_approved_at'] < t['order_purchase_timestamp']).sum()
bad_deliv = (t['order_delivered_customer_date'] < t['order_delivered_carrier_date']).sum()
print(f'批准早于下单（违规）: {int(bad_purchase)}')
print(f'送达顾客早于送达承运商（违规）: {int(bad_deliv)}')
note('orders', '批准早于下单（违规）', int(bad_purchase), '')
note('orders', '顾客送达早于承运商送达（违规）', int(bad_deliv),
     '仅记录：预测时点只用顾客送达时间，不影响使用')

# %% 4. reviews 专项：评分分布、多评价订单、跨表覆盖
print('\n=== reviews 专项 ===')
r = tables['reviews']
print('评分分布:')
print(r['review_score'].value_counts(dropna=False).sort_index().to_string())
multi = r.groupby('order_id')['review_id'].nunique()
print(f'有评价的订单数: {len(multi)} | 其中多条评价的订单: {int((multi > 1).sum())}')
o_ids = set(tables['orders']['order_id'])
r_ids = set(r['order_id'])
print(f'reviews 中出现但 orders 中不存在的 order_id: {len(r_ids - o_ids)}')
sc = r['review_score'].value_counts().sort_index()
note('reviews', '评分分布(1-5)', ' / '.join(map(str, sc.values)),
     f"对应评分 1-5 的订单数；低评(≤3)占比 {sc.loc[[1, 2, 3]].sum() / len(r):.2%}")
note('reviews', '多条评价的订单数', int((multi > 1).sum()), '标签规则需处理：按最早创建时间取首条')
note('reviews', '孤儿 order_id（不在 orders 表中）', len(r_ids - o_ids), '')
note('reviews', 'review_id 重复行数', len(r) - int(r['review_id'].nunique()),
     '不影响标签逻辑：按订单内处理，review_id 仅作同刻同分 tiebreak')

# %% 5. items 专项：每单 item 数分布（B 聚合前的底数）
print('\n=== items 专项 ===')
it = tables['items']
per_order = it.groupby('order_id').size()
print(f"items 覆盖订单数: {it['order_id'].nunique()} / {len(o_ids)}")
print(f"order_item_id 是订单内序号（1..N，全表仅 {it['order_item_id'].nunique()} 个不同值），不做全局唯一性检查")
print(f'每单 item 数: min={per_order.min()} 中位数={per_order.median():.0f} max={per_order.max()}')
neg = ((it['price'] < 0) | (it['freight_value'] < 0)).sum()
print(f'负 price 或 freight_value 行数: {int(neg)}')
prod_match = it['product_id'].isin(set(tables['products']['product_id'])).mean()
print(f"product_id 能匹配 products 表: {prod_match:.2%}")
note('items', '覆盖订单数', f"{it['order_id'].nunique()} / {len(o_ids)}",
     f"{len(o_ids) - it['order_id'].nunique()} 单无 items 记录：B 聚合后保留缺失标记，不删订单")
note('items', '每单 item 数 min/中位数/max',
     f"{per_order.min()} / {per_order.median():.0f} / {per_order.max()}", 'B 聚合前的底数')
note('items', '负 price 或 freight_value 行数', int(neg), '')
note('items', 'product_id 匹配 products 表比例', f'{prod_match:.2%}', '')

# %% 6. payments 专项
print('\n=== payments 专项 ===')
pm = tables['payments']
print(f"payments 覆盖订单数: {pm['order_id'].nunique()} / {len(o_ids)}")
print(f"payment_sequential 最大: {pm['payment_sequential'].max()} | 每单支付记录数中位数: {pm.groupby('order_id').size().median():.0f}")
print(f'负 payment_value 行数: {int((pm["payment_value"] < 0).sum())}')
print(f'payment_installments 为 0 的行数: {int((pm["payment_installments"] == 0).sum())}')
note('payments', '覆盖订单数', f"{pm['order_id'].nunique()} / {len(o_ids)}",
     '1 单无支付记录：B 合并时保留缺失标记，不删订单')
note('payments', 'payment_installments 为 0 的行数', int((pm["payment_installments"] == 0).sum()), '')

# %% 7. 其余维度表
print('\n=== products/customers/sellers/geolocation/translation ===')
p = tables['products']
print(f'products 缺 product_category_name 行数: {int(p["product_category_name"].isna().sum())}')
print(f"原始字段拼写注意: product_name_lenght / product_description_lenght（官方拼写错误，B 使用时要小心）")
num_cols = ['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']
neg_num = (p[num_cols] < 0).sum().sum()
zero_num = (p[num_cols] == 0).sum().sum()
print(f"product 数值字段为负的行数合计: {int(neg_num)} | 为 0 的行数合计: {int(zero_num)}")
note('products', '缺 product_category_name 行数', int(p['product_category_name'].isna().sum()), '')
note('products', '官方字段拼写错误', 'product_name_lenght / product_description_lenght',
     'B 使用列名时不要"纠正"拼写')
note('products', '数值字段为 0 的行数合计', int(zero_num), '')
c = tables['customers']
print(f"customers 行数 {len(c)} | 唯一 customer_id {c['customer_id'].nunique()} | 重复 customer_id 行数 {int(c.duplicated('customer_id').sum())}（B 连接前需注意）")
note('customers', '重复 customer_id 行数', int(c.duplicated('customer_id').sum()), '')
g = tables['geolocation']
print(f"geolocation 行数 {len(g)} | 完全重复行: {int(g.duplicated().sum())} | 唯一 (zip_prefix,lat,lng): {g.drop_duplicates(['geolocation_zip_code_prefix','geolocation_lat','geolocation_lng']).shape[0]}（核心范围不用此表）")
note('geolocation', '完全重复行数', int(g.duplicated().sum()), '核心范围不用此表，保持可选')
tr = tables['translation']
cov = tables['products']['product_category_name'].dropna().isin(set(tr['product_category_name'])).mean()
print(f"translation 唯一类别 {tr['product_category_name'].nunique()} | 能覆盖 products 类别: {cov:.2%}")
note('translation', '覆盖 products 类别比例', f'{cov:.2%}', '')

# %% 7.5 键与连接关系 —— B 的施工图纸（keys_for_join.csv）
join_specs = [
    ('orders', 'order_id', 'one_to_one', 'order_id', '基础表，直接使用'),
    ('reviews', 'order_id', 'one_to_many', None, 'A 已聚合为首评表 labels.parquet（一单一标签），直接用'),
    ('items', 'order_id', 'one_to_many', None, '必须先按 order_id 聚合再合并（每单 1~21 行）'),
    ('payments', 'order_id', 'one_to_many', None, '必须先按 order_id 聚合再合并（每单 1~29 条）'),
    ('products', 'product_id', 'many_to_one', 'product_id', 'items 经 product_id 安全连接'),
    ('customers', 'customer_id', 'many_to_one', 'customer_id', 'orders 经 customer_id 安全连接'),
    ('sellers', 'seller_id', 'many_to_one', 'seller_id', '经 items 关联到订单'),
]
join_rows = []
for table, jk, rel, uniq_col, action in join_specs:
    df = tables[table]
    join_rows.append({
        'table': table, 'join_key': jk,
        'unique': 'yes' if (uniq_col and df[uniq_col].is_unique) else 'no',
        'n_unique': int(df[uniq_col].nunique()) if uniq_col else int(df[jk].nunique()),
        'n_rows': len(df), 'relationship': rel, 'action_for_B': action,
    })
pd.DataFrame(join_rows).to_csv(PROC / 'keys_for_join.csv', index=False, encoding='utf-8-sig')

# %% 8. 保存审计汇总与专项发现
summary_df = pd.DataFrame(summary)
summary_df.to_csv(PROC / 'audit_summary.csv', index=False, encoding='utf-8-sig')
findings_df = pd.DataFrame(findings, columns=['table', 'check', 'value', 'note'])
findings_df.to_csv(PROC / 'audit_findings.csv', index=False, encoding='utf-8-sig')
print(f'\n[out] data/processed/audit_summary.csv（{len(summary_df)} 表）、'
      f'audit_findings.csv（{len(findings_df)} 条发现）、'
      f'keys_for_join.csv（{len(join_rows)} 张表）已保存')
