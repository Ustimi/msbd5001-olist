# -*- coding: utf-8 -*-
"""
01_base — 基础订单表 + 训练集 EDA（A 负责，Step 5）
组装 base_orders（E1 候选特征 + 追踪列），EDA 只用训练集，测试集仅做完整性检查。
E1 候选特征（是否进 X、归哪个特征组由 features.json 决定）：
  order_approved_wait_days  批准等待时长 = approved_at - purchase（秒/86400）
  estimated_delivery_days   预计配送天数 = estimated_delivery - purchase（秒/86400）
  purchase_month            下单月份（1-12）
  purchase_hour             下单小时（0-23）
  purchase_dayofweek        下单星期（0=周一）
追踪列（绝不进 X）：order_id, customer_id, split, low_rating, review_id, review_score,
  review_creation_date, n_reviews, 三个时间戳。
产出：data/processed/base_orders.parquet、outputs/figures/*.png、data/processed/eda_train_summary.csv
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

RAW = Path('data/raw')
PROC = Path('data/processed')
OUT = Path('outputs/figures')
OUT.mkdir(parents=True, exist_ok=True)

# 图表样式（dataviz 参考色板，静态 PNG，无 hover 层）
plt.rcParams.update({
    'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb',
    'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': '#0b0b0b',
    'text.color': '#0b0b0b', 'xtick.color': '#898781', 'ytick.color': '#898781',
    'grid.color': '#e1e0d9', 'axes.grid': True, 'axes.axisbelow': True,
    'font.size': 10, 'axes.titlesize': 12, 'figure.dpi': 150,
})
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'

labels = pd.read_parquet(PROC / 'labels.parquet')
split = pd.read_csv(PROC / 'split_ids.csv')
orders = pd.read_csv(RAW / 'olist_orders_dataset.csv')

# 重建合格样本 + 挂 split（同一筛选链）
orders['order_delivered_customer_date'] = pd.to_datetime(
    orders['order_delivered_customer_date'], errors='coerce')
delivered = orders[orders['order_status'] == 'delivered'].copy()
m = delivered.merge(labels, on='order_id', how='inner')
m = m[m['order_delivered_customer_date'].notna()]
m = m[m['review_creation_date'] > m['order_delivered_customer_date']]
m = m.merge(split[['order_id', 'split']], on='order_id', how='inner', validate='one_to_one')
assert len(m) == 87514, f'合格样本数不一致: {len(m)}'

# 时间字段
for c in ['order_purchase_timestamp', 'order_approved_at', 'order_estimated_delivery_date']:
    m[c] = pd.to_datetime(m[c], errors='coerce')

# E1 候选特征
m['order_approved_wait_days'] = (
    m['order_approved_at'] - m['order_purchase_timestamp']).dt.total_seconds() / 86400
m['estimated_delivery_days'] = (
    m['order_estimated_delivery_date'] - m['order_purchase_timestamp']).dt.total_seconds() / 86400
m['purchase_month'] = m['order_purchase_timestamp'].dt.month
m['purchase_hour'] = m['order_purchase_timestamp'].dt.hour
m['purchase_dayofweek'] = m['order_purchase_timestamp'].dt.dayofweek

FEAT = ['order_approved_wait_days', 'estimated_delivery_days',
        'purchase_month', 'purchase_hour', 'purchase_dayofweek']
TRACK = ['order_id', 'customer_id', 'split', 'low_rating', 'review_id', 'review_score',
         'review_creation_date', 'n_reviews', 'order_purchase_timestamp',
         'order_delivered_customer_date', 'order_estimated_delivery_date']
base = m[TRACK + FEAT].copy()
assert base['order_id'].is_unique and len(base) == 87514
base.to_parquet(PROC / 'base_orders.parquet', index=False)
base.to_csv(PROC / 'base_orders.csv', index=False, encoding='utf-8-sig')
print(f'[out] base_orders.parquet / base_orders.csv: {base.shape[0]} 行 x {base.shape[1]} 列，order_id 唯一')

# ---------- 训练集 EDA ----------
tr = base[base['split'] == 'train']
te = base[base['split'] == 'test']
va = base[base['split'] == 'val']
print(f'\n[EDA] train={len(tr)} val={len(va)} test={len(te)}')
print(f'[EDA] 训练集低评率: {tr["low_rating"].mean():.3%}')

# 图 1：类别分布（train）
cnt = tr['low_rating'].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(4.5, 3.2))
bars = ax.bar(['Not low (4-5)', 'Low (1-3)'], cnt.values,
              color=[BLUE, ORANGE], width=0.55)
for b, v in zip(bars, cnt.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 800, f'{v:,}', ha='center',
            fontsize=9, color='#52514e')
ax.set_title('Train set: low_rating distribution')
ax.set_ylabel('orders')
ax.grid(axis='x', visible=False)
ax.margins(y=0.12)
fig.tight_layout()
fig.savefig(OUT / 'fig_class_distribution.png')
plt.close(fig)

# 图 2：特征缺失率（train）
miss = tr[FEAT].isna().mean().sort_values(ascending=False) * 100
fig, ax = plt.subplots(figsize=(5.5, 3.0))
bars = ax.barh(miss.index, miss.values, color=BLUE, height=0.55)
for b, v in zip(bars, miss.values):
    ax.text(v + 0.3, b.get_y() + b.get_height() / 2, f'{v:.2f}%',
            va='center', fontsize=8, color='#52514e')
ax.set_title('Train set: feature missingness (%)')
ax.set_xlabel('% missing')
ax.grid(axis='y', visible=False)
ax.set_xlim(0, max(miss.max() * 1.25, 1))
fig.tight_layout()
fig.savefig(OUT / 'fig_missingness.png')
plt.close(fig)

# 图 3：两个连续特征直方图（train，展示时截断到 99.5% 分位，仅影响展示不影响数据）
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
for ax, col, label in zip(axes,
                          ['order_approved_wait_days', 'estimated_delivery_days'],
                          ['Approved wait (days)', 'Estimated delivery (days)']):
    s = tr[col].dropna()
    q = s.quantile(0.995)
    ax.hist(s.clip(upper=q), bins=40, color=BLUE)
    ax.set_title(f'{label}  (x clipped at 99.5th pct = {q:.1f} d)')
    ax.set_ylabel('orders')
fig.tight_layout()
fig.savefig(OUT / 'fig_feature_histograms.png')
plt.close(fig)

# 图 4：各 split 按送达月份的订单量（3 类固定色序）
timeline = (base.assign(month=base['order_delivered_customer_date'].dt.to_period('M'))
                .groupby(['month', 'split']).size().unstack(fill_value=0))
timeline = timeline[['train', 'val', 'test']]  # unstack 默认字母序，图例顺序改回 train/val/test
colors = {'train': BLUE, 'val': ORANGE, 'test': AQUA}
ax = timeline.plot(kind='bar', stacked=True, figsize=(9, 3.4),
                   color=[colors[c] for c in timeline.columns], width=0.85)
ax.set_title('Eligible orders by delivery month and split')
ax.set_ylabel('orders')
ax.legend(title='split', fontsize=8)
ax.tick_params(axis='x', labelsize=8)
ax.grid(axis='x', visible=False)
fig = ax.get_figure()
fig.tight_layout()
fig.savefig(OUT / 'fig_split_timeline.png')
plt.close(fig)

# 图 5：低评率随送达月份漂移（含划分边界，回应"时间漂移怎么办"）
monthly = (base.assign(month=base['order_delivered_customer_date'].dt.to_period('M'))
               .groupby('month')['low_rating'].mean())
roll = monthly.rolling(3, min_periods=1).mean()
x_ts = monthly.index.to_timestamp()
fig, ax = plt.subplots(figsize=(9, 3.4))
ax.plot(x_ts, monthly.values, color=BLUE, linewidth=2, label='Monthly low-rate')
ax.plot(x_ts, roll.values, color=BLUE, alpha=0.45, linewidth=1.5, linestyle='--',
        label='3-month rolling mean')
for split_name in ['val', 'test']:
    b = base.loc[base['split'] == split_name, 'order_delivered_customer_date'].min()
    ax.axvline(b, color='#c3c2b7', linewidth=1, linestyle=':')
    ax.text(b, ax.get_ylim()[1] * 0.97, f'{split_name} start', fontsize=8,
            color='#898781', rotation=90, va='top', ha='left')
ax.set_title('Low-rate drift by delivery month (eligible orders)')
ax.set_ylabel('low_rating rate')
ax.legend(fontsize=8, loc='upper right')
ax.tick_params(axis='x', labelsize=8)
ax.grid(axis='x', visible=False)
fig.tight_layout()
fig.savefig(OUT / 'fig_lowrate_drift.png')
plt.close(fig)

# ---------- 测试集仅做完整性检查 ----------
print(f'\n[integrity] test 行数 {len(te)} | 低评订单数 {int(te["low_rating"].sum())} '
      f'| 低评率 {te["low_rating"].mean():.3%}')
print(f'[integrity] test 日期范围 {te["order_delivered_customer_date"].min().date()} ~ '
      f'{te["order_delivered_customer_date"].max().date()}')
print(f'[integrity] test 特征缺失数:\n{te[FEAT].isna().sum().to_string()}')

# EDA 汇总存档
eda = pd.DataFrame({
    'metric': ['n_orders', 'low_rate', 'n_low'] + [f'missing_{c}' for c in FEAT],
    'train': [len(tr), tr['low_rating'].mean(), int(tr['low_rating'].sum())]
             + tr[FEAT].isna().sum().tolist(),
    'val': [len(va), va['low_rating'].mean(), int(va['low_rating'].sum())]
           + va[FEAT].isna().sum().tolist(),
    'test': [len(te), te['low_rating'].mean(), int(te['low_rating'].sum())]
            + te[FEAT].isna().sum().tolist(),
})
eda.to_csv(PROC / 'eda_train_summary.csv', index=False, encoding='utf-8-sig')
print('\n[out] data/processed/eda_train_summary.csv、outputs/figures/ 5 张图已保存')
