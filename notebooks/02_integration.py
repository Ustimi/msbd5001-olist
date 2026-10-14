"""Part B: aggregate items/payments, prepare customers and build E2 master.

Aggregates all raw item orders; master uses A's authoritative base unchanged.
Average product weight is item-row weighted; missing weights remain unfilled.
Run with: conda run -n olist python notebooks/02_integration.py
Run only payments with: conda run -n olist python notebooks/02_integration.py --stage payments
Run only customers with: conda run -n olist python notebooks/02_integration.py --stage customers
Run only master with: conda run -n olist python notebooks/02_integration.py --stage master
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"


def aggregate_items():
    items = pd.read_csv(RAW / "olist_order_items_dataset.csv")
    products = pd.read_csv(RAW / "olist_products_dataset.csv")
    sellers = pd.read_csv(RAW / "olist_sellers_dataset.csv")

    for name, frame in [("items", items), ("products", products), ("sellers", sellers)]:
        print(f"[input] {name}: {len(frame):,} rows")
        print(f"[missing] {name}:\n{frame.isna().sum().to_string()}")
    print(f"[input] unique item orders: {items['order_id'].nunique():,}")

    assert items[['order_id', 'order_item_id', 'product_id', 'seller_id']].notna().all().all(), "Missing item keys"
    assert not items.duplicated(['order_id', 'order_item_id']).any(), "Duplicate item keys"
    assert products['product_id'].notna().all() and products['product_id'].is_unique, "Invalid product keys"
    assert sellers['seller_id'].notna().all() and sellers['seller_id'].is_unique, "Invalid seller keys"
    assert items[['price', 'freight_value']].notna().all().all(), "Missing monetary values"

    enriched = items.merge(
        products, on="product_id", how="left", validate="many_to_one",
        indicator="product_match",
    ).merge(
        sellers, on="seller_id", how="left", validate="many_to_one",
        indicator="seller_match",
    )
    assert len(enriched) == len(items), "Dimension joins changed item row count"
    for column in ['product_match', 'seller_match']:
        unmatched = int((enriched[column] != 'both').sum())
        print(f"[join] {column}: {unmatched:,} unmatched item rows")
    print(f"[missing] enriched items:\n{enriched.isna().sum().to_string()}")

    result = enriched.groupby('order_id', as_index=False).agg(
        order_item_count=('order_item_id', 'size'),
        seller_count=('seller_id', 'nunique'),
        total_price=('price', 'sum'),
        total_freight=('freight_value', 'sum'),
        avg_product_weight_g=('product_weight_g', 'mean'),
    )

    assert result['order_id'].is_unique, "Duplicate output orders"
    assert set(result['order_id']) == set(items['order_id']), "Order coverage changed"
    assert result['order_item_count'].sum() == len(items), "Item count changed"
    for source, target in [('price', 'total_price'), ('freight_value', 'total_freight')]:
        before = items[source].sum()
        after = result[target].sum()
        np.testing.assert_allclose(after, before, rtol=0, atol=1e-6)
        print(f"[check] {source}: input={before:.2f}, output={after:.2f}; preserved")

    print(f"[missing] output:\n{result.isna().sum().to_string()}")
    PROC.mkdir(parents=True, exist_ok=True)
    path = PROC / '02_items_agg.parquet'
    result.to_parquet(path, index=False)
    result.to_csv(PROC / '02_items_agg.csv', index=False, encoding='utf-8-sig')
    pd.testing.assert_frame_equal(pd.read_parquet(path), result)
    print(f"[output] 02_items_agg.parquet / 02_items_agg.csv: shape={result.shape}")
    print('[check] uniqueness, order coverage, item count, monetary totals and parquet round-trip passed')


def aggregate_payments():
    payments = pd.read_csv(RAW / 'olist_order_payments_dataset.csv')
    print(f"[input] payments: shape={payments.shape}")
    print(f"[input] unique payment orders: {payments['order_id'].nunique():,}")
    print(f"[missing] payments:\n{payments.isna().sum().to_string()}")

    assert len(payments) == 103886, 'Unexpected payment row count'
    assert payments['order_id'].nunique() == 99440, 'Unexpected payment order count'
    assert payments.notna().all().all(), 'Missing payment fields'
    assert not payments.duplicated(['order_id', 'payment_sequential']).any(), 'Duplicate payment keys'
    assert np.isfinite(payments['payment_value']).all(), 'Non-finite payment amounts'
    negative_count = int((payments['payment_value'] < 0).sum())
    print(f'[quality] negative payment amounts: {negative_count:,}')
    assert negative_count == 0, 'Negative payment amounts'
    zero_installments = int((payments['payment_installments'] == 0).sum())
    print(f'[quality] zero-installment records: {zero_installments:,}; retained without filtering')
    print(f"[quality] payment types:\n{payments['payment_type'].value_counts().to_string()}")

    payments['has_credit_card'] = payments['payment_type'].eq('credit_card')
    payments['has_voucher'] = payments['payment_type'].eq('voucher')
    result = payments.groupby('order_id', as_index=False).agg(
        payment_total=('payment_value', 'sum'),
        payment_count=('payment_sequential', 'size'),
        max_installments=('payment_installments', 'max'),
        payment_type_count=('payment_type', 'nunique'),
        has_credit_card=('has_credit_card', 'any'),
        has_voucher=('has_voucher', 'any'),
    )

    assert result['order_id'].is_unique and len(result) == 99440, 'Invalid output order count'
    assert set(result['order_id']) == set(payments['order_id']), 'Payment order coverage changed'
    assert result['payment_count'].sum() == len(payments), 'Payment records lost'
    before = payments['payment_value'].sum()
    after = result['payment_total'].sum()
    np.testing.assert_allclose(after, before, rtol=0, atol=1e-6)
    print(f'[check] payment value: input={before:.2f}, output={after:.2f}; preserved')
    print(f"[missing] payment output:\n{result.isna().sum().to_string()}")

    PROC.mkdir(parents=True, exist_ok=True)
    path = PROC / '02_payments_agg.parquet'
    result.to_parquet(path, index=False)
    result.to_csv(PROC / '02_payments_agg.csv', index=False, encoding='utf-8-sig')
    pd.testing.assert_frame_equal(pd.read_parquet(path), result)
    print(f'[output] 02_payments_agg.parquet / 02_payments_agg.csv: shape={result.shape}')
    print('[check] input counts, unique orders, order coverage, record count, nonnegative amounts, monetary total and parquet round-trip passed')


def prepare_customers():
    """Save and return the customer lookup; IDs/zip codes are tracking only.

    State and city are candidate categorical 02_features. This function does not
    encode features or join base_orders. Preserve the original zip text,
    including leading zeros, rather than treating it as a numeric feature.
    """
    customers = pd.read_csv(
        RAW / 'olist_customers_dataset.csv',
        dtype={'customer_zip_code_prefix': 'string'},
    )
    print(f'[input] customers: shape={customers.shape}')
    print(f"[schema] customers:\n{customers.dtypes.to_string()}")
    print(f"[missing] customers:\n{customers.isna().sum().to_string()}")
    unique_count = customers['customer_id'].nunique()
    duplicate_count = int(customers.duplicated('customer_id').sum())
    print(f'[check] unique customer IDs: {unique_count:,}; duplicate IDs: {duplicate_count:,}')
    assert customers['customer_id'].notna().all(), 'Missing customer IDs'
    assert customers['customer_id'].is_unique, 'Duplicate customer IDs'

    result = customers[[
        'customer_id', 'customer_state', 'customer_city', 'customer_zip_code_prefix',
    ]].copy()
    assert len(result) == len(customers), 'Customer rows changed'
    assert result['customer_id'].is_unique, 'Duplicate lookup keys'
    state_counts = result['customer_state'].value_counts(dropna=False).sort_index()
    print(f"[coverage] states: {result['customer_state'].nunique():,}")
    print(f'[coverage] customer rows by state:\n{state_counts.to_string()}')
    print(f"[missing] customer lookup:\n{result.isna().sum().to_string()}")

    PROC.mkdir(parents=True, exist_ok=True)
    path = PROC / '02_customers_lookup.parquet'
    result.to_parquet(path, index=False)
    result.to_csv(PROC / '02_customers_lookup.csv', index=False, encoding='utf-8-sig')
    pd.testing.assert_frame_equal(pd.read_parquet(path), result)
    print(f'[output] 02_customers_lookup.parquet / 02_customers_lookup.csv: shape={result.shape}')
    print('[check] nonmissing unique customer keys, row preservation and parquet round-trip passed')
    print('[features] customer_id and customer_zip_code_prefix are for joining/traceability only')
    return result


def build_master_orders():
    """Left join E2 inputs; preserve all base columns and missing values.

    Review fields remain tracking columns only, never model predictors.
    IDs and customer zip codes are also excluded from candidate predictors.
    No E3 features, imputation or model encoding are performed here.
    """
    base = pd.read_parquet(PROC / 'base_orders.parquet')
    assert base.columns.is_unique, 'Duplicate base column names'
    assert len(base) == 87514 and base['order_id'].is_unique, 'Invalid base orders'
    assert base['order_id'].notna().all(), 'Missing base order keys'
    expected_splits = {'train': 61260, 'val': 13127, 'test': 13127}
    assert base['split'].value_counts().to_dict() == expected_splits, 'Unexpected base splits'
    assert base['low_rating'].isin([0, 1]).all(), 'Invalid base labels'
    print(f'[input] base_orders: shape={base.shape}')

    result = base.copy()
    audits = []
    for name, key, relationship in [
        ('02_items_agg', 'order_id', 'one_to_one'),
        ('02_payments_agg', 'order_id', 'one_to_one'),
        ('02_customers_lookup', 'customer_id', 'many_to_one'),
    ]:
        lookup = pd.read_parquet(PROC / f'{name}.parquet')
        print(f'[input] {name}: shape={lookup.shape}')
        print(f"[missing] {name}:\n{lookup.isna().sum().to_string()}")
        assert lookup.columns.is_unique, f'Duplicate columns in {name}'
        assert lookup[key].notna().all() and lookup[key].is_unique, f'Invalid {name} keys'
        overlaps = (set(result.columns) & set(lookup.columns)) - {key}
        assert not overlaps, f'Overlapping non-key columns in {name}: {overlaps}'
        rows_before = len(result)
        left_key_unique = result[key].is_unique
        result = result.merge(
            lookup, on=key, how='left', validate=relationship, indicator='_join_match',
        )
        matched = int(result['_join_match'].eq('both').sum())
        unmatched = int(result['_join_match'].eq('left_only').sum())
        result = result.drop(columns='_join_match')
        assert len(result) == rows_before, f'{name} changed row count'
        assert result['order_id'].is_unique, f'{name} duplicated orders'
        assert result.columns.is_unique, f'{name} duplicated columns'
        assert set(result['order_id']) == set(base['order_id']), f'{name} changed order coverage'
        audits.append({
            'join': name, 'key': key, 'validation': relationship,
            'rows_before': rows_before, 'lookup_rows': len(lookup),
            'rows_after': len(result), 'matched_rows': matched,
            'unmatched_rows': unmatched, 'match_rate': matched / rows_before,
            'left_key_unique': left_key_unique, 'lookup_key_unique': lookup[key].is_unique,
            'output_order_id_unique': result['order_id'].is_unique,
            'output_unique_orders': result['order_id'].nunique(),
            'output_duplicate_orders': int(result.duplicated('order_id').sum()),
            'order_set_preserved': True,
        })
        print(f'[join] {name}: matched={matched:,}, unmatched={unmatched:,}, rate={matched / rows_before:.4%}')

    assert len(result) == result['order_id'].nunique() == 87514, 'Invalid master order count'
    # Compare by order_id, including exact values, dtypes and missingness of
    # every original column: labels, split, E1 features and tracking fields.
    expected = base.set_index('order_id').sort_index()
    actual = result[base.columns].set_index('order_id').sort_index()
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert result['split'].value_counts().to_dict() == expected_splits, 'Split counts changed'
    new_columns = [c for c in result.columns if c not in base.columns]
    print(f"[missing] newly added columns:\n{result[new_columns].isna().sum().to_string()}")
    print(f"[check] splits: {result['split'].value_counts().to_dict()}")
    print('[check] all base columns, labels and splits unchanged by order_id; unique orders and order coverage passed')
    excluded = {
        'order_id', 'customer_id', 'split', 'low_rating', 'review_id',
        'review_score', 'review_creation_date', 'n_reviews',
        'order_purchase_timestamp', 'order_delivered_customer_date',
        'order_estimated_delivery_date', 'customer_zip_code_prefix',
    }
    candidates = [c for c in result.columns if c not in excluded]
    print(f'[features] candidate E1/E2 predictors (tracking/review fields excluded): {candidates}')

    result.to_parquet(PROC / '02_master_orders.parquet', index=False)
    result.to_csv(PROC / '02_master_orders.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(audits).to_csv(PROC / '02_join_audit.csv', index=False, encoding='utf-8-sig')
    pd.testing.assert_frame_equal(pd.read_parquet(PROC / '02_master_orders.parquet'), result, check_exact=True)
    print(f'[output] 02_master_orders.parquet / 02_master_orders.csv: shape={result.shape}; 02_join_audit.csv saved')
    print('[check] master parquet round-trip passed')
    return result


def engineer_e3():
    """Derive delivery-time features without altering E2 or using reviews as X."""
    master = pd.read_parquet(PROC / '02_master_orders.parquet')
    assert len(master) == 87514 and master['order_id'].is_unique
    definitions = {
        'delivery_duration_days': ('(actual_delivery - purchase).total_seconds / 86400', ['orders'], 'customer delivery', 'Missing if either timestamp is missing'),
        'delivery_delay_days': ('(actual_delivery - estimated_delivery).total_seconds / 86400; signed', ['orders'], 'customer delivery', 'Missing if either timestamp is missing; early delivery stays negative'),
        'late_delivery': ('1 if actual_delivery > estimated_delivery else 0', ['orders'], 'customer delivery', 'Missing if either timestamp is missing'),
        'freight_ratio': ('total_freight / total_price where total_price > 0', ['items'], 'customer delivery', 'Missing for missing/nonpositive price or missing freight'),
        'invalid_price': ('1 if total_price is missing or <= 0 else 0', ['items'], 'customer delivery', 'Always defined'),
        'multi_seller': ('1 if seller_count > 1 else 0', ['items'], 'customer delivery', 'Missing if seller_count is missing'),
        'same_state_share': ('same-state item rows / item rows with both seller and customer state available', ['items', 'sellers', 'orders', 'customers'], 'customer delivery', 'Missing when no item rows have both states'),
    }
    assert not set(definitions) & set(master.columns), 'E3 columns already exist in E2'
    times = {c: pd.to_datetime(master[c], errors='raise') for c in [
        'order_purchase_timestamp', 'order_delivered_customer_date',
        'order_estimated_delivery_date', 'review_creation_date',
    ]}
    purchase = times['order_purchase_timestamp']
    actual = times['order_delivered_customer_date']
    estimated = times['order_estimated_delivery_date']
    review = times['review_creation_date']
    assert all(t.notna().all() for t in times.values()), 'Missing eligible-sample timestamps'
    assert (purchase < review).all() and (actual < review).all(), 'Purchase/delivery not before first review'
    print(f'[time] purchase and actual delivery precede review for all {len(master):,} orders')
    print(f'[time] estimated delivery >= first review: {int((estimated >= review).sum()):,}; promise date, not a future observed event')

    items = pd.read_csv(RAW / 'olist_order_items_dataset.csv', usecols=['order_id', 'order_item_id', 'seller_id'])
    sellers = pd.read_csv(RAW / 'olist_sellers_dataset.csv', usecols=['seller_id', 'seller_state'])
    customers = pd.read_csv(RAW / 'olist_customers_dataset.csv', usecols=['customer_id', 'customer_state'])
    for frame, key in [(sellers, 'seller_id'), (customers, 'customer_id')]:
        assert frame[key].notna().all() and frame[key].is_unique
        print(f'[missing] raw {key} lookup:\n{frame.isna().sum().to_string()}')
    assert items[['order_id', 'seller_id']].notna().all().all()
    assert not items.duplicated(['order_id', 'order_item_id']).any()
    eligible = items[items['order_id'].isin(master['order_id'])].copy()
    n_items = len(eligible)
    for lookup, key in [
        (sellers, 'seller_id'),
        (master[['order_id', 'customer_id']], 'order_id'),
        (customers, 'customer_id'),
    ]:
        eligible = eligible.merge(lookup, on=key, how='left', validate='many_to_one', indicator='_state_match')
        print(f'[join] item states via {key}: unmatched={int(eligible._state_match.eq("left_only").sum()):,}')
        eligible = eligible.drop(columns='_state_match')
        assert len(eligible) == n_items, 'State join multiplied item rows'
    customer_states = master[['order_id', 'customer_id', 'customer_state']].merge(
        customers, on='customer_id', how='left', validate='many_to_one', suffixes=('_e2', '_raw'),
    )
    pd.testing.assert_series_equal(customer_states.customer_state_e2, customer_states.customer_state_raw, check_names=False)
    valid_states = eligible.seller_state.notna() & eligible.customer_state.notna()
    eligible['state_match'] = eligible.seller_state.eq(eligible.customer_state).astype(float).where(valid_states)
    state_share = eligible.groupby('order_id', as_index=False).agg(same_state_share=('state_match', 'mean'))
    result = master.merge(state_share, on='order_id', how='left', validate='one_to_one')
    print(f'[states] eligible item rows={n_items:,}; rows without both states={int((~valid_states).sum()):,}')
    result['delivery_duration_days'] = (actual - purchase).dt.total_seconds() / 86400
    result['delivery_delay_days'] = (actual - estimated).dt.total_seconds() / 86400
    result['late_delivery'] = actual.gt(estimated).astype('Int64').where(actual.notna() & estimated.notna())
    valid_price = result.total_price.notna() & result.total_price.gt(0)
    result['freight_ratio'] = result.total_freight / result.total_price.where(valid_price)
    result['invalid_price'] = (~valid_price).astype('Int64')
    result['multi_seller'] = result.seller_count.gt(1).astype('Int64').where(result.seller_count.notna())

    assert len(result) == result.order_id.nunique() == 87514
    assert result.columns.is_unique and set(result.order_id) == set(master.order_id)
    pd.testing.assert_frame_equal(
        result[master.columns].set_index('order_id').sort_index(),
        master.set_index('order_id').sort_index(), check_exact=True,
    )
    assert result.split.value_counts().to_dict() == {'train': 61260, 'val': 13127, 'test': 13127}
    assert result.same_state_share.dropna().between(0, 1).all()
    for c in ['late_delivery', 'multi_seller', 'invalid_price']:
        assert result[c].dropna().isin([0, 1]).all()
    expected_missing = {
        'delivery_duration_days': purchase.isna() | actual.isna(),
        'delivery_delay_days': actual.isna() | estimated.isna(),
        'late_delivery': actual.isna() | estimated.isna(),
        'freight_ratio': ~valid_price | result.total_freight.isna(),
        'invalid_price': pd.Series(False, index=result.index),
        'multi_seller': result.seller_count.isna(),
        'same_state_share': ~result.order_id.isin(eligible.loc[valid_states, 'order_id']),
    }
    diagnostics = []
    for c in definitions:
        s = result[c]
        assert s.isna().equals(expected_missing[c]), f'Unexpected missing values: {c}'
        assert np.isfinite(s.dropna().astype(float)).all(), f'Infinite values: {c}'
        q = s.dropna().astype(float).quantile([0.01, 0.5, 0.99])
        diagnostics.append({'feature': c, 'missing': int(s.isna().sum()),
                            'min': s.min(), 'p01': q.loc[0.01], 'median': q.loc[0.5],
                            'p99': q.loc[0.99], 'max': s.max(),
                            'negative': int(s.lt(0).sum()),
                            'below_p01': int(s.lt(q.loc[0.01]).sum()),
                            'above_p99': int(s.gt(q.loc[0.99]).sum())})
        print(f'[extremes] {c}:\n{result.loc[s.notna(), ["order_id", c]].sort_values(c).iloc[[0, -1]].to_string(index=False)}')
    summary = pd.DataFrame(diagnostics)
    print(f'[diagnostics]\n{summary.to_string(index=False)}')
    print(f'[anomalies] duration > 100 days: {int(result.delivery_duration_days.gt(100).sum())}; delay > 30 days: {int(result.delivery_delay_days.gt(30).sum())}; freight ratio > 1: {int(result.freight_ratio.gt(1).sum())}; zero durations: {int(result.delivery_duration_days.eq(0).sum())}')

    result.to_parquet(PROC / '02_master_orders_e3.parquet', index=False)
    result.to_csv(PROC / '02_master_orders_e3.csv', index=False, encoding='utf-8-sig')
    summary.to_csv(PROC / '02_e3_feature_checks.csv', index=False, encoding='utf-8-sig')
    dictionary = {
        'prediction_time': 'Actual customer delivery, before first review',
        'timestamp_policy': 'Naive Olist timestamps consistently compared as supplied; fractional days = seconds / 86400; no date truncation',
        'review_policy': 'Review creation time used only for temporal validation; review fields never predictors',
        'excluded_predictors': ['order_id', 'customer_id', 'customer_zip_code_prefix', 'split', 'low_rating', 'review_id', 'review_score', 'review_creation_date', 'n_reviews', 'order_purchase_timestamp', 'order_delivered_customer_date', 'order_estimated_delivery_date'],
        'E3': {name: dict(formula=formula, source_tables=sources, available_at=available, missing_policy=missing)
               for name, (formula, sources, available, missing) in definitions.items()},
    }
    (PROC / '02_features_e3.json').write_text(json.dumps(dictionary, indent=2) + '\n', encoding='utf-8')
    pd.testing.assert_frame_equal(pd.read_parquet(PROC / '02_master_orders_e3.parquet'), result, check_exact=True)
    print(f'[output] 02_master_orders_e3 shape={result.shape}; all E2 columns, labels and splits unchanged; all E3 validations and round-trip passed')
    return result


def prepare_feature_specs():
    """Write provisional nested experiment lists and C's modeling handoff."""
    paths = {'E1': 'base_orders', 'E2': '02_master_orders', 'E3': '02_master_orders_e3'}
    data = {stage: pd.read_parquet(PROC / f'{name}.parquet') for stage, name in paths.items()}
    e3_dict = json.loads((PROC / '02_features_e3.json').read_text(encoding='utf-8'))
    # Member B provisionally assigns purchase-only calendar features to E1;
    # members A and C must still confirm this team decision.
    pending = ['purchase_month', 'purchase_hour', 'purchase_dayofweek']
    metadata = {
        'order_approved_wait_days': ('E1', '批准等待天数', 'orders', '(approved_at - purchase).total_seconds / 86400', 'numeric_continuous'),
        'estimated_delivery_days': ('E1', '预计配送天数', 'orders', '(estimated_delivery - purchase).total_seconds / 86400', 'numeric_continuous'),
        'purchase_month': ('E1', '下单月份；暂定归 E1，待 A/C 确认', 'orders', 'purchase.dt.month (1-12)', 'categorical_calendar'),
        'purchase_hour': ('E1', '下单小时；暂定归 E1，待 A/C 确认', 'orders', 'purchase.dt.hour (0-23)', 'categorical_calendar'),
        'purchase_dayofweek': ('E1', '下单星期；暂定归 E1，待 A/C 确认', 'orders', 'purchase.dt.dayofweek (0=Monday)', 'categorical_calendar'),
        'order_item_count': ('E2', '订单商品明细行数', 'items', 'count item rows per order', 'numeric_count'),
        'seller_count': ('E2', '订单不同卖家数', 'items', 'nunique seller_id per order', 'numeric_count'),
        'total_price': ('E2', '商品金额合计', 'items', 'sum price per order', 'numeric_continuous'),
        'total_freight': ('E2', '运费合计', 'items', 'sum freight_value per order', 'numeric_continuous'),
        'avg_product_weight_g': ('E2', '按商品明细行计算的平均重量（克）', 'items;products', 'mean product_weight_g; skip missing, no fill', 'numeric_continuous'),
        'payment_total': ('E2', '支付金额合计', 'payments', 'sum payment_value per order', 'numeric_continuous'),
        'payment_count': ('E2', '支付记录数', 'payments', 'count payment rows per order', 'numeric_count'),
        'max_installments': ('E2', '最大分期数', 'payments', 'max payment_installments per order; retain zero', 'numeric_count'),
        'payment_type_count': ('E2', '不同支付方式数', 'payments', 'nunique payment_type per order; include not_defined', 'numeric_count'),
        'has_credit_card': ('E2', '是否使用信用卡', 'payments', 'any payment_type == credit_card', 'binary'),
        'has_voucher': ('E2', '是否使用代金券', 'payments', 'any payment_type == voucher', 'binary'),
        'customer_state': ('E2', '客户州', 'customers', 'lookup by customer_id', 'categorical_nominal'),
        'customer_city': ('E2', '客户城市，保留原始拼写', 'customers', 'lookup by customer_id', 'categorical_nominal'),
    }
    descriptions = {
        'delivery_duration_days': '实际配送天数', 'delivery_delay_days': '带符号的配送延迟天数',
        'late_delivery': '是否晚到', 'freight_ratio': '运费与商品金额比值',
        'invalid_price': '商品金额缺失或非正值标记', 'multi_seller': '是否多卖家',
        'same_state_share': '有效州信息商品行中同州比例',
    }
    for name, entry in e3_dict['E3'].items():
        metadata[name] = ('E3', descriptions[name], ';'.join(entry['source_tables']), entry['formula'],
                          'binary' if name in ['late_delivery', 'invalid_price', 'multi_seller'] else 'numeric_continuous')
    lists = {stage: [name for name, entry in metadata.items() if int(entry[0][1]) <= int(stage[1])]
             for stage in paths}
    excluded = e3_dict['excluded_predictors']
    assert set(lists['E1']) <= set(lists['E2']) <= set(lists['E3'])
    checks = []
    rows = []
    for stage, frame in data.items():
        assert len(frame) == frame.order_id.nunique() == 87514
        assert frame.split.value_counts().to_dict() == {'train': 61260, 'val': 13127, 'test': 13127}
        assert frame.low_rating.isin([0, 1]).all()
        assert len(lists[stage]) == len(set(lists[stage]))
        assert set(lists[stage]) <= set(frame.columns)
        assert not set(lists[stage]) & set(excluded)
        assert set(pending) <= set(lists[stage])
        for name in lists[stage]:
            assert name in metadata and not name.endswith('_id') and not name.startswith('review_')
            assert not pd.api.types.is_datetime64_any_dtype(frame[name])
            if metadata[name][4] == 'binary':
                assert frame[name].dropna().isin([0, 1]).all()
            elif metadata[name][4].startswith('numeric'):
                assert np.isfinite(frame[name].dropna().astype(float)).all()
        checks.append({'stage': stage, 'feature_count': len(lists[stage]), 'rows': len(frame),
                       'features_exist': True, 'forbidden_predictors_excluded': True,
                       'unique_orders': True, 'split_counts_preserved': True})
        pd.testing.assert_frame_equal(
            frame[data['E1'].columns].set_index('order_id').sort_index(),
            data['E1'].set_index('order_id').sort_index(), check_exact=True,
        )
    pd.testing.assert_frame_equal(
        data['E3'][data['E2'].columns].set_index('order_id').sort_index(),
        data['E2'].set_index('order_id').sort_index(), check_exact=True,
    )
    for name, (stage, desc, source, formula, kind) in metadata.items():
        frame = data[stage]
        rows.append(dict(feature_name=name, experiment_stage=stage,
                         used_in=';'.join(s for s in paths if name in lists[s]), description=desc,
                         source=source, formula=formula, data_type=kind, storage_dtype=str(frame[name].dtype),
                         missing_value_count=int(frame[name].isna().sum()),
                         prediction_time_availability=('Available at purchase' if name in pending else 'Available at actual customer delivery, before first review'),
                         status='provisional included', dataset=f'{paths[stage]}.parquet'))
    specs = {**lists, 'status': 'provisional_calendar_E1_pending_A_C_confirmation',
             'datasets': {s: f'data/processed/{n}.parquet' for s, n in paths.items()},
             'label': 'low_rating', 'split': 'split', 'prediction_time': e3_dict['prediction_time'],
             'feature_types': {n: entry[4] for n, entry in metadata.items()},
             'excluded_predictors': excluded,
             'pending_decisions': [{'features': pending, 'source': 'docs/A_work_log.md section 7; notebooks/01_base.py candidate list',
                                    'decision': 'Provisionally E1 because derived solely from original purchase timestamp; pending A and C confirmation',
                                    'current_policy': 'Included in E1, E2 and E3; provisional decision by member B on 2026-10-14'}],
             'preprocessing': 'Fit imputers, encoders, scalers and data-driven thresholds on train only; transform val/test'}
    (PROC / '02_features.json').write_text(json.dumps(specs, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    dictionary = pd.DataFrame(rows)
    dictionary.to_csv(PROC / '02_feature_dictionary.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(checks).to_csv(PROC / '02_feature_spec_checks.csv', index=False, encoding='utf-8-sig')
    sections = '\n\n'.join(f"### {s}（{len(lists[s])} 个，累计清单）\n\n" + ', '.join(f'`{n}`' for n in lists[s]) for s in paths)
    handoff = f'''# B → C 建模交接

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

{sections}

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
'''
    (ROOT / 'docs' / '02_B_handoff.md').write_text(handoff, encoding='utf-8')
    assert json.loads((PROC / '02_features.json').read_text(encoding='utf-8')) == specs
    assert pd.read_csv(PROC / '02_feature_dictionary.csv')['feature_name'].tolist() == dictionary.feature_name.tolist()
    print(pd.DataFrame(checks).to_string(index=False))
    print('[check] nested lists, actual schemas, predictor types, finite numeric values, identifiers/reviews excluded, exact cross-stage preservation and specification round-trip passed')
    print(f'[pending] calendar features: {pending}; provisionally included in E1/E2/E3, pending A/C confirmation')
    print('[output] 02_features.json, 02_feature_dictionary.csv, 02_feature_spec_checks.csv, docs/02_B_handoff.md; datasets unchanged')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['all', 'items', 'payments', 'customers', 'master', 'e3', 'specs'], default='all')
    args = parser.parse_args()
    if args.stage in ['all', 'items']:
        aggregate_items()
    if args.stage in ['all', 'payments']:
        aggregate_payments()
    if args.stage in ['all', 'customers']:
        prepare_customers()
    if args.stage in ['all', 'master']:
        build_master_orders()
    if args.stage in ['all', 'e3']:
        engineer_e3()
    if args.stage in ['all', 'specs']:
        prepare_feature_specs()


if __name__ == '__main__':
    main()
