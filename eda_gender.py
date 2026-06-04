# -*- coding: utf-8 -*-
"""성별별 카테고리/금액 분포 차이 EDA"""
import pandas as pd
import numpy as np

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 50)

import os
DATA = "competition_data" if os.path.exists(os.path.join("competition_data","train_transactions.csv")) else "."

tx = pd.read_csv(os.path.join(DATA, "train_transactions.csv"), encoding='cp949')
y  = pd.read_csv(os.path.join(DATA, "y_train.csv"))

print("=== 1. 기본 정보 ===")
print("거래 shape:", tx.shape, "| 고객 수(거래):", tx['custid'].nunique())
print("y shape:", y.shape, "| 고객 수(y):", y['custid'].nunique())
print("\n성별 분포(y):")
print(y['gender'].value_counts())
print((y['gender'].value_counts(normalize=True)*100).round(2).astype(str)+'%')

print("\n결측치(거래):")
print(tx.isna().sum()[tx.isna().sum()>0])

# 거래에 성별 부여
df = tx.merge(y, on='custid', how='left')
print("\ny와 매칭 안된 거래 수:", df['gender'].isna().sum())

# 파생
df['sales_datetime'] = pd.to_datetime(df['sales_datetime'])
df['month'] = df['sales_datetime'].dt.month
df['hour'] = df['sales_datetime'].dt.hour
df['weekday'] = df['sales_datetime'].dt.weekday
df['is_weekend'] = df['weekday'].isin([5,6]).astype(int)
df['is_refund'] = (df['net_amt'] < 0).astype(int)

print("\n=== 2. 고객 단위 금액/행동 성별 비교 ===")
cust = df.groupby('custid').agg(
    n_txn=('goodcd','size'),
    total_net=('net_amt','sum'),
    mean_net=('net_amt','mean'),
    n_brand=('brd_nm','nunique'),
    import_ratio=('import_flg','mean'),
    weekend_ratio=('is_weekend','mean'),
    refund_ratio=('is_refund','mean'),
).merge(y, on='custid')

print(cust.groupby('gender').agg(
    cnt=('custid','size'),
    n_txn_med=('n_txn','median'),
    total_net_med=('total_net','median'),
    mean_net_med=('mean_net','median'),
    n_brand_med=('n_brand','median'),
    import_ratio_mean=('import_ratio','mean'),
    weekend_ratio_mean=('weekend_ratio','mean'),
    refund_ratio_mean=('refund_ratio','mean'),
).round(3).T)

def gender_skew(col, topn=15, min_txn=500):
    """카테고리별 gender==1 비율(거래 가중) — 성별 변별력 높은 값 추출"""
    g = df.groupby(col).agg(n=('gender','size'), g1=('gender','mean'))
    g = g[g['n']>=min_txn].copy()
    g['g1_ratio'] = g['g1'].round(3)
    base = df['gender'].mean()
    g['lift'] = (g['g1']-base).round(3)
    return g.sort_values('g1_ratio')

base_rate = df['gender'].mean()
print(f"\n전체 gender==1 거래비율(기준선): {base_rate:.3f}")

for col in ['team_nm','part_nm','pc_nm']:
    print(f"\n=== 3. [{col}] gender==1 비율 — 낮은쪽(성별0 선호) Top8 / 높은쪽(성별1 선호) Top8 ===")
    g = gender_skew(col, min_txn=1000)
    show = pd.concat([g.head(8), g.tail(8)])
    print(show[['n','g1_ratio','lift']])

print("\n=== 4. 브랜드 Top: 성별 변별력 큰 브랜드 (거래>=800) ===")
gb = gender_skew('brd_nm', min_txn=800)
print("[성별0(남성추정) 선호 브랜드]")
print(gb.head(12)[['n','g1_ratio']])
print("[성별1(여성추정) 선호 브랜드]")
print(gb.tail(12)[['n','g1_ratio']])

print("\n=== 5. 시간대/월별 gender==1 비율 ===")
print("시간(hour):")
print(df.groupby('hour')['gender'].agg(['size','mean']).round(3))
print("월(month):")
print(df.groupby('month')['gender'].agg(['size','mean']).round(3))

print("\nDONE")
