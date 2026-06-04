"""데이터 로드(CP949) + 거래→고객(custid) 베이스라인 집계 피처.

본격 파생변수/타깃인코딩/임베딩은 features.py(추후)에서. 여기는 '돌아가는 최소셋'.
build_xy()는 정규 custid 순서(folds.py 출력)에 '정렬된' X_train, y, X_test를 돌려줌
→ 이렇게 해야 모든 모델의 OOF/test npy 행 순서가 팀 전체에서 일치함.
"""
import numpy as np
import pandas as pd
from . import config as C


def load_raw():
    tr = pd.read_csv(C.TRAIN_CSV, encoding=C.ENCODING)
    te = pd.read_csv(C.TEST_CSV, encoding=C.ENCODING)
    y = pd.read_csv(C.YTRAIN_CSV)
    return tr, te, y


def make_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """거래 단위 → 고객 단위 기본 집계 피처. index = custid."""
    df = df.copy()
    df["sales_datetime"] = pd.to_datetime(df["sales_datetime"])
    df["hour"] = df["sales_datetime"].dt.hour
    df["weekday"] = df["sales_datetime"].dt.weekday
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
    df["day"] = df["sales_datetime"].dt.normalize()
    df["is_refund"] = (df["net_amt"] < 0).astype(int)
    df["is_inst"] = (df["inst_mon"] > 1).astype(int)
    df["is_evening"] = (df["hour"] >= 18).astype(int)   # EDA: 저녁=g0 성향
    df["is_morning"] = (df["hour"] <= 11).astype(int)   # EDA: 오전=g1 성향

    g = df.groupby(C.ID_COL)
    feat = g.agg(
        txn_cnt=("goodcd", "size"),
        n_goods=("goodcd", "nunique"),
        n_brand=("brd_nm", "nunique"),
        n_corner=("corner_nm", "nunique"),
        n_pc=("pc_nm", "nunique"),
        n_part=("part_nm", "nunique"),
        n_team=("team_nm", "nunique"),
        n_store=("str_nm", "nunique"),
        visit_days=("day", "nunique"),
        net_sum=("net_amt", "sum"),
        net_mean=("net_amt", "mean"),
        net_max=("net_amt", "max"),
        net_min=("net_amt", "min"),
        net_std=("net_amt", "std"),
        net_median=("net_amt", "median"),
        tot_sum=("tot_amt", "sum"),
        dis_sum=("dis_amt", "sum"),
        refund_ratio=("is_refund", "mean"),
        import_ratio=("import_flg", "mean"),
        inst_ratio=("is_inst", "mean"),
        inst_mon_mean=("inst_mon", "mean"),
        weekend_ratio=("is_weekend", "mean"),
        evening_ratio=("is_evening", "mean"),
        morning_ratio=("is_morning", "mean"),
        hour_mean=("hour", "mean"),
        hour_std=("hour", "std"),
    )
    feat["goods_div"] = feat["n_goods"] / feat["txn_cnt"]
    feat["brand_div"] = feat["n_brand"] / feat["txn_cnt"]
    feat["txn_per_visit"] = feat["txn_cnt"] / feat["visit_days"]
    feat["amt_per_visit"] = feat["net_sum"] / feat["visit_days"]
    feat["discount_rate"] = feat["dis_sum"] / (feat["tot_sum"].abs() + 1)
    return feat


def build_xy():
    """정규 순서에 정렬된 (X_train, y, X_test) 반환."""
    tr, te, y = load_raw()
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)

    ftr = make_baseline_features(tr).reindex(train_ids).fillna(0)
    fte = make_baseline_features(te).reindex(test_ids).fillna(0)

    y = y.set_index(C.ID_COL).reindex(train_ids)[C.TARGET].to_numpy()
    return ftr.reset_index(drop=True), y, fte.reset_index(drop=True)
