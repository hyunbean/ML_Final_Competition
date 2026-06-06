"""txn 멀티집계 — 거래→방문(영수증)→고객 다단계 집계 (가족/동반쇼핑 신호).

리서치 P5: 1단계(거래→고객)만 하면 놓치는 '한 방문에서 여러 코너 도는 패턴'(=동반/가족쇼핑)을
방문(custid×날짜) 중간레벨로 집계 → 고객 통계. 우리 피처에 덧붙여 LGBM. 성별 신호 강함.
실행: python -m src.folds → python -m src.train_multiagg
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from . import config as C
from . import data as D
from .features import build_features
from .oof_io import save_predictions

MODEL_NAME = "multiagg_lgbm"
PARAMS = dict(objective="binary", metric="auc", learning_rate=0.02, num_leaves=64,
              min_child_samples=80, feature_fraction=0.6, bagging_fraction=0.8,
              bagging_freq=1, n_estimators=3000, n_jobs=4, verbosity=-1)


def _visit_features(df, ids):
    d = df[[C.ID_COL, "sales_datetime", "corner_nm", "net_amt"]].copy()
    d["day"] = pd.to_datetime(d["sales_datetime"]).dt.normalize()
    # 방문(custid×날짜) 단위 집계
    v = d.groupby([C.ID_COL, "day"]).agg(
        v_goods=("net_amt", "size"),
        v_corner=("corner_nm", "nunique"),
        v_spend=("net_amt", "sum"),
    ).reset_index()
    g = v.groupby(C.ID_COL)
    out = pd.DataFrame({
        "n_visits": g.size(),
        "visit_goods_mean": g["v_goods"].mean(), "visit_goods_max": g["v_goods"].max(),
        "visit_goods_std": g["v_goods"].std(),
        "visit_corner_mean": g["v_corner"].mean(),   # 방문당 코너수 (동반쇼핑↑)
        "visit_corner_max": g["v_corner"].max(),
        "visit_spend_mean": g["v_spend"].mean(), "visit_spend_max": g["v_spend"].max(),
        "visit_spend_std": g["v_spend"].std(),
        "multi_corner_visit_ratio": v.assign(mc=(v["v_corner"] >= 3).astype(float)).groupby(C.ID_COL)["mc"].mean(),
        "big_basket_ratio": v.assign(b=(v["v_goods"] >= 5).astype(float)).groupby(C.ID_COL)["b"].mean(),
        "goods_per_visit": d.groupby(C.ID_COL).size() / g.size(),
    })
    return out.reindex(ids).fillna(0.0)


def main():
    X, y, Xtest = build_features()
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    tr, te, _ = D.load_raw()

    Dtr = _visit_features(tr, train_ids); Dte = _visit_features(te, test_ids)
    print(f"방문단위 멀티집계 {Dtr.shape[1]}개: {list(Dtr.columns)}")

    Xa = np.hstack([np.nan_to_num(X.values, posinf=0, neginf=0), Dtr.values]).astype(np.float32)
    Xta = np.hstack([np.nan_to_num(Xtest.values, posinf=0, neginf=0), Dte.values]).astype(np.float32)
    oof = np.full(len(y), np.nan); test_sum = np.zeros(len(test_ids))
    for f in range(C.N_FOLDS):
        trn, va = np.where(folds != f)[0], np.where(folds == f)[0]
        m = lgb.LGBMClassifier(**PARAMS)
        m.fit(Xa[trn], y[trn], eval_set=[(Xa[va], y[va])], callbacks=[lgb.early_stopping(120, verbose=False)])
        oof[va] = m.predict_proba(Xa[va])[:, 1]; test_sum += m.predict_proba(Xta)[:, 1]
        print(f"[fold {f}] AUC={roc_auc_score(y[va], oof[va]):.5f}")
    cv = float(roc_auc_score(y, oof))
    print(f"\n==== {MODEL_NAME}  CV AUC = {cv:.5f} ====")
    save_predictions(MODEL_NAME, oof, test_sum / C.N_FOLDS, meta=dict(
        cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS, feature_set="our + 방문단위 멀티집계",
        created_by="hyunbean", notes="multi-level (visit/receipt) aggregation features"))


if __name__ == "__main__":
    main()
