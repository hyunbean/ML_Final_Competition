"""가족카드 노이즈 피처 + LGBM — '남편카드 쓰는 여성' 같은 혼합구매 탐지.

각 거래의 카테고리 성별 log-odds를 fold-safe로 매겨, 고객별 분포의
표준편차·왜도·첨도·**이봉성(bimodality)**·강한남성/여성 비율을 피처화.
한 고객이 강한남성+강한여성 상품을 둘 다 사면 이봉성↑ = 가족카드 신호.
김민형 dispersion 블록(+0.0008) 재현. 풀피처 위에 추가 → 새 모델 lgbm_fam.

실행: python -m src.folds → python -m src.train_lgbm_fam
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import skew, kurtosis
from sklearn.metrics import roc_auc_score

from . import config as C
from . import data as D
from .features import build_features
from .oof_io import save_predictions

MODEL_NAME = "lgbm_fam"
CAT = "brd_nm"          # 성별신호 강한 카테고리
ALPHA = 20.0
PARAMS = dict(objective="binary", metric="auc", learning_rate=0.02, num_leaves=63,
              feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1,
              min_child_samples=40, n_estimators=3000, n_jobs=-1, verbosity=-1)


def _rate(map_df, gmean):
    """카테고리별 (가중)성별비율 → 스무딩."""
    g = map_df.groupby(CAT)["_g"].agg(["sum", "count"])
    return (g["sum"] + ALPHA * gmean) / (g["count"] + ALPHA)


def _disp_stats(txn, ids):
    """고객별 거래 log-odds 분포 통계."""
    txn = txn.dropna(subset=["_lo"])
    g = txn.groupby(C.ID_COL)["_lo"]
    out = pd.DataFrame({
        "fam_lo_std": g.std(),
        "fam_lo_skew": g.apply(lambda v: skew(v) if len(v) > 2 else 0.0),
        "fam_lo_kurt": g.apply(lambda v: kurtosis(v) if len(v) > 3 else 0.0),
        "fam_lo_range": g.max() - g.min(),
        "fam_male_strong": txn.assign(s=(txn["_lo"] > 0.85).astype(float)).groupby(C.ID_COL)["s"].mean(),
        "fam_female_strong": txn.assign(s=(txn["_lo"] < -0.85).astype(float)).groupby(C.ID_COL)["s"].mean(),
    })
    # 이봉성 계수: 강한남성 & 강한여성 둘 다 있으면 ↑
    out["fam_bimodal"] = out["fam_male_strong"] * out["fam_female_strong"] * 4.0
    return out.reindex(ids).fillna(0.0)


def main():
    X, y, Xtest = build_features()
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    fold_of = pd.Series(folds, index=train_ids)
    gser = pd.Series(y, index=train_ids)
    gmean = float(y.mean())

    tr, te, _ = D.load_raw()
    tr = tr[[C.ID_COL, CAT]].copy(); te = te[[C.ID_COL, CAT]].copy()
    tr[CAT] = tr[CAT].astype(str); te[CAT] = te[CAT].astype(str)
    tr["_g"] = tr[C.ID_COL].map(gser)
    tr["_fold"] = tr[C.ID_COL].map(fold_of)

    # fold-safe: fold f 거래엔 나머지 fold로 만든 rate 부여
    tr["_lo"] = np.nan
    for f in range(C.N_FOLDS):
        rate = _rate(tr[tr["_fold"] != f], gmean)
        lo = np.log(rate / (1 - rate))
        m = tr["_fold"] == f
        tr.loc[m, "_lo"] = tr.loc[m, CAT].map(lo)
    rate_all = _rate(tr, gmean)
    lo_all = np.log(rate_all / (1 - rate_all))
    te["_lo"] = te[CAT].map(lo_all)

    Dtr = _disp_stats(tr, train_ids)
    Dte = _disp_stats(te, test_ids)
    print(f"가족카드 피처: {list(Dtr.columns)}  이봉성 평균={Dtr['fam_bimodal'].mean():.4f}")

    Xa = np.hstack([X.values, Dtr.values]).astype(np.float32)
    Xta = np.hstack([Xtest.values, Dte.values]).astype(np.float32)

    oof = np.full(len(y), np.nan)
    test_sum = np.zeros(len(test_ids))
    for f in range(C.N_FOLDS):
        trn, va = np.where(folds != f)[0], np.where(folds == f)[0]
        m = lgb.LGBMClassifier(**PARAMS)
        m.fit(Xa[trn], y[trn], eval_set=[(Xa[va], y[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict_proba(Xa[va])[:, 1]
        test_sum += m.predict_proba(Xta)[:, 1]
        print(f"[fold {f}] AUC={roc_auc_score(y[va], oof[va]):.5f}")

    cv = float(roc_auc_score(y, oof))
    print(f"\n==== {MODEL_NAME}  CV AUC = {cv:.5f} ====")
    save_predictions(MODEL_NAME, oof, test_sum / C.N_FOLDS, meta=dict(
        cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS, feature_set="full + 가족카드(dispersion/bimodal)",
        created_by="hyunbean", notes=f"family-card noise: per-customer {CAT} log-odds dispersion + bimodality"))


if __name__ == "__main__":
    main()
