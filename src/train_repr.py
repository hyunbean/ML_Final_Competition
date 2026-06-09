"""Representation-first 파이프라인 (GPT 최종 승부수) — 73과 독립된 임베딩 중심 모델.

지금까지는 항상 '강한 521 FE + 임베딩'(73이 흡수). 이건 반대:
  goodcd FastText/W2V representation + customer aggregation + 최소 handcrafted → XGB.
73 없이 처음부터 representation-first. 게이트: corr(73) < 0.97이면 직교신호=희망, 0.99면 천장확정.

실행(GPU+gensim): XGB_GPU=1 python -m src.train_repr
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from . import config as C
from .oof_io import save_predictions
from .train_first import build_emb, build_base, build_goodcd_svd, build_cooc, _load

GPU = os.environ.get("XGB_GPU", "1") == "1"
# 최소 handcrafted만 (representation을 죽이지 않게 소수)
MINI = ["base_freq", "base_recency", "base_tot_amt_sum", "base_tot_amt_mean",
        "base_ticket_avg", "base_weekend_ratio", "base_disc_rate_mean"]


def main():
    tr_id = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    te_id = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    _, _, y_df, full = _load()
    print("repr: build_emb(W2V+FastText) + goodcd_svd + cooc + 최소 handcrafted ...")
    emb = build_emb(full)                                   # representation 핵심
    svd = build_goodcd_svd(full)                            # goodcd SVD representation
    coo = build_cooc(full)                                  # PPMI representation (transductive)
    base = build_base(full)
    mini = base[[c for c in MINI if c in base.columns]]
    allf = emb.join(svd, how="outer").join(coo, how="outer").join(mini, how="left")
    allf.index.name = "custid"; allf = allf.fillna(0.0)
    import re
    allf.columns = [re.sub(r"[^0-9a-zA-Z가-힣_]", "_", str(c)) for c in allf.columns]
    X = allf.reindex(tr_id).fillna(0.0); Xt = allf.reindex(te_id).fillna(0.0)
    y = y_df.set_index("custid").reindex(tr_id)["gender"].to_numpy()
    print(f"repr-first X={X.shape} (emb+svd+cooc {X.shape[1]-len(mini.columns)} + mini {len(mini.columns)})")

    import xgboost as xgb
    oof = np.full(len(y), np.nan); test_sum = np.zeros(len(te_id))
    for f in range(C.N_FOLDS):
        tri, va = np.where(folds != f)[0], np.where(folds == f)[0]
        m = xgb.XGBClassifier(n_estimators=4000, early_stopping_rounds=150, objective="binary:logistic",
                              eval_metric="auc", learning_rate=0.02, max_depth=6, min_child_weight=5, gamma=0.1,
                              subsample=0.8, colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=5.0,
                              random_state=C.SEED, tree_method="hist", device="cuda" if GPU else "cpu")
        m.fit(X.iloc[tri], y[tri], eval_set=[(X.iloc[va], y[va])], verbose=False)
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]; test_sum += m.predict_proba(Xt)[:, 1]
        print(f"  [fold {f}] AUC={roc_auc_score(y[va], oof[va]):.5f}")
    cv = float(roc_auc_score(y, oof)); name = "repr_xgb"
    print(f"==== {name}  CV={cv:.5f} ====")
    # 게이트: corr(73)
    def rn(a): return rankdata(a) / len(a)
    for m in ["kim73", "first_xgb_pl2", "mh_bestblend69"]:
        p = f"artifacts/oof/{m}__oof.npy"
        if os.path.exists(p):
            c = np.corrcoef(rn(oof), rn(np.load(p)))[0, 1]
            flag = "  <-- 직교! 희망" if (m == "kim73" and c < 0.97) else ("  <<< 흡수=천장" if m == "kim73" else "")
            print(f"  corr(repr, {m})={c:.4f}{flag}")
    save_predictions(name, oof, test_sum / C.N_FOLDS, meta=dict(cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS,
                     feature_set="representation-first (emb+svd+cooc + 최소 handcrafted)", created_by="hyunbean",
                     notes="GPT 최종 승부수: 73독립 representation-first. corr(73)<0.97이면 직교신호"))


if __name__ == "__main__":
    main()
