"""Co-training (GPT#4): xgb↔lgbm이 서로의 고신뢰 test를 교환 pseudo.

기존 pseudo는 teacher=외부 강한모델(mega). co-training은 teacher=상대 GBDT:
  xgb가 test에서 고신뢰로 뽑은 라벨 → lgbm student의 train에 추가 (그 반대도).
conditional independence가 있으면 이득. 단 우리 xgb/lgbm corr 0.99라 GPT는 기대 낮게 봄.
val fold 순수(OOF 정직): test pseudo만 추가, val은 학습에 안 들어감.

실행(GPU): CT_HI=0.9 CT_LO=0.1 XGB_GPU=1 python -m src.train_cotrain
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from . import config as C
from .oof_io import save_predictions
from .train_first import build_all
from .train_pseudo_strict import _fit

HI = float(os.environ.get("CT_HI", "0.90"))
LO = float(os.environ.get("CT_LO", "0.10"))
ROUNDS = int(os.environ.get("CT_ROUNDS", "1"))


def main():
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    allf, ydf, _, _ = build_all()
    X = allf.reindex(train_ids).fillna(0.0); Xt = allf.reindex(test_ids).fillna(0.0)
    y = ydf.set_index("custid").reindex(train_ids)["gender"].to_numpy()
    print(f"[cotrain] X={X.shape} HI/LO={HI}/{LO} rounds={ROUNDS}")

    oof = np.full(len(y), np.nan); test_sum = np.zeros(len(test_ids))
    for f in range(C.N_FOLDS):
        tri, va = np.where(folds != f)[0], np.where(folds == f)[0]
        Xtr, ytr = X.iloc[tri], y[tri]
        # round 0: base 각자 학습 → test 예측
        vx, tx = _fit("xgb", Xtr, ytr, X.iloc[va], y[va], Xt)
        vl, tl = _fit("lgbm", Xtr, ytr, X.iloc[va], y[va], Xt)
        for r in range(ROUNDS):
            cx = (tx >= HI) | (tx <= LO); plx = (tx[cx] >= 0.5).astype(int)   # xgb의 고신뢰 → lgbm에
            cl = (tl >= HI) | (tl <= LO); pll = (tl[cl] >= 0.5).astype(int)   # lgbm의 고신뢰 → xgb에
            Xl = pd.concat([Xtr, Xt.iloc[cx]]); yl = np.r_[ytr, plx]          # lgbm: train + xgb pseudo
            Xx = pd.concat([Xtr, Xt.iloc[cl]]); yx = np.r_[ytr, pll]          # xgb: train + lgbm pseudo
            vx, tx = _fit("xgb", Xx, yx, X.iloc[va], y[va], Xt)
            vl, tl = _fit("lgbm", Xl, yl, X.iloc[va], y[va], Xt)
            if f == 0:
                print(f"  [fold0 round{r}] xgb pseudo {cx.sum()} / lgbm pseudo {cl.sum()}")
        oof[va] = (vx + vl) / 2; test_sum += (tx + tl) / 2
        print(f"  [fold {f}] AUC={roc_auc_score(y[va], oof[va]):.5f}")
    cv = float(roc_auc_score(y, oof)); name = "cotrain_xl"
    print(f"==== {name}  CV={cv:.5f} ====")
    for m in ["first_xgb_pl2", "first_lgbm_pl2", "mh_bestblend69"]:
        p = f"artifacts/oof/{m}__oof.npy"
        if os.path.exists(p):
            print(f"  corr(cotrain, {m})={np.corrcoef(rankdata(oof), rankdata(np.load(p)))[0,1]:.4f}")
    save_predictions(name, oof, test_sum / C.N_FOLDS, meta=dict(cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS,
                     feature_set="co-training xgb<->lgbm 교환pseudo", created_by="hyunbean",
                     notes=f"co-training(GPT#4) HI/LO={HI}/{LO} rounds={ROUNDS}, val fold 순수"))


if __name__ == "__main__":
    main()
