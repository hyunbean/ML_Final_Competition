"""household/conflict 진단 (GPT 요청): BASE vs +conflict CV, importance rank, 공통오답군 AUC.

build_all(현재 household 제외)로 BASE, 거기에 build_household 조인해 +conflict 버전 비교.
실행(GPU): pip install xgboost gensim catboost → python -m src.diag_household
"""
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from . import config as C
from .train_first import build_all, build_household, _load

PARAMS = dict(objective="binary:logistic", eval_metric="auc", learning_rate=0.02, max_depth=7,
              min_child_weight=5, gamma=0.1, subsample=0.8, colsample_bytree=0.7,
              reg_alpha=1.0, reg_lambda=5.0, random_state=C.SEED, tree_method="hist", device="cuda")


def xgb_oof(X, y, folds):
    oof = np.full(len(y), np.nan); imp = np.zeros(X.shape[1])
    for f in range(C.N_FOLDS):
        tri, va = np.where(folds != f)[0], np.where(folds == f)[0]
        m = xgb.XGBClassifier(n_estimators=4000, early_stopping_rounds=150, **PARAMS)
        m.fit(X.iloc[tri], y[tri], eval_set=[(X.iloc[va], y[va])], verbose=False)
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        imp += m.feature_importances_
    return oof, imp / C.N_FOLDS


def main():
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    allf, ydf, _, _ = build_all()
    _, _, _, full = _load()
    y = ydf.set_index("custid").reindex(train_ids)["gender"].to_numpy()

    Xbase = allf.reindex(train_ids).fillna(0.0).reset_index(drop=True)
    hh = build_household(full)
    Xaug = allf.join(hh, how="left").reindex(train_ids).fillna(0.0).reset_index(drop=True)
    print(f"BASE 피처수={Xbase.shape[1]}, +household 피처수={Xaug.shape[1]} (+{Xaug.shape[1]-Xbase.shape[1]})")

    oof_b, _ = xgb_oof(Xbase, y, folds)
    oof_a, imp_a = xgb_oof(Xaug, y, folds)
    cvb, cva = roc_auc_score(y, oof_b), roc_auc_score(y, oof_a)
    print("\n========== CV (xgb, 동일 파라미터) ==========")
    print(f"BASE        CV = {cvb:.5f}")
    print(f"+conflict   CV = {cva:.5f}   (delta {cva-cvb:+.5f})")

    # importance rank
    cols = list(Xaug.columns)
    order = [cols[j] for j in np.argsort(imp_a)[::-1]]
    rank = {c: i + 1 for i, c in enumerate(order)}
    n = len(cols)
    print(f"\n========== household 피처 importance rank (/{n}) ==========")
    for f in ["hh_conflict", "hh_proxy_male2", "hh_proxy_male", "hh_homemaker",
              "hh_male_sig", "hh_female_sig", "hh_cos_vs_male", "hh_family_breadth"]:
        if f in rank:
            print(f"  {f:18s} rank {rank[f]:3d}/{n}   imp={imp_a[cols.index(f)]:.4f}")

    # 공통오답군 (base 예측이 틀린 고객)
    wrong = ((oof_b >= 0.5).astype(int) != y)
    print(f"\n========== 공통오답군 AUC ({wrong.sum()}명, {wrong.mean()*100:.1f}%) ==========")
    print(f"  오답군내 AUC  BASE      = {roc_auc_score(y[wrong], oof_b[wrong]):.5f}")
    print(f"  오답군내 AUC  +conflict = {roc_auc_score(y[wrong], oof_a[wrong]):.5f}")
    # 정답군도 (conflict가 정답군을 망쳤나)
    ok = ~wrong
    print(f"  정답군내 AUC  BASE      = {roc_auc_score(y[ok], oof_b[ok]):.5f}")
    print(f"  정답군내 AUC  +conflict = {roc_auc_score(y[ok], oof_a[ok]):.5f}")


if __name__ == "__main__":
    main()
