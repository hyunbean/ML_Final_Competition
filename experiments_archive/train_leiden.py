"""Brand co-visitation Leiden 군집 → 고객별 군집 지출비중 → 단독 LGBM ('leiden_lgbm').

같은 고객이 같은 달 산 브랜드끼리 링크 → Leiden 커뮤니티 → 소비군집. 군집별 지출비중만으로
단독 모델 → 수치피처와 직교(위상 맥락) → 스택서 안 죽는 멤버. (Gemini 제안 베이스라인)
실행: pip install python-igraph lightgbm → python -m src.folds → python -m src.train_leiden
"""
import numpy as np
import pandas as pd
import igraph as ig
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from . import config as C
from .oof_io import save_predictions

MODEL_NAME = "leiden_lgbm"
MIN_EDGE = 3          # 동시구매 이만큼 미만 링크는 노이즈 컷
RESOLUTION = 1.0


def _brand_clusters(all_tx):
    co = all_tx[[C.ID_COL, "ym", "brd_nm"]].drop_duplicates()
    e = co.merge(co, on=[C.ID_COL, "ym"])
    e = e[e["brd_nm_x"] < e["brd_nm_y"]]
    w = e.groupby(["brd_nm_x", "brd_nm_y"]).size().reset_index(name="w")
    w = w[w["w"] >= MIN_EDGE]
    brands = pd.Index(sorted(set(w["brd_nm_x"]) | set(w["brd_nm_y"])))
    bid = {b: i for i, b in enumerate(brands)}
    g = ig.Graph(); g.add_vertices(len(brands))
    g.add_edges(list(zip(w["brd_nm_x"].map(bid), w["brd_nm_y"].map(bid))))
    part = g.community_leiden(objective_function="modularity", weights=w["w"].tolist(), resolution=RESOLUTION)
    member = {}
    for cid, nodes in enumerate(part):
        for n in nodes:
            member[brands[n]] = cid
    print(f"브랜드 {len(brands)} → 군집 {len(part)}개")
    return member


def _cluster_share(tx, member, ncl):
    tx = tx.copy()
    tx["cl"] = tx["brd_nm"].map(member)
    tx = tx.dropna(subset=["cl"]); tx["cl"] = tx["cl"].astype(int)
    amt = tx.assign(a=tx["net_amt"].clip(lower=0)).groupby([C.ID_COL, "cl"])["a"].sum().unstack(fill_value=0.0)
    amt = amt.reindex(columns=range(ncl), fill_value=0.0)
    share = amt.div(amt.sum(axis=1).replace(0, 1), axis=0)
    share.columns = [f"leiden_cl_{c}" for c in share.columns]
    return share


def main():
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    y = pd.read_csv(C.YTRAIN_CSV).set_index(C.ID_COL).reindex(train_ids)[C.TARGET].to_numpy()
    use = [C.ID_COL, "sales_datetime", "brd_nm", "net_amt"]
    tr = pd.read_csv(C.TRAIN_CSV, encoding=C.ENCODING, usecols=use)
    te = pd.read_csv(C.TEST_CSV, encoding=C.ENCODING, usecols=use)
    alltx = pd.concat([tr, te], ignore_index=True)
    alltx["ym"] = pd.to_datetime(alltx["sales_datetime"]).dt.to_period("M").astype(str)
    tr["ym"] = ""; te["ym"] = ""

    member = _brand_clusters(alltx)
    ncl = max(member.values()) + 1
    Xtr = _cluster_share(tr.assign(brd_nm=tr["brd_nm"]), member, ncl).reindex(train_ids).fillna(0.0)
    Xte = _cluster_share(te, member, ncl).reindex(test_ids).fillna(0.0)
    print(f"leiden X={Xtr.shape}")

    params = dict(objective="binary", metric="auc", learning_rate=0.03, num_leaves=31,
                  min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                  reg_alpha=1.0, reg_lambda=5.0, n_jobs=-1, random_state=C.SEED, verbose=-1)
    oof = np.full(len(y), np.nan); test_sum = np.zeros(len(test_ids))
    for f in range(C.N_FOLDS):
        tri, va = np.where(folds != f)[0], np.where(folds == f)[0]
        m = lgb.LGBMClassifier(n_estimators=2000, **params)
        m.fit(Xtr.iloc[tri], y[tri], eval_set=[(Xtr.iloc[va], y[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict_proba(Xtr.iloc[va])[:, 1]; test_sum += m.predict_proba(Xte)[:, 1]
        print(f"[fold {f}] AUC={roc_auc_score(y[va], oof[va]):.5f}")
    cv = float(roc_auc_score(y, oof))
    print(f"\n==== {MODEL_NAME}  CV AUC = {cv:.5f} ====")
    save_predictions(MODEL_NAME, oof, test_sum / C.N_FOLDS, meta=dict(
        cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS, feature_set=f"Leiden 브랜드군집({ncl}) 지출비중 LGBM",
        created_by="hyunbean", notes="brand co-visitation Leiden cluster share, 단독 OOF (직교 멤버)"))


if __name__ == "__main__":
    main()
