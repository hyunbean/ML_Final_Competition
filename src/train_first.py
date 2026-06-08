"""작년 1등 솔루션 피처셋 + 튜닝 CatBoost → 5-fold OOF 베이스모델 'first_cat'.

1등 노트북(2025 ML Pipeline Hackathon)의 FE를 그대로 포팅(우리 sales_datetime→date/time 파생).
우리 기존 피처와 구성이 달라(dwell/buyer/social/recent-K/수작업 성별파트) 결이 다른 멤버 기대.
TE는 내부 KFold로 leak-safe, 모델 OOF는 우리 folds.npy 사용.
실행: pip install catboost → python -m src.folds → python -m src.train_first
"""
import os
import re
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

from . import config as C
from .oof_io import save_predictions

MODEL_NAME = "first_cat"
np.random.seed(C.SEED)


def _load():
    tr = pd.read_csv(C.TRAIN_CSV, encoding=C.ENCODING)
    te = pd.read_csv(C.TEST_CSV, encoding=C.ENCODING)
    y = pd.read_csv(C.YTRAIN_CSV, encoding=C.ENCODING)
    for df in (tr, te):
        dt = pd.to_datetime(df["sales_datetime"])
        df["sales_date"] = dt.dt.normalize()
        df["sales_time"] = (dt.dt.hour * 100 + dt.dt.minute).astype(int)
    tr["dataset"] = "train"; te["dataset"] = "test"
    full = pd.concat([tr, te], ignore_index=True)
    for c in ["tot_amt", "dis_amt", "net_amt", "inst_mon", "sales_time"]:
        if c not in full.columns:
            full[c] = 0
    for c in ["brd_nm", "corner_nm", "part_nm", "buyer_nm", "str_nm"]:
        if c not in full.columns:
            full[c] = ""
    return tr, te, y, full


# ---------- 2. 쇼핑 스타일 ----------
def build_style(df):
    d = df.copy(); d["corner_nm"] = d["corner_nm"].fillna("").astype(str)
    lux = ["수입", "명품", "부띠끄", "부틱", "디자이너"]
    bar = ["행사", "균일가", "대전", "기획", "특가", "이월"]
    d["st_luxury_flag"] = d["corner_nm"].apply(lambda x: int(any(k in x for k in lux)))
    d["st_bargain_flag"] = d["corner_nm"].apply(lambda x: int(any(k in x for k in bar)))
    d["st_disc_ratio"] = d["dis_amt"] / (d["tot_amt"] + 1e-5)
    g = d.groupby("custid")
    agg = g[["st_luxury_flag", "st_bargain_flag"]].mean()
    agg.columns = ["st_ratio_luxury", "st_ratio_bargain"]
    agg["st_disc_std"] = g["st_disc_ratio"].std().fillna(0)
    agg["st_disc_mean"] = g["st_disc_ratio"].mean().fillna(0)
    return agg


# ---------- 3. 행동 패턴 ----------
def build_behavior(df):
    d = df.copy()
    base = np.where(d["net_amt"].fillna(0) != 0, d["net_amt"].fillna(0), d["tot_amt"].fillna(0))
    d["bh_refund_flag"] = (base < 0).astype(int)
    refund = d.groupby("custid")["bh_refund_flag"].agg(["sum", "mean"])
    refund.columns = ["bh_refund_cnt", "bh_refund_rate"]
    d["bh_inst_flag"] = (d["inst_mon"] > 0).astype(int)
    inst_rate = d.groupby("custid")["bh_inst_flag"].mean().to_frame("bh_inst_rate")
    inst_max = d.groupby("custid")["inst_mon"].max().to_frame("bh_inst_max")
    div = d.groupby("custid")[["brd_nm", "corner_nm", "part_nm"]].nunique()
    div.columns = ["bh_nunique_brd", "bh_nunique_corner", "bh_nunique_part"]
    amt = d.groupby("custid")["tot_amt"].agg(["sum", "count"])
    ticket = (amt["sum"] / (amt["count"] + 1e-5)).to_frame("bh_ticket_size")
    d["bh_high_amt_flag"] = (d["tot_amt"] > 300000).astype(int)
    hi = d.groupby("custid")["bh_high_amt_flag"].sum().to_frame("bh_high_amt_cnt")
    return pd.concat([refund, inst_rate, inst_max, div, ticket, hi], axis=1)


_REPLACE = {"가정용품파트": "가정용품", "공산품파트": "공산품", "생식품파트": "생식품",
            "잡화파트": "잡화", "로얄부틱": "로얄부띠끄", "스포츠캐쥬얼": "스포츠캐주얼",
            "여성캐쥬얼": "여성캐주얼"}
_MALE_PARTS = ["가정용품", "공산품", "생식품", "케주얼,구두,아동", "남성정장", "남성캐주얼"]
_FEMALE_PARTS = ["여성캐주얼", "영캐릭터", "영플라자", "패션잡화", "여성정장"]


# ---------- 4. 시간/요일/시즌/파트·바이어 ----------
def build_calendar(df):
    d = df.copy(); d["sales_time"] = d["sales_time"].fillna(0).astype(int)
    visit = d.groupby("custid")["sales_date"].nunique().to_frame("cal_visit_days")

    def wd(x):
        w = x.dayofweek
        return "WD_MonTueWed" if w <= 2 else ("WD_ThuFri" if w <= 4 else "WD_Weekend")

    def ss(x):
        m = x.month
        return "SS_Spring" if 2 <= m <= 4 else ("SS_Summer" if 5 <= m <= 7 else ("SS_Fall" if 8 <= m <= 10 else "SS_Winter"))

    def tm(t):
        return ("TM_Morning" if t < 1200 else "TM_Lunch" if t < 1400 else
                "TM_Afternoon1" if t < 1600 else "TM_Afternoon2" if t < 1800 else "TM_Evening")
    d["cal_wd_group"] = d["sales_date"].apply(wd)
    d["cal_season_group"] = d["sales_date"].apply(ss)
    d["cal_time_group"] = d["sales_time"].apply(tm)
    wd_pv = d.pivot_table(index="custid", columns="cal_wd_group", values="tot_amt", aggfunc="size", fill_value=0).add_prefix("cal_wd_")
    ss_pv = d.pivot_table(index="custid", columns="cal_season_group", values="tot_amt", aggfunc="size", fill_value=0).add_prefix("cal_season_")
    tm_pv = d.pivot_table(index="custid", columns="cal_time_group", values="tot_amt", aggfunc="size", fill_value=0).add_prefix("cal_time_")
    d["part_norm"] = d["part_nm"].fillna("").astype(str).replace(_REPLACE)
    d["cal_male_part_flag"] = d["part_norm"].isin(_MALE_PARTS).astype(int)
    d["cal_female_part_flag"] = d["part_norm"].isin(_FEMALE_PARTS).astype(int)
    pc = d.groupby("custid")[["cal_male_part_flag", "cal_female_part_flag"]].sum()
    pc.columns = ["cal_male_part_cnt", "cal_female_part_cnt"]
    tot = pc["cal_male_part_cnt"] + pc["cal_female_part_cnt"]
    pc["cal_male_ratio"] = pc["cal_male_part_cnt"] / (tot + 1e-5)
    pc["cal_female_ratio"] = pc["cal_female_part_cnt"] / (tot + 1e-5)
    buyer_pv = d.pivot_table(index="custid", columns="buyer_nm", values="tot_amt", aggfunc="size", fill_value=0).add_prefix("cal_buyer_")
    return pd.concat([visit, wd_pv, ss_pv, tm_pv, pc, buyer_pv], axis=1)


# ---------- 5. 관심 키워드 비중 ----------
def build_interest(df):
    d = df.copy()
    d["full_txt"] = d["brd_nm"].fillna("") + " " + d["corner_nm"].fillna("") + " " + d["part_nm"].fillna("")
    keys = ["남성", "여성", "아동", "학생", "골프", "스포츠", "영", "마담", "캐주얼", "정장", "화장품"]
    tot = d.groupby("custid")["tot_amt"].sum()
    out = pd.DataFrame(index=tot.index)
    for kw in keys:
        m = d["full_txt"].str.contains(kw, na=False)
        s = d[m].groupby("custid")["tot_amt"].sum()
        out[f"int_ratio_{kw}"] = (s / (tot + 1e-5)).reindex(tot.index).fillna(0)
    return out


# ---------- 6. Entropy ----------
def build_entropy(df):
    d = df.copy()

    def ent(col):
        cnt = d.groupby(["custid", col]).size().unstack(fill_value=0)
        prob = cnt.div(cnt.sum(axis=1), axis=0)
        return prob.apply(lambda x: entropy(x), axis=1)
    return pd.concat([ent("brd_nm").to_frame("ent_brd"), ent("corner_nm").to_frame("ent_corner")], axis=1)


# ---------- 7. Target Encoding (gender) ----------
def _kfold_te(train_tx, y_df, col, n=5, noise=0.01):
    tx = train_tx.merge(y_df, on="custid", how="left")
    scores = pd.DataFrame(index=tx["custid"].unique())
    nc = f"te_{col}_gender"; scores[nc] = 0.0
    base = y_df[y_df["custid"].isin(tx["custid"])]
    kf = StratifiedKFold(n_splits=n, shuffle=True, random_state=42)
    for tri, vai in kf.split(base, base["gender"]):
        tri_ids = base.iloc[tri]["custid"]; vai_ids = base.iloc[vai]["custid"]
        t = tx[tx["custid"].isin(tri_ids)].copy(); t[col] = t[col].fillna("UNKNOWN")
        st = t.groupby(col)["gender"].agg(["mean", "count"]); gm = t["gender"].mean()
        st["te_value"] = np.where(st["count"] < 10, gm, st["mean"])
        v = tx[tx["custid"].isin(vai_ids)].copy(); v[col] = v[col].fillna("UNKNOWN")
        v = v.merge(st[["te_value"]], on=col, how="left")
        v["te_value"] = v["te_value"].fillna(gm) + np.random.normal(0, noise, len(v))
        cm = v.groupby("custid")["te_value"].mean()
        scores.loc[cm.index, nc] = cm
    return scores


def _te_test(train_tx, test_tx, y_df, col):
    tx = train_tx.merge(y_df, on="custid", how="left"); tx[col] = tx[col].fillna("UNKNOWN")
    st = tx.groupby(col)["gender"].agg(["mean", "count"]); gm = tx["gender"].mean()
    st["te_value"] = np.where(st["count"] < 10, gm, st["mean"])
    t = test_tx.copy(); t[col] = t[col].fillna("UNKNOWN")
    t = t.merge(st[["te_value"]], on=col, how="left"); t["te_value"] = t["te_value"].fillna(gm)
    return t.groupby("custid")["te_value"].mean().to_frame(f"te_{col}_gender")


# ---------- 8. RFM/기본 ----------
def build_base(df):
    d = df.copy()
    d["base_hour"] = (d["sales_time"].fillna(0).astype(int) // 100)
    ref = d["sales_date"].max()
    rec = d.groupby("custid")["sales_date"].apply(lambda x: (ref - x.max()).days).to_frame("base_recency")
    agg = d.groupby("custid").agg({"tot_amt": ["sum", "mean", "max", "min", "std"],
                                   "dis_amt": ["sum", "mean"], "inst_mon": ["mean", "max"],
                                   "sales_date": ["count"], "base_hour": ["mean", "std"]})
    agg.columns = ["base_" + "_".join(c).strip() for c in agg.columns]
    agg = agg.rename(columns={"base_sales_date_count": "base_freq"})
    agg["base_ticket_avg"] = agg["base_tot_amt_sum"] / (agg["base_freq"] + 1e-5)
    agg["base_disc_rate_sum"] = agg["base_dis_amt_sum"] / (agg["base_tot_amt_sum"] + 1e-5)
    agg["base_disc_rate_mean"] = (agg["base_dis_amt_mean"] / (agg["base_tot_amt_mean"] + 1e-5)) * 100
    store = d.pivot_table(index="custid", columns="str_nm", values="tot_amt", aggfunc="size", fill_value=0).add_prefix("base_str_cnt_")
    d["wk"] = d["sales_date"].dt.dayofweek >= 5
    wke = d.groupby("custid")["wk"].mean().to_frame("base_weekend_ratio")
    return pd.concat([agg, rec, store, wke], axis=1).fillna(0)


# ---------- 9. TF-IDF + SVD ----------
def build_tfidf(df):
    d = df.copy(); d["brd_nm"] = d["brd_nm"].astype(str).fillna("UNKNOWN"); d["corner_nm"] = d["corner_nm"].astype(str).fillna("UNKNOWN")
    brd = d.groupby("custid")["brd_nm"].apply(lambda x: " ".join(x))
    cor = d.groupby("custid")["corner_nm"].apply(lambda x: " ".join(x))
    bm = TfidfVectorizer(max_features=1000).fit_transform(brd)
    cm = TfidfVectorizer(max_features=500).fit_transform(cor)
    bs = pd.DataFrame(TruncatedSVD(10, random_state=42).fit_transform(bm), index=brd.index).add_prefix("svd_brd_")
    cs = pd.DataFrame(TruncatedSVD(10, random_state=42).fit_transform(cm), index=cor.index).add_prefix("svd_corner_")
    bt = pd.DataFrame(TfidfVectorizer(max_features=200).fit_transform(brd).toarray(), index=brd.index).add_prefix("tfidf_brd_")
    ct = pd.DataFrame(TfidfVectorizer(max_features=100).fit_transform(cor).toarray(), index=cor.index).add_prefix("tfidf_corner_")
    return bt, ct, bs, cs


def build_goodcd_svd(df):
    """정크코드(용기보증/미확인 22.6%) 제거한 goodcd TF-IDF→SVD. 11k 상품의 finest 신호."""
    d = df.copy(); d["goodcd"] = d["goodcd"].astype(str)
    junk = d["goodcd"].eq("2700000000000") | d["pc_nm"].astype(str).eq("미확인pc") | d["corner_nm"].astype(str).eq("용기보증")
    d = d[~junk]
    doc = d.groupby("custid")["goodcd"].apply(lambda x: " ".join(x))
    gm = TfidfVectorizer(max_features=2000, min_df=5).fit_transform(doc)
    gs = pd.DataFrame(TruncatedSVD(24, random_state=42).fit_transform(gm), index=doc.index).add_prefix("svd_goodcd_")
    return gs


def build_cooc(df, dim=24, mincust=10):
    """train+test goodcd 동시출현 PPMI→SVD→고객임베딩 (test-aware co-occurrence, GPT조언 #2).
    W2V/SVD/cluster와 달리 item-item PMI 그래프 = 다른 구성. 전체데이터(test포함) 사용=transductive."""
    from scipy.sparse import csr_matrix
    d = df.copy(); d["goodcd"] = d["goodcd"].astype(str)
    junk = d["goodcd"].eq("2700000000000") | d["pc_nm"].astype(str).eq("미확인pc") | d["corner_nm"].astype(str).eq("용기보증")
    d = d[~junk]
    vc = d["goodcd"].value_counts(); keep = set(vc[vc >= mincust].index)   # 희귀 goodcd 제거(메모리/노이즈)
    d = d[d["goodcd"].isin(keep)]
    cust = d["custid"].astype("category"); good = d["goodcd"].astype("category")
    M = csr_matrix((np.ones(len(d), np.float32), (cust.cat.codes.values, good.cat.codes.values)),
                   shape=(len(cust.cat.categories), len(good.cat.categories)))
    M.sum_duplicates(); M.data[:] = 1.0                                    # 고객별 goodcd 존재(binary)
    C = (M.T @ M).tocoo().astype(np.float64)                               # goodcd×goodcd 동시출현
    N = C.data.sum(); s = np.asarray(csr_matrix((C.data, (C.row, C.col)), shape=C.shape).sum(1)).ravel()
    pmi = np.log((C.data * N) / (s[C.row] * s[C.col] + 1e-9) + 1e-12); pmi[pmi < 0] = 0.0   # PPMI
    P = csr_matrix((pmi, (C.row, C.col)), shape=C.shape)
    V = TruncatedSVD(dim, random_state=42).fit_transform(P)                # goodcd 임베딩
    cnt = np.asarray(M.sum(1)).ravel() + 1e-9
    cv = (M @ V) / cnt[:, None]                                            # 고객=goodcd벡터 평균
    R = pd.DataFrame(cv, index=cust.cat.categories).add_prefix("cooc_"); R.index.name = "custid"
    return R


def build_emb(df, w2v_dim=32, ft_dim=48):
    """goodcd 시퀀스 W2V + FastText → 고객 임베딩 (교수힌트 #1 W2V +0.00033, #3 FastText +0.00089).
    GPT조언: TF-IDF가중mean + max + std 집계(mean단독은 공통상품에 끌림). FastText subword=접두사계층."""
    from gensim.models import Word2Vec, FastText
    d = df.copy(); d["goodcd"] = d["goodcd"].astype(str)
    junk = d["goodcd"].eq("2700000000000") | d["pc_nm"].astype(str).eq("미확인pc") | d["corner_nm"].astype(str).eq("용기보증")
    d = d[~junk].sort_values(["custid", "sales_datetime"])
    seqs = d.groupby("custid")["goodcd"].apply(list)
    N = len(seqs); dfreq = d.groupby("goodcd")["custid"].nunique()
    idf = np.log(N / (dfreq + 1.0)).to_dict()
    blocks = []
    goodcd_w2v_wv = None
    for Model, pref, dim in [(Word2Vec, "w2v", w2v_dim), (FastText, "ft", ft_dim)]:
        m = Model(sents := seqs.tolist(), vector_size=dim, window=5, min_count=3, workers=4, sg=1, epochs=10, seed=42)
        wv = m.wv
        if pref == "w2v":
            goodcd_w2v_wv = wv
        wm = np.zeros((N, dim)); mx = np.zeros((N, dim)); sd = np.zeros((N, dim))
        for i, s in enumerate(seqs.values):
            vs = np.array([wv[g] for g in s if g in wv], dtype=np.float32)
            if len(vs) == 0:
                continue
            w = np.array([idf.get(g, 1.0) for g in s if g in wv], dtype=np.float32)
            wm[i] = (vs * w[:, None]).sum(0) / (w.sum() + 1e-9)         # TF-IDF(IDF)가중 평균
            mx[i] = vs.max(0); sd[i] = vs.std(0)
        for tag, arr in [("wm", wm), ("mx", mx), ("sd", sd)]:
            blocks.append(pd.DataFrame(arr, index=seqs.index).add_prefix(f"{pref}{tag}_gc_"))
    # #1 Doc2Vec: 고객을 직접 임베딩 (집계 불필요)
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    docs = [TaggedDocument(s, [str(c)]) for c, s in seqs.items()]
    dm = Doc2Vec(docs, vector_size=24, window=5, min_count=3, workers=4, epochs=15, seed=42)
    d2v = np.array([dm.dv[str(c)] for c in seqs.index])
    blocks.append(pd.DataFrame(d2v, index=seqs.index).add_prefix("d2v_gc_"))
    # #4 brand/corner 임베딩 (W2V mean, 작은 dim)
    for col, cdim in [("brd_nm", 16), ("corner_nm", 16)]:
        cs = df.sort_values([C.ID_COL, "sales_datetime"]).groupby(C.ID_COL)[col].apply(lambda x: [str(v) for v in x])
        mc = Word2Vec(cs.tolist(), vector_size=cdim, window=5, min_count=3, workers=4, sg=1, epochs=8, seed=42)
        wv = mc.wv
        vm = np.array([np.mean([wv[g] for g in s if g in wv] or [np.zeros(cdim)], axis=0) for s in cs.values])
        blocks.append(pd.DataFrame(vm, index=cs.index).add_prefix(f"w2v_{col}_"))
    # GPT Q1: NN "유사 feature" — goodcd 임베딩 KMeans 군집 → 고객 군집노출도(비슷한 상품군 묶음)
    from sklearn.cluster import MiniBatchKMeans
    gw2v = goodcd_w2v_wv                                       # 위 루프서 저장한 W2V wv
    vocab = [g for g in gw2v.index_to_key]
    gv = np.array([gw2v[g] for g in vocab])
    km = MiniBatchKMeans(n_clusters=40, random_state=42, n_init=3).fit(gv)
    g2c = dict(zip(vocab, km.labels_))
    rows = np.zeros((N, 40))
    for i, s in enumerate(seqs.values):
        for g in s:
            c = g2c.get(g)
            if c is not None:
                rows[i, c] += 1
    rows = rows / (rows.sum(1, keepdims=True) + 1e-9)          # 군집 노출 분포
    blocks.append(pd.DataFrame(rows, index=seqs.index).add_prefix("gcclu_"))
    R = pd.concat(blocks, axis=1); R.index.name = "custid"
    return R


# ---------- 10. 할인+피크 ----------
def build_discpeak(df):
    d = df.copy(); d["sales_time"] = d["sales_time"].fillna(0).astype(int); d["dp_hour"] = d["sales_time"] // 100
    d["dp_disc_rate"] = np.where(d["tot_amt"] > 0, (d["dis_amt"] / (d["tot_amt"] + 1e-5)) * 100, 0)
    d["dp_high_disc_flag"] = (d["dp_disc_rate"] >= 10).astype(int); d["dp_disc_flag"] = (d["dis_amt"] > 0).astype(int)
    da = d.groupby("custid").agg({"dp_disc_rate": ["mean", "max"], "dp_high_disc_flag": ["sum", "mean"], "dp_disc_flag": ["mean"]})
    da.columns = ["dp_disc_mean", "dp_disc_max", "dp_high_disc_cnt", "dp_high_disc_ratio", "dp_disc_ratio"]
    d["dp_lunch_flag"] = d["dp_hour"].between(11, 13).astype(int); d["dp_evening_flag"] = d["dp_hour"].between(17, 19).astype(int)
    pa = d.groupby("custid").agg({"dp_lunch_flag": ["sum", "mean"], "dp_evening_flag": ["sum", "mean"], "dp_hour": ["std"]})
    pa.columns = ["dp_lunch_cnt", "dp_lunch_ratio", "dp_evening_cnt", "dp_evening_ratio", "dp_hour_std"]
    return pd.concat([da, pa], axis=1)


# ---------- 11. 환불 강도 ----------
def build_refund(df):
    d = df.copy()
    base = np.where(d["net_amt"].fillna(0) != 0, d["net_amt"].fillna(0), d["tot_amt"].fillna(0))
    d["ri_refund_flag"] = (base < 0).astype(int)
    tot = d.groupby("custid")["ri_refund_flag"].size().to_frame("ri_total_trx_cnt")
    ref = d[d["ri_refund_flag"] == 1].copy()
    if ref.empty:
        out = tot.copy()
        for c in ["ri_refund_cnt", "ri_refund_ratio", "ri_total_refund_amt", "ri_mean_refund_amt", "ri_max_refund_amt", "ri_heavy_refund_flag"]:
            out[c] = 0.0
        return out
    bref = np.where(ref["net_amt"].fillna(0) != 0, ref["net_amt"].fillna(0), ref["tot_amt"].fillna(0))
    ref["ri_refund_amt"] = np.abs(bref)
    ra = ref.groupby("custid")["ri_refund_amt"].agg(["sum", "mean", "max", "count"])
    ra.columns = ["ri_total_refund_amt", "ri_mean_refund_amt", "ri_max_refund_amt", "ri_refund_cnt"]
    out = tot.join(ra, how="left").fillna(0)
    out["ri_refund_ratio"] = out["ri_refund_cnt"] / (out["ri_total_trx_cnt"] + 1e-5)
    out["ri_heavy_refund_flag"] = ((out["ri_refund_ratio"] >= 0.10) | (out["ri_refund_cnt"] >= 3)).astype(int)
    return out


# ---------- 12. Social Time ----------
def build_social(df):
    d = df.copy(); d["sales_time"] = d["sales_time"].fillna(0).astype(int); d["st_hour"] = d["sales_time"] // 100
    dow = d["sales_date"].dt.dayofweek
    flag = ((dow <= 4) & (d["st_hour"] >= 18)) | ((dow >= 5) & (d["st_hour"].between(12, 21)))
    d["st_social_flag"] = flag.astype(int); d["st_social_amt"] = d["tot_amt"] * d["st_social_flag"]
    g = d.groupby("custid")
    tt = g["st_social_flag"].size().rename("st_total_trx_cnt"); st = g["st_social_flag"].sum().rename("st_social_trx_cnt")
    ta = g["tot_amt"].sum().rename("st_total_amt"); sa = g["st_social_amt"].sum().rename("st_social_amt")
    return pd.concat([tt, st, (st / (tt + 1e-5)).rename("st_social_trx_ratio"),
                      ta, sa, (sa / (ta + 1e-5)).rename("st_social_amt_ratio")], axis=1)


# ---------- 13. Dwell Time ----------
def build_dwell(df):
    d = df.copy(); d["sales_time"] = d["sales_time"].fillna(0).astype(int)
    hour = d["sales_time"] // 100; minute = (d["sales_time"] % 100).clip(0, 59)
    d["dw_time_min"] = hour * 60 + minute
    dow = d["sales_date"].dt.dayofweek
    soc = ((dow <= 4) & (hour >= 18)) | ((dow >= 5) & (hour.between(12, 21)))
    d["dw_is_social_trx"] = soc.astype(int)
    g = d.groupby(["custid", "sales_date"])
    basic = g["dw_time_min"].agg(["min", "max", "count"]).rename(columns={"count": "day_trx_cnt"})
    basic["span_min"] = (basic["max"] - basic["min"]).clip(lower=0)
    day = pd.concat([basic, g["tot_amt"].sum().rename("day_total_amt"), g["dw_is_social_trx"].max().rename("day_social_flag")], axis=1)
    day["long_flag"] = (day["span_min"] >= 60).astype(int)
    day["amt_per_dwell"] = day["day_total_amt"] / (day["span_min"] + 1)
    cg = day.groupby("custid")
    sp = cg["span_min"].agg(["mean", "max"]); sp.columns = ["dw_span_mean_min", "dw_span_max_min"]
    out = pd.concat([sp, cg["long_flag"].mean().rename("dw_longstay_ratio"),
                     cg["amt_per_dwell"].mean().rename("dw_amt_per_dwell_mean"),
                     (((day["day_social_flag"] == 1) & (day["long_flag"] == 1)).groupby(day.index.get_level_values("custid")).mean()).rename("dw_social_longstay_ratio")], axis=1).fillna(0)
    return out


# ---------- 14. 최근 K회 ----------
def build_recent(df, k):
    d = df.copy(); d["sales_time"] = d["sales_time"].fillna(0).astype(int)
    hour = d["sales_time"] // 100; minute = (d["sales_time"] % 100).clip(0, 59)
    d["dtf"] = d["sales_date"] + pd.to_timedelta(hour, unit="h") + pd.to_timedelta(minute, unit="m")
    d = d.sort_values(["custid", "dtf"], ascending=[True, False])
    d["rn"] = d.groupby("custid").cumcount()
    r = d[d["rn"] < k].copy()
    r["part_norm"] = r["part_nm"].fillna("").astype(str).replace(_REPLACE)
    r["mf"] = r["part_norm"].isin(_MALE_PARTS).astype(int); r["ff"] = r["part_norm"].isin(_FEMALE_PARTS).astype(int)
    g = r.groupby("custid")
    out = pd.DataFrame(index=g.size().index)
    out[f"recent_{k}_tot_amt_mean"] = g["tot_amt"].mean()
    out[f"recent_{k}_tot_amt_max"] = g["tot_amt"].max()
    out[f"recent_{k}_disc_rate_mean"] = g["dis_amt"].sum() / (g["tot_amt"].sum() + 1e-5)
    out[f"recent_{k}_male_ratio"] = g["mf"].mean(); out[f"recent_{k}_female_ratio"] = g["ff"].mean()
    return out.fillna(0)


def build_household(df):
    """가족대표/가정주부 쇼퍼 신호 (에러분석: 여성이 가족용 대량구매→남성으로 오분류).
    식품·아동·생활 = 가정주부 바스켓, 남성×식품/아동 = 대리구매(여성이 남편/가족용)."""
    d = df.copy()
    cat = (d["part_nm"].astype(str) + " " + d["pc_nm"].astype(str) + " " + d["corner_nm"].astype(str))
    w = d["tot_amt"].clip(lower=0)
    gid = d[C.ID_COL]; tot = w.groupby(gid).sum() + 1.0
    segs = {"food": "식품|농산|생식|청과|수산|축산", "kids": "아동|유아|키즈|완구|문구",
            "living": "가정용품|공산품|주방|생활|침구", "men": "남성|신사", "women": "여성|숙녀",
            "cos": "화장품|향수", "golf": "골프|스포츠"}
    r = {}
    for s, kw in segs.items():
        r[f"hh_{s}"] = (w * cat.str.contains(kw, regex=True)).groupby(gid).sum() / tot
    R = pd.DataFrame(r)
    R["hh_homemaker"] = R["hh_food"] + R["hh_kids"] + R["hh_living"]          # 가정주부 바스켓
    R["hh_proxy_male"] = R["hh_men"] * (R["hh_food"] + R["hh_kids"])          # 대리구매(여성→남성용)
    R["hh_family_breadth"] = (R[[f"hh_{s}" for s in segs]] > 0.02).sum(axis=1)  # 세그먼트 폭(가족쇼핑)
    R["hh_cos_vs_male"] = R["hh_cos"] - R["hh_men"]                            # 화장품 우위(여성단서)
    # 신호충돌(conflict): 남성신호·여성신호 동시에 강함 = 가족대표쇼퍼만의 특수축 (차이로는 못잡음)
    R["hh_male_sig"] = R["hh_men"] + R["hh_golf"]                              # 남성신호
    R["hh_female_sig"] = R["hh_cos"] + R["hh_women"]                           # 여성신호
    R["hh_conflict"] = R["hh_male_sig"] * R["hh_female_sig"]                   # 둘다高=충돌(핵심)
    R["hh_proxy_male2"] = R["hh_men"] * (R["hh_food"] + R["hh_kids"] + R["hh_living"])  # 대리구매(living포함)
    return R


def build_giftkr(df):
    """한국 명절 gift-window (어버이날/추석/설/빼빼로/어린이날) — 대리·교차성별 선물 신호.
    서양명절만 있던 gift-season의 한국판. 선물기간×선물성카테고리 = 가족대표/교차성별 단서(누수아님,달력)."""
    d = df[[C.ID_COL, "sales_datetime", "tot_amt", "part_nm", "pc_nm", "corner_nm"]].copy()
    dt = pd.to_datetime(d["sales_datetime"], errors="coerce")
    md = dt.dt.strftime("%m-%d")
    w = d["tot_amt"].clip(lower=0); gid = d[C.ID_COL]; tot = w.groupby(gid).sum() + 1.0
    win = {
        "parents": ((md >= "04-28") & (md <= "05-08")),                       # 어버이날·어린이날(자식↔부모)
        "pepero": ((md >= "11-05") & (md <= "11-11")),                        # 빼빼로데이(교차성별)
        "chuseok": ((dt >= "2017-09-20") & (dt <= "2017-10-06")),             # 추석2017(가족선물)
        "seollal": ((dt >= "2018-01-28") & (dt <= "2018-02-17")),             # 설날2018(가족선물)
    }
    giftcat = (d["part_nm"].astype(str) + d["pc_nm"].astype(str) + d["corner_nm"].astype(str)).str.contains(
        "화장품|향수|주얼리|시계|상품권|건강|홍삼|선물|기프트|넥타이|지갑|벨트", regex=True)
    out = {}; anyg = pd.Series(False, index=d.index)
    for k, m in win.items():
        m = m.fillna(False); out[f"gk_{k}"] = (w * m).groupby(gid).sum() / tot
        anyg = anyg | m
    out["gk_any"] = (w * anyg).groupby(gid).sum() / tot
    out["gk_giftcat"] = (w * anyg.values * giftcat.values).groupby(gid).sum() / tot   # 선물기간×선물카테고리
    out["gk_giftcat_off"] = (w * (~anyg).values * giftcat.values).groupby(gid).sum() / tot  # 비선물기간 선물카테고리(자기용)
    R = pd.DataFrame(out); R.index.name = "custid"
    return R


def build_brandmeta(df):
    """외부 브랜드지식(brand_meta.csv: gender/age/price/kids/luxury/sports/cosmetic) 금액가중 평균.
    강모델(build_all)이 미사용하던 하드라벨 사전확률 → 데이터TE와 다른 신호(특히 희귀브랜드). 누수아님(외부지식).
    bm_gender_std = 한 고객이 남성/여성 브랜드 섞어 사는 정도 = 가족대표쇼퍼 단서."""
    path = C.DATA_DIR / "brand_meta.csv"
    if not path.exists():
        return pd.DataFrame(index=df[C.ID_COL].unique())
    meta = pd.read_csv(path); meta["brd_nm"] = meta["brd_nm"].astype(str)
    attrs = [c for c in meta.columns if c != "brd_nm"]
    d = df[[C.ID_COL, "brd_nm", "tot_amt"]].copy()
    d["brd_nm"] = d["brd_nm"].astype(str)
    d = d.merge(meta, on="brd_nm", how="left")
    w = d["tot_amt"].clip(lower=0) + 1.0; gid = d[C.ID_COL]
    wsum = w.groupby(gid).sum() + 1e-9
    out = {}
    for a in attrs:
        out[f"bm_{a}"] = (d[a] * w).groupby(gid).sum() / wsum
    out["bm_gender_std"] = d.groupby(gid)["gender"].std()        # 남녀브랜드 혼재도
    out["bm_gender_min"] = d.groupby(gid)["gender"].min()
    out["bm_gender_max"] = d.groupby(gid)["gender"].max()
    R = pd.DataFrame(out); R.index.name = "custid"
    return R


def build_all():
    tr, te, y, full = _load()
    full["str_part_key"] = full["str_nm"].astype(str) + "_" + full["part_nm"].astype(str)
    tr_only = full[full["dataset"] == "train"].copy(); te_only = full[full["dataset"] == "test"].copy()
    print("FE: style/behavior/calendar/interest/entropy/base/tfidf/disc/refund/social/dwell/recent ...")
    bt, ct, bs, cs = build_tfidf(full)
    gs = build_goodcd_svd(full)   # 정크제거 goodcd SVD (finest 신호)
    te_blocks = []
    for col in ["brd_nm", "corner_nm", "part_nm", "str_part_key"]:
        te_blocks.append(pd.concat([_kfold_te(tr_only, y, col), _te_test(tr_only, te_only, y, col)], axis=0))
    te_feats = pd.concat(te_blocks, axis=1)
    allf = (build_base(full)
            .join(build_style(full), how="left").join(build_behavior(full), how="left")
            .join(build_calendar(full), how="left").join(build_interest(full), how="left")
            .join(build_entropy(full), how="left").join(bt, how="left").join(ct, how="left")
            .join(bs, how="left").join(cs, how="left").join(gs, how="left").join(te_feats, how="left")
            .join(build_discpeak(full), how="left").join(build_refund(full), how="left")
            .join(build_dwell(full), how="left").join(build_social(full), how="left")
            .join(build_recent(full, 5), how="left").join(build_recent(full, 3), how="left"))
    if os.environ.get("KML_COOC") == "1":          # test-aware goodcd 동시출현(GPT #2). 검증전까진 env로 분리
        allf = allf.join(build_cooc(full), how="left")
        print("  +cooc (test-aware goodcd PPMI)")
    if os.environ.get("KML_EMB") == "1":           # 교수힌트 #1 W2V + #3 FastText goodcd 임베딩
        allf = allf.join(build_emb(full), how="left")
        print("  +emb (W2V + FastText goodcd)")
    # NOTE: build_household — CV 하락(-0.0019) → 제외
    # NOTE: build_brandmeta + goodcd접두사TE — CV -0.00065(데이터TE에 흡수) → 제외
    # NOTE: build_giftkr(한국명절 어버이날/추석/설/빼빼로) — CV -0.00028(캘린더+카테고리TE에 흡수) → 제외
    allf.index.name = "custid"; allf = allf.fillna(0)
    allf.columns = [re.sub(r"[^0-9a-zA-Z가-힣_]", "_", str(c)) for c in allf.columns]
    print(f"통합 직후: {allf.shape[1]}")
    visit = allf["base_freq"] + 1e-5
    for c in list(allf.columns):
        if any(c.startswith(p) for p in ["cal_wd_", "cal_season_", "cal_time_", "cal_buyer_", "base_str_cnt_"]):
            allf[c + "_ratio"] = allf[c] / visit
    if {"cal_male_ratio", "cal_female_ratio"}.issubset(allf.columns):
        allf["gender_part_score"] = allf["cal_male_ratio"] - allf["cal_female_ratio"]
    allf["gender_interest_score"] = (allf.get("int_ratio_남성", 0) - allf.get("int_ratio_여성", 0) + 0.5 * allf.get("int_ratio_아동", 0))
    for c in allf.columns:
        if allf[c].dtype.kind in "iufb" and any(kw in c for kw in ["amt", "sum", "ticket", "refund", "recency", "span", "freq", "cnt", "total"]):
            col = allf[c].astype(float); col = np.where(col < 0, 0, col)
            lo, hi = np.nanpercentile(col, [1, 99]); allf[c] = np.log1p(np.clip(col, lo, hi))
    v = allf.var(); allf = allf.drop(columns=v[v < 1e-6].index.tolist())
    nz = (allf != 0).mean(axis=0); allf = allf.drop(columns=nz[nz < 0.003].index.tolist())
    print(f"정리 후 최종: {allf.shape[1]}")
    return allf, y, tr, te


def main():
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    allf, y_df, _, _ = build_all()
    X = allf.reindex(train_ids).fillna(0)
    Xt = allf.reindex(test_ids).fillna(0)
    y = y_df.set_index("custid").reindex(train_ids)["gender"].to_numpy()
    print(f"X={X.shape} Xt={Xt.shape}")

    params = dict(iterations=1658, learning_rate=0.0107, depth=7, l2_leaf_reg=13.9,
                  subsample=0.708, colsample_bylevel=0.818, min_data_in_leaf=39,
                  random_strength=3.15, eval_metric="AUC", random_seed=C.SEED,
                  verbose=0, allow_writing_files=False)
    oof = np.full(len(y), np.nan); test_sum = np.zeros(len(test_ids))
    for f in range(C.N_FOLDS):
        tri, va = np.where(folds != f)[0], np.where(folds == f)[0]
        m = CatBoostClassifier(**params)
        m.fit(X.iloc[tri], y[tri])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]; test_sum += m.predict_proba(Xt)[:, 1]
        print(f"[fold {f}] AUC={roc_auc_score(y[va], oof[va]):.5f}")
    cv = float(roc_auc_score(y, oof))
    print(f"\n==== {MODEL_NAME}  CV AUC = {cv:.5f} ====")
    save_predictions(MODEL_NAME, oof, test_sum / C.N_FOLDS, meta=dict(
        cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS, feature_set=f"작년1등 FE ({X.shape[1]}feat) CatBoost",
        created_by="hyunbean", notes="2025 1st-place notebook FE ported + tuned CatBoost, 5fold OOF"))


if __name__ == "__main__":
    main()
