"""작년 1등 솔루션 피처셋 + 튜닝 CatBoost → 5-fold OOF 베이스모델 'first_cat'.

1등 노트북(2025 ML Pipeline Hackathon)의 FE를 그대로 포팅(우리 sales_datetime→date/time 파생).
우리 기존 피처와 구성이 달라(dwell/buyer/social/recent-K/수작업 성별파트) 결이 다른 멤버 기대.
TE는 내부 KFold로 leak-safe, 모델 OOF는 우리 folds.npy 사용.
실행: pip install catboost → python -m src.folds → python -m src.train_first
"""
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


def build_all():
    tr, te, y, full = _load()
    full["str_part_key"] = full["str_nm"].astype(str) + "_" + full["part_nm"].astype(str)
    tr_only = full[full["dataset"] == "train"].copy(); te_only = full[full["dataset"] == "test"].copy()
    print("FE: style/behavior/calendar/interest/entropy/base/tfidf/disc/refund/social/dwell/recent ...")
    bt, ct, bs, cs = build_tfidf(full)
    te_blocks = []
    for col in ["brd_nm", "corner_nm", "part_nm", "str_part_key"]:
        te_blocks.append(pd.concat([_kfold_te(tr_only, y, col), _te_test(tr_only, te_only, y, col)], axis=0))
    te_feats = pd.concat(te_blocks, axis=1)
    allf = (build_base(full)
            .join(build_style(full), how="left").join(build_behavior(full), how="left")
            .join(build_calendar(full), how="left").join(build_interest(full), how="left")
            .join(build_entropy(full), how="left").join(bt, how="left").join(ct, how="left")
            .join(bs, how="left").join(cs, how="left").join(te_feats, how="left")
            .join(build_discpeak(full), how="left").join(build_refund(full), how="left")
            .join(build_dwell(full), how="left").join(build_social(full), how="left")
            .join(build_recent(full, 5), how="left").join(build_recent(full, 3), how="left"))
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
