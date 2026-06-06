"""피처 엔지니어링 — 베이스라인 집계 + 구성비 + 다양성/엔트로피 + OOF 타깃인코딩 + W2V 임베딩.

설계 원칙
- 모든 산출물은 '정규 custid 순서'(folds.py 출력)에 정렬 → 팀 OOF 규격 일치.
- **타깃인코딩은 OOF(fold-safe)** : val 폴드는 그 폴드를 제외한 train으로만 인코딩, test는 전체 train으로.
- **임베딩/구성비/다양성은 비지도** : train+test 전체로 만들어도 누수 없음.
- 무거운 단계(TE/임베딩)는 artifacts/features 에 캐시 → 재실행 시 즉시 로드.

사용:
  python -m src.folds            # (선행) 정규순서/폴드
  python -m src.features         # 피처 빌드 + 캐시 (shape 확인)
  # 코드에서:  from src.features import build_features;  X, y, Xtest = build_features()
"""
import json
import math
import re
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation

from . import config as C
from . import data as D

FEAT_DIR = C.ARTIFACTS / "features"
FEAT_DIR.mkdir(parents=True, exist_ok=True)

# 어떤 컬럼에 무엇을 적용할지 (카디널리티 고려)
TE_COLS = ["part_nm", "pc_nm", "brd_nm", "corner_nm", "buyer_nm", "str_nm", "goodcd"]  # 타깃인코딩
CROSS_TE = [("part_nm", "time_zone"), ("part_nm", "season")]   # 교차 타깃인코딩(상호작용)
EMB_COLS = ["corner_nm", "brd_nm", "pc_nm", "part_nm"]          # W2V 임베딩
COMP_COLS = ["team_nm", "part_nm", "season", "time_zone"]       # 저카디널리티 → 구성비(crosstab)
DIV_COLS = ["brd_nm", "part_nm", "corner_nm", "time_zone", "str_nm"]  # 다양성/엔트로피
TE_ALPHA = 20.0   # 베이지안 스무딩 강도


# ----------------------------- 공통 -----------------------------
def _load_canonical():
    for p in (C.TRAIN_IDS_NPY, C.TEST_IDS_NPY, C.FOLDS_NPY):
        if not p.exists():
            raise FileNotFoundError(f"{p.name} 없음 — 먼저 `python -m src.folds` 실행하세요.")
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    y = pd.read_csv(C.YTRAIN_CSV).set_index(C.ID_COL).reindex(train_ids)[C.TARGET].to_numpy()
    return train_ids, test_ids, folds, y


def _sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """LightGBM이 거부하는 JSON 특수문자(,[]{}\":)를 컬럼명에서 _ 로 치환 + 중복 방지.
    (구성비 컬럼에 한글 카테고리명의 특수문자가 섞여 들어오는 문제 해결)"""
    seen, new_cols = {}, []
    for c in df.columns:
        c2 = re.sub(r'[\[\]{}":,<>]', "_", str(c))
        if c2 in seen:
            seen[c2] += 1
            c2 = f"{c2}__{seen[c2]}"
        else:
            seen[c2] = 0
        new_cols.append(c2)
    df.columns = new_cols
    return df


def _amount_stats(df, ids):
    """금액 분포 통계 (분위/IQR/skew) + 고가·명품 비율 — 도메인 파생 v3."""
    g = df.groupby(C.ID_COL)["net_amt"]
    out = pd.DataFrame({
        "net_q25": g.quantile(0.25),
        "net_q75": g.quantile(0.75),
        "net_skew": g.skew(),
    })
    out["net_iqr"] = out["net_q75"] - out["net_q25"]
    thr = df["net_amt"].quantile(0.9)
    tmp = df.assign(_hv=(df["net_amt"] >= thr).astype(float),
                    _lux=((df["import_flg"] == 1) & (df["net_amt"] >= thr)).astype(float))
    out["high_value_ratio"] = tmp.groupby(C.ID_COL)["_hv"].mean()
    out["luxury_ratio"] = tmp.groupby(C.ID_COL)["_lux"].mean()
    return out.reindex(ids).fillna(0.0)


def _temporal_dynamics(df, ids):
    """시간축 행동 동역학 — recency/tenure/활동개월/월빈도/후반지출비중.
    정적 집계와 달리 '시간 변화' 정보라 새 신호(중복 아님)."""
    d = df[[C.ID_COL, "sales_datetime", "net_amt"]].copy()
    d["sales_datetime"] = pd.to_datetime(d["sales_datetime"])
    ref = d["sales_datetime"].max()
    first = d.groupby(C.ID_COL)["sales_datetime"].transform("min")
    last = d.groupby(C.ID_COL)["sales_datetime"].transform("max")
    span = (last - first).dt.total_seconds()
    pos = ((d["sales_datetime"] - first).dt.total_seconds() / span.where(span > 0)).fillna(0.5)
    w = d["net_amt"].clip(lower=0)
    gid = d[C.ID_COL]
    g_last = d.groupby(C.ID_COL)["sales_datetime"].max()
    g_first = d.groupby(C.ID_COL)["sales_datetime"].min()
    out = pd.DataFrame({
        "recency_days": (ref - g_last).dt.days.astype(float),
        "tenure_days": (g_last - g_first).dt.days.astype(float),
    })
    out["active_months"] = d.assign(_m=d["sales_datetime"].dt.to_period("M")).groupby(C.ID_COL)["_m"].nunique().astype(float)
    out["txn_per_month"] = d.groupby(C.ID_COL).size().astype(float) / out["active_months"].clip(lower=1)
    out["late_spend_ratio"] = (w * (pos >= 0.5)).groupby(gid).sum() / (w.groupby(gid).sum() + 1.0)
    return out.reindex(ids).fillna(0.0)


def _brand_meta_features(df, ids):
    """data/brand_meta.csv (brd_nm + 숫자 속성)를 거래에 join → 고객별 금액가중 평균.
    파일 없으면 None. 외부 지식(브랜드→성별/연령/가격대 등)이라 누수 아님."""
    path = C.DATA_DIR / "brand_meta.csv"
    if not path.exists():
        return None
    meta = pd.read_csv(path)
    meta["brd_nm"] = meta["brd_nm"].astype(str)
    attrs = [c for c in meta.columns if c != "brd_nm"]
    defaults = {a: float(meta[a].mean()) for a in attrs}
    m = df[[C.ID_COL, "brd_nm", "net_amt"]].copy()
    m["brd_nm"] = m["brd_nm"].astype(str)
    m = m.merge(meta, on="brd_nm", how="left")
    m["_w"] = m["net_amt"].clip(lower=0) + 1.0
    gid = m[C.ID_COL]
    wsum = m["_w"].groupby(gid).sum()
    out = {}
    for a in attrs:
        out[f"bm_{a}"] = (m[a].fillna(defaults[a]) * m["_w"]).groupby(gid).sum() / wsum
    return pd.DataFrame(out).reindex(ids).fillna(0.0)


def _add_time(df: pd.DataFrame) -> pd.DataFrame:
    df["sales_datetime"] = pd.to_datetime(df["sales_datetime"])
    df["hour"] = df["sales_datetime"].dt.hour
    df["month"] = df["sales_datetime"].dt.month
    season_map = {3: "봄", 4: "봄", 5: "봄", 6: "여름", 7: "여름", 8: "여름",
                  9: "가을", 10: "가을", 11: "가을", 12: "겨울", 1: "겨울", 2: "겨울"}
    df["season"] = df["month"].map(season_map)
    df["time_zone"] = pd.cut(df["hour"], bins=[-1, 11, 14, 17, 24],
                             labels=["오전", "점심", "오후", "저녁"]).astype(str)
    return df


# ----------------------------- 구성비 -----------------------------
def _composition(df, col, ids):
    ct = pd.crosstab(df[C.ID_COL], df[col], normalize="index").add_prefix(f"ratio_{col}_")
    return ct.reindex(ids).fillna(0.0)


# ----------------------------- 다양성/엔트로피/집중도 -----------------------------
def _diversity(df, col, ids):
    cc = df.groupby([C.ID_COL, col]).size().rename("n").reset_index()
    tot = cc.groupby(C.ID_COL)["n"].transform("sum")
    cc["p"] = cc["n"] / tot
    cc["plogp"] = -cc["p"] * np.log(cc["p"] + 1e-12)
    cc["p2"] = cc["p"] ** 2
    g = cc.groupby(C.ID_COL).agg(
        **{f"{col}_entropy": ("plogp", "sum"),
           f"{col}_hhi": ("p2", "sum"),          # Herfindahl 집중도
           f"{col}_top1": ("p", "max")})
    return g.reindex(ids).fillna(0.0)


# ----------------------------- 1등 노트북 차용 피처 (dwell/social/recent-K/gender/buyer) -----------------------------
# 작년 1등 솔루션에서 우리가 안 쓰던 신호만 선별 차용. 모두 비지도(누수 없음).
_REPLACE_PART = {"가정용품파트": "가정용품", "공산품파트": "공산품", "생식품파트": "생식품",
                 "잡화파트": "잡화", "로얄부틱": "로얄부띠끄", "스포츠캐쥬얼": "스포츠캐주얼", "여성캐쥬얼": "여성캐주얼"}
_MALE_PARTS = ["가정용품", "공산품", "생식품", "케주얼,구두,아동", "남성정장", "남성캐주얼"]
_FEMALE_PARTS = ["여성캐주얼", "영캐릭터", "영플라자", "패션잡화", "여성정장"]


def _first_extra_features(df, ids):
    d = df[[C.ID_COL, "sales_datetime", "part_nm", "buyer_nm", "brd_nm", "corner_nm", "net_amt"]].copy()
    dt = pd.to_datetime(d["sales_datetime"])
    d["date"] = dt.dt.normalize()
    hour = dt.dt.hour; d["tmin"] = hour * 60 + dt.dt.minute
    dow = dt.dt.dayofweek
    gid = d[C.ID_COL]
    amt = d["net_amt"].clip(lower=0)
    res = {}

    # --- social time (평일저녁 + 주말낮) ---
    d["_soc"] = (((dow <= 4) & (hour >= 18)) | ((dow >= 5) & (hour.between(12, 21)))).astype(int)
    tot_trx = d.groupby(C.ID_COL).size()
    res["fe_social_trx_ratio"] = d.groupby(C.ID_COL)["_soc"].sum() / tot_trx
    res["fe_social_amt_ratio"] = (amt * d["_soc"]).groupby(gid).sum() / (amt.groupby(gid).sum() + 1)

    # --- dwell time (하루 첫~마지막 거래 시간차) ---
    dg = d.groupby([C.ID_COL, "date"])
    day = pd.DataFrame({"span": (dg["tmin"].max() - dg["tmin"].min()).clip(lower=0),
                        "amt": d.groupby([C.ID_COL, "date"])["net_amt"].sum().clip(lower=0),
                        "soc": dg["_soc"].max()})
    day["long"] = (day["span"] >= 60).astype(int)
    day["apd"] = day["amt"] / (day["span"] + 1)
    cg = day.groupby(level=0)
    res["fe_dwell_span_mean"] = cg["span"].mean()
    res["fe_dwell_span_max"] = cg["span"].max()
    res["fe_dwell_longstay_ratio"] = cg["long"].mean()
    res["fe_dwell_amt_per_min"] = cg["apd"].mean()
    res["fe_dwell_social_long_ratio"] = ((day["soc"] == 1) & (day["long"] == 1)).groupby(level=0).mean()

    # --- 수작업 male/female 파트 비율 + gender score ---
    pn = d["part_nm"].fillna("").astype(str).replace(_REPLACE_PART)
    mc = pn.isin(_MALE_PARTS).astype(int).groupby(gid).sum()
    fc = pn.isin(_FEMALE_PARTS).astype(int).groupby(gid).sum()
    tt = mc + fc
    res["fe_male_part_ratio"] = mc / (tt + 1e-5)
    res["fe_female_part_ratio"] = fc / (tt + 1e-5)
    res["fe_gender_part_score"] = (mc - fc) / (tt + 1e-5)

    # --- 관심 키워드 성별 스코어 ---
    txt = d["brd_nm"].fillna("") + " " + d["corner_nm"].fillna("") + " " + d["part_nm"].fillna("")
    atot = amt.groupby(gid).sum() + 1e-5

    def _kwr(kw):
        return (amt * txt.str.contains(kw, na=False)).groupby(gid).sum() / atot
    r_m, r_f, r_k = _kwr("남성"), _kwr("여성"), _kwr("아동")
    res["fe_int_men"] = r_m; res["fe_int_women"] = r_f
    res["fe_gender_interest_score"] = r_m - r_f + 0.5 * r_k

    # --- buyer(바이어) 다양성/집중도 ---
    res["fe_n_buyers"] = d.groupby(C.ID_COL)["buyer_nm"].nunique()
    bc = d.groupby([C.ID_COL, "buyer_nm"]).size()
    res["fe_buyer_top_share"] = bc.groupby(level=0).max() / bc.groupby(level=0).sum()

    # --- recent-K (최근 3·5건) ---
    d2 = d.sort_values([C.ID_COL, "sales_datetime"], ascending=[True, False])
    d2["rn"] = d2.groupby(C.ID_COL).cumcount()
    pn2 = d2["part_nm"].fillna("").astype(str).replace(_REPLACE_PART)
    d2["_m"] = pn2.isin(_MALE_PARTS).astype(int); d2["_f"] = pn2.isin(_FEMALE_PARTS).astype(int)
    for k in (3, 5):
        rg = d2[d2["rn"] < k].groupby(C.ID_COL)
        res[f"fe_recent{k}_male_ratio"] = rg["_m"].mean()
        res[f"fe_recent{k}_female_ratio"] = rg["_f"].mean()
        res[f"fe_recent{k}_amt_mean"] = rg["net_amt"].mean()

    out = pd.DataFrame({k: v.reindex(ids) for k, v in res.items()}).fillna(0.0)
    out.index = ids
    return out


# ----------------------------- 의미그룹 비중 (도메인 파생변수 v2) -----------------------------
# EDA(카테고리별 성비) 기반. part_nm+pc_nm+corner_nm 텍스트에 키워드 매칭 → 고객별 구매 비중.
# 방향(어느 성별)은 모델이 학습. 철자변형(캐주얼/캐쥬얼 등) 커버 위해 부분일치 사용.
SEMANTIC_GROUPS = {
    "cosmetic":    ["화장품", "향수", "코스메", "뷰티"],
    "women_young": ["영캐", "여성캐", "미씨", "어덜트캐", "캐릭터캐", "란제리", "영플라자",
                    "영라이브", "영어덜트", "패션잡화", "악세", "액세", "머플러", "스카프", "영트랜디", "진캐주얼", "진케주얼"],
    "men":         ["남성정장", "남성의류", "신사", "정장"],
    "sport_golf":  ["골프", "스포츠"],
    "kids":        ["아동", "유아", "키즈", "완구", "문구", "레고", "문화용품"],
    "food":        ["식품", "농산", "생식", "청과", "수산", "축산", "델리"],
    "living":      ["가정용품", "공산품", "침구", "수예", "주방", "생활", "가전"],
}


def _semantic_group_features(df, ids):
    cat = (df["part_nm"].astype(str) + " " + df["pc_nm"].astype(str) + " " + df["corner_nm"].astype(str))
    w = df["net_amt"].clip(lower=0)                       # 금액 가중(환불 제외)
    gid = df[C.ID_COL]
    amt_tot = w.groupby(gid).sum()
    out = {}
    for name, kws in SEMANTIC_GROUPS.items():
        flag = cat.str.contains("|".join(map(re.escape, kws)), regex=True).astype(float)
        out[f"grp_{name}_cnt"] = flag.groupby(gid).mean()                       # 거래수 비중
        out[f"grp_{name}_amt"] = (flag * w).groupby(gid).sum() / (amt_tot + 1)  # 금액 비중
    return pd.DataFrame(out).reindex(ids).fillna(0.0)


# ----------------------------- 행렬분해 SVD/LDA (피처② 임베딩) -----------------------------
# 고객×카테고리 동시발생 행렬을 분해 → 잠재 성향 축. 비지도(train+test 전체)라 누수 없음.
FE2_SVD = {"brd_nm": 16, "goodcd": 16, "corner_nm": 12}   # col: n_components
FE2_LDA = {"brd_nm": 12, "part_nm": 8}                     # col: n_topics


def _cust_cat_csr(tr, te, col, all_ids):
    """고객(all_ids) × 카테고리 카운트 희소행렬 (train+test 합쳐)."""
    s = pd.concat([tr[[C.ID_COL, col]], te[[C.ID_COL, col]]], ignore_index=True)
    s[col] = s[col].astype(str)
    row = pd.Categorical(s[C.ID_COL], categories=all_ids).codes
    ccode, _ = pd.factorize(s[col])
    keep = row >= 0
    n_cols = int(ccode.max()) + 1 if keep.any() else 1
    return csr_matrix((np.ones(int(keep.sum()), dtype=np.float32), (row[keep], ccode[keep])),
                      shape=(len(all_ids), n_cols))


def _fe2_matrix_factors(tr, te, train_ids, test_ids):
    all_ids = np.concatenate([train_ids, test_ids])
    n_tr = len(train_ids)
    out = {}
    for col, k in FE2_SVD.items():
        M = _cust_cat_csr(tr, te, col, all_ids)
        k_eff = min(k, M.shape[1] - 1)
        if k_eff < 1:
            continue
        comp = TruncatedSVD(n_components=k_eff, random_state=C.SEED).fit_transform(M)
        for i in range(comp.shape[1]):
            out[f"svd_{col}_{i}"] = comp[:, i]
        print(f"[features] SVD {col} (dim={k_eff}) 완료")
    for col, k in FE2_LDA.items():
        M = _cust_cat_csr(tr, te, col, all_ids)
        lda = LatentDirichletAllocation(n_components=k, random_state=C.SEED,
                                        learning_method="online", max_iter=10, n_jobs=-1)
        topic = lda.fit_transform(M)
        for i in range(topic.shape[1]):
            out[f"lda_{col}_{i}"] = topic[:, i]
        print(f"[features] LDA {col} (topics={k}) 완료")
    full = pd.DataFrame(out)
    tr_b = full.iloc[:n_tr].copy(); tr_b.index = train_ids
    te_b = full.iloc[n_tr:].copy(); te_b.index = test_ids
    return tr_b, te_b


# ----------------------------- OOF 타깃인코딩 -----------------------------
def _cc_counts(df, col):
    """고객×카테고리 거래 수."""
    return df.groupby([C.ID_COL, col]).size().rename("cnt").reset_index()


def _category_rate(cc, gender, id_set, col, alpha, global_rate):
    """주어진 train 고객 집합으로 카테고리별 성별 비율(거래수 가중 + 스무딩). Series(idx=카테고리)."""
    sub = cc[cc[C.ID_COL].isin(id_set)].copy()
    sub["g"] = sub[C.ID_COL].map(gender)
    sub["wg"] = sub["cnt"] * sub["g"]
    grp = sub.groupby(col).agg(w=("cnt", "sum"), wg=("wg", "sum"))
    return (grp["wg"] + alpha * global_rate) / (grp["w"] + alpha)


def _agg_rate(cc, rate, col, global_rate):
    """고객이 산 카테고리들의 rate를 집계 → 고객별 피처(거래수 가중평균/최대/최소/표준편차)."""
    sub = cc.copy()
    sub["r"] = sub[col].map(rate).fillna(global_rate)
    sub["rc"] = sub["r"] * sub["cnt"]
    g = sub.groupby(C.ID_COL).agg(rc=("rc", "sum"), c=("cnt", "sum"),
                                  rmax=("r", "max"), rmin=("r", "min"), rstd=("r", "std"))
    return pd.DataFrame({
        f"te_{col}_wmean": g["rc"] / g["c"],
        f"te_{col}_max": g["rmax"],
        f"te_{col}_min": g["rmin"],
        f"te_{col}_std": g["rstd"].fillna(0.0),
    })


def _target_encode(train_df, test_df, col, train_ids, test_ids, folds, y, alpha):
    gender = pd.Series(y, index=train_ids)
    global_rate = float(y.mean())
    cc_tr = _cc_counts(train_df, col)
    cc_te = _cc_counts(test_df, col)

    # train: OOF (val 폴드는 다른 폴드 train으로만 인코딩)
    parts = []
    for f in range(C.N_FOLDS):
        fit_ids = set(train_ids[folds != f])
        val_ids = set(train_ids[folds == f])
        rate = _category_rate(cc_tr, gender, fit_ids, col, alpha, global_rate)
        parts.append(_agg_rate(cc_tr[cc_tr[C.ID_COL].isin(val_ids)], rate, col, global_rate))
    train_te = pd.concat(parts).reindex(train_ids).fillna(global_rate)

    # test: 각 폴드 인코더(80% train)로 만든 TE의 평균 → train OOF와 같은 분포(시프트 완화)
    test_parts = []
    for f in range(C.N_FOLDS):
        fit_ids = set(train_ids[folds != f])
        rate_f = _category_rate(cc_tr, gender, fit_ids, col, alpha, global_rate)
        test_parts.append(_agg_rate(cc_te, rate_f, col, global_rate).reindex(test_ids).fillna(global_rate))
    test_te = sum(test_parts) / C.N_FOLDS
    return train_te, test_te


# ----------------------------- W2V 멀티풀링 임베딩 -----------------------------
def _w2v_pooled(train_df, test_df, col, train_ids, test_ids, vector_size, window=5, epochs=10, seed=42):
    from gensim.models import Word2Vec

    cols = [C.ID_COL, "sales_datetime", col]
    alldf = pd.concat([train_df[cols], test_df[cols]], ignore_index=True)
    alldf = alldf.sort_values([C.ID_COL, "sales_datetime"])
    alldf[col] = alldf[col].astype(str)
    seqs = alldf.groupby(C.ID_COL)[col].apply(list)

    model = Word2Vec(sentences=seqs.tolist(), vector_size=vector_size, window=window,
                     min_count=1, sg=1, workers=4, epochs=epochs, seed=seed)
    kv = model.wv

    # idf (문서=고객) — TF-IDF 가중 평균용
    dfreq = {}
    for s in seqs:
        for t in set(s):
            dfreq[t] = dfreq.get(t, 0) + 1
    N = len(seqs)
    idf = {t: math.log(N / (1 + c)) for t, c in dfreq.items()}

    z = np.zeros(vector_size, dtype=np.float32)

    def pool(tokens):
        idxs = [t for t in tokens if t in kv.key_to_index]
        if not idxs:
            return np.concatenate([z, z, z, z])
        M = np.vstack([kv[t] for t in idxs])
        w = np.array([idf.get(t, 0.0) for t in idxs], dtype=np.float32)
        wmean = (M * w[:, None]).sum(0) / (w.sum() + 1e-9) if w.sum() > 0 else M.mean(0)
        return np.concatenate([M.mean(0), M.max(0), M.std(0), wmean])

    mat = np.vstack(seqs.apply(pool).values).astype(np.float32)
    names = []
    for p in ("mean", "max", "std", "tfidf"):
        names += [f"{col}_w2v_{p}_{i}" for i in range(vector_size)]
    emb = pd.DataFrame(mat, index=seqs.index, columns=names)
    return emb.reindex(train_ids).fillna(0.0), emb.reindex(test_ids).fillna(0.0)


# ----------------------------- 빌드 -----------------------------
def build_features(use_te=True, use_emb=True, use_groups=True, use_fe2=False, emb_vector_size=16,
                   te_cols=TE_COLS, emb_cols=EMB_COLS, alpha=TE_ALPHA, cache=True):
    bm_on = (C.DATA_DIR / "brand_meta.csv").exists()
    key = f"feat_te{int(use_te)}_emb{int(use_emb)}_g{int(use_groups)}_fe2{int(use_fe2)}_v{emb_vector_size}_r6{'_bm' if bm_on else ''}"
    f_tr, f_te = FEAT_DIR / f"{key}_train.pkl", FEAT_DIR / f"{key}_test.pkl"
    train_ids, test_ids, folds, y = _load_canonical()

    if cache and f_tr.exists() and f_te.exists():
        print(f"[features] 캐시 로드: {key}")
        return _sanitize_columns(pd.read_pickle(f_tr)), y, _sanitize_columns(pd.read_pickle(f_te))

    tr, te, _ = D.load_raw()
    tr = _add_time(tr)
    te = _add_time(te)

    blocks_tr, blocks_te = [], []

    # 1) 베이스라인 집계
    blocks_tr.append(D.make_baseline_features(tr).reindex(train_ids))
    blocks_te.append(D.make_baseline_features(te).reindex(test_ids))
    print("[features] baseline 집계 완료")

    # 2) 구성비
    for col in COMP_COLS:
        blocks_tr.append(_composition(tr, col, train_ids))
        blocks_te.append(_composition(te, col, test_ids))
    print("[features] 구성비 완료")

    # 3) 다양성/엔트로피/집중도
    for col in DIV_COLS:
        blocks_tr.append(_diversity(tr, col, train_ids))
        blocks_te.append(_diversity(te, col, test_ids))
    print("[features] 다양성/엔트로피 완료")

    # 3.5) 의미그룹 비중 (도메인 파생변수 v2)
    if use_groups:
        blocks_tr.append(_semantic_group_features(tr, train_ids))
        blocks_te.append(_semantic_group_features(te, test_ids))
        print("[features] 의미그룹 비중(v2) 완료")

    # 3.6) 금액 분포 통계 (파생 v3)
    blocks_tr.append(_amount_stats(tr, train_ids))
    blocks_te.append(_amount_stats(te, test_ids))
    print("[features] 금액분포 통계(v3) 완료")

    # 3.7) 시간축 행동 동역학 (새 신호)
    blocks_tr.append(_temporal_dynamics(tr, train_ids))
    blocks_te.append(_temporal_dynamics(te, test_ids))
    print("[features] 행동 동역학(시간) 완료")

    # 3.8) 1등 노트북 차용 (dwell/social/recent-K/gender-score/buyer)
    blocks_tr.append(_first_extra_features(tr, train_ids))
    blocks_te.append(_first_extra_features(te, test_ids))
    print("[features] 1등차용(dwell/social/recent/gender/buyer) 완료")

    # 4) OOF 타깃인코딩 (+ 교차 TE)
    if use_te:
        for ca, cb in CROSS_TE:                      # 교차 컬럼 생성
            name = f"x_{ca}_{cb}"
            tr[name] = tr[ca].astype(str) + "|" + tr[cb].astype(str)
            te[name] = te[ca].astype(str) + "|" + te[cb].astype(str)
        for col in list(te_cols) + [f"x_{ca}_{cb}" for ca, cb in CROSS_TE]:
            tr_te, te_te = _target_encode(tr, te, col, train_ids, test_ids, folds, y, alpha)
            blocks_tr.append(tr_te)
            blocks_te.append(te_te)
            print(f"[features] TE(OOF) {col} 완료")

    # 5) W2V 임베딩
    if use_emb:
        for col in emb_cols:
            a, b = _w2v_pooled(tr, te, col, train_ids, test_ids, emb_vector_size)
            blocks_tr.append(a)
            blocks_te.append(b)
            print(f"[features] W2V {col} (dim={emb_vector_size}x4) 완료")

    # 6) 행렬분해 SVD/LDA (피처②)
    if use_fe2:
        a, b = _fe2_matrix_factors(tr, te, train_ids, test_ids)
        blocks_tr.append(a)
        blocks_te.append(b)
        print("[features] SVD/LDA(피처②) 완료")

    # 7) 브랜드 외부 메타데이터 (data/brand_meta.csv 있으면 자동)
    if bm_on:
        bm_tr = _brand_meta_features(tr, train_ids)
        bm_te = _brand_meta_features(te, test_ids)
        if bm_tr is not None:
            blocks_tr.append(bm_tr)
            blocks_te.append(bm_te)
            print("[features] 브랜드 메타데이터 완료")

    # train/test 컬럼 정렬 일치시켜 합치기
    X_train = pd.concat(blocks_tr, axis=1).fillna(0.0)
    X_test = pd.concat(blocks_te, axis=1).fillna(0.0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)  # 구성비 카테고리 차이 보정
    X_train = _sanitize_columns(X_train.reset_index(drop=True))
    X_test = _sanitize_columns(X_test.reset_index(drop=True))

    if cache:
        X_train.to_pickle(f_tr)
        X_test.to_pickle(f_te)
        with open(FEAT_DIR / f"{key}_cols.json", "w", encoding="utf-8") as fjson:
            json.dump(list(X_train.columns), fjson, ensure_ascii=False, indent=2)
        print(f"[features] 캐시 저장: {key}")

    return X_train, y, X_test


if __name__ == "__main__":
    X, y, Xtest = build_features()
    print(f"\nX_train={X.shape}  X_test={Xtest.shape}  features={X.shape[1]}  pos_rate={y.mean():.4f}")
