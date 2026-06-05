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

    # test: 전체 train으로 인코딩
    rate_full = _category_rate(cc_tr, gender, set(train_ids), col, alpha, global_rate)
    test_te = _agg_rate(cc_te, rate_full, col, global_rate).reindex(test_ids).fillna(global_rate)
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
    key = f"feat_te{int(use_te)}_emb{int(use_emb)}_g{int(use_groups)}_fe2{int(use_fe2)}_v{emb_vector_size}_r3"
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
