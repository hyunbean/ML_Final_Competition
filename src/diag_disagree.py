"""정답 불일치(disagreement) 분석 (GPT 조언) — strongest 3모델 OOF에서
'A만 정답/B만 정답/C만 정답/전부오답' 고객군 수 + 특징.

남은 점수는 '누가 틀리냐'(공통오답=흡수됨)보다 '누가 혼자 맞추냐'(희귀 직교신호)에 있다는 가설 검증.
실행(로컬, GPU불필요): python -m src.diag_disagree
"""
import numpy as np
import pandas as pd

from . import config as C

MODELS = ["first_xgb", "first_cat", "first_lgbm"]
SEGS = {"식품": "식품|농산|생식|청과|수산|축산", "화장품": "화장품|향수", "여성": "여성|숙녀",
        "남성": "남성|신사", "골프": "골프|스포츠", "아동": "아동|유아|키즈|완구",
        "명품": "명품|해외|수입", "가정": "가정용품|주방|생활|침구"}


def _raw_profile(train_ids):
    use = ["custid", "part_nm", "pc_nm", "corner_nm", "net_amt"]
    tr = pd.read_csv(C.TRAIN_CSV, encoding=C.ENCODING, usecols=use)
    cat = tr["part_nm"].astype(str) + " " + tr["pc_nm"].astype(str) + " " + tr["corner_nm"].astype(str)
    w = tr["net_amt"].clip(lower=0)
    g = tr.groupby("custid")
    prof = pd.DataFrame(index=train_ids)
    prof["n_txn"] = g.size().reindex(train_ids)
    prof["amt"] = g["net_amt"].sum().reindex(train_ids)
    tot = w.groupby(tr["custid"]).sum().reindex(train_ids) + 1.0
    for s, kw in SEGS.items():
        prof[s] = ((w * cat.str.contains(kw, regex=True)).groupby(tr["custid"]).sum().reindex(train_ids) / tot)
    return prof


def main():
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    y = pd.read_csv(C.YTRAIN_CSV).set_index(C.ID_COL).reindex(train_ids)[C.TARGET].to_numpy()
    preds = {m: (np.load(f"artifacts/oof/{m}__oof.npy") >= 0.5).astype(int) for m in MODELS}
    correct = {m: (preds[m] == y) for m in MODELS}
    cx, cc, cl = correct["first_xgb"], correct["first_cat"], correct["first_lgbm"]
    ncorr = cx.astype(int) + cc.astype(int) + cl.astype(int)

    groups = {
        "전부정답(3)": ncorr == 3,
        "2개정답": ncorr == 2,
        "xgb만정답": cx & ~cc & ~cl,
        "cat만정답": ~cx & cc & ~cl,
        "lgbm만정답": ~cx & ~cc & cl,
        "전부오답(0)": ncorr == 0,
    }
    prof = _raw_profile(train_ids)
    N = len(y)
    print(f"전체 {N}명, y=1비율 {y.mean():.3f}\n")
    print(f"{'그룹':14s} {'수':>6s} {'%':>6s} {'y=1':>6s} {'거래':>6s} {'금액(만)':>8s} | " +
          " ".join(f"{s:>5s}" for s in SEGS))
    for name, mask in groups.items():
        n = mask.sum()
        if n == 0:
            print(f"{name:14s} {n:>6d}"); continue
        row = f"{name:14s} {n:>6d} {n/N*100:>5.1f}% {y[mask].mean():>6.3f} " \
              f"{prof['n_txn'][mask].mean():>6.0f} {prof['amt'][mask].mean()/1e4:>8.0f} | " + \
              " ".join(f"{prof[s][mask].mean():>5.3f}" for s in SEGS)
        print(row)

    # 핵심: '정확히 1개만 정답' = 희귀 직교신호 고객 (다음 점수원 후보)
    exactly1 = (ncorr == 1)
    print(f"\n=== '정확히 1모델만 정답' 총 {exactly1.sum()}명 ({exactly1.mean()*100:.1f}%) ===")
    print("이 고객들 = 한 모델만 잡는 희귀신호. 이걸 더 잡는 멤버가 다음 점수원.")
    print(f"공통오답(전부틀림) {groups['전부오답(0)'].sum()}명 = 이미 분석함(흡수됨/어려움).")
    # 각 단독정답 그룹의 차별 특징 (전체평균 대비)
    print("\n=== 단독정답 그룹 특징 (전체평균 대비 배수, >1.3 또는 <0.7만) ===")
    base = {s: prof[s].mean() for s in SEGS}
    base["거래"] = prof["n_txn"].mean(); base["금액"] = prof["amt"].mean()
    for name in ["xgb만정답", "cat만정답", "lgbm만정답"]:
        mask = groups[name]
        feats = {**{s: prof[s][mask].mean() for s in SEGS},
                 "거래": prof["n_txn"][mask].mean(), "금액": prof["amt"][mask].mean()}
        diff = {k: feats[k] / (base[k] + 1e-9) for k in feats}
        hot = {k: f"{v:.2f}x" for k, v in sorted(diff.items(), key=lambda x: -x[1]) if v > 1.3 or v < 0.7}
        print(f"  {name} (y1={y[mask].mean():.2f}): {hot}")


if __name__ == "__main__":
    main()
