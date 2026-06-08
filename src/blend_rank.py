"""실제 제출 블렌드(#78~99 + ms/plb) 재현 — rank 가중평균. 누수 literal 대조용.

⚠️ 이 제출들은 원래 인라인 python으로 만들어 git에 코드가 없었음. 세션 트랜스크립트에서
   실제 실행 명령을 추출·정리해 1:1 재현 가능하게 박아둔 파일(2026-06-08, 신현빈).

모든 제출은 **동일 패턴**:
    R(m)   = rankdata(test_npy[m] 를 73-custid 순서로 reindex) / N      # 멤버 test rank
    group  = 멤버 rank들의 평균 (예: plXL = (xgb_pl2 + lgbm_pl2) rank평균)
    blend  = Σ wᵢ · R(sourceᵢ)                                          # 가중합
    최종    = rankdata(blend)/N                                          # 다시 rank-normalize

73 base = 김민형 외부 파일(73_giftstack_sig1_minhyung.csv) — OOF가 없으므로
누수검증의 OOF-CV는 mh_bestblend69(73 프록시, corr 0.996) OOF로 대체 계산.

블렌드 자체엔 누수 경로 없음(각 멤버 OOF는 이미 fold-safe, 단순 rank 가중평균).
누수가 있다면 멤버 OOF 생성(pseudo) 단계 → src/train_pseudo_strict.py 참고.

실행:  python -m src.blend_rank [제출명|all]
       F73=<경로> 로 73 파일 위치 지정 가능 (기본 Downloads).
       OUT=<dir>  로 출력 폴더 지정 (기본 artifacts/submissions).
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from . import config as C

TR = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
TE = np.load(C.TEST_IDS_NPY, allow_pickle=True)
F73 = os.environ.get("F73", "C:/Users/guszh/Downloads/73_giftstack_sig1_minhyung.csv")
OUT = os.environ.get("OUT", "artifacts/submissions")
OOF73_PROXY = "mh_bestblend69"   # 73 외부파일은 OOF없음 → 누수검증 프록시(corr 0.996)

# 멤버 그룹(여러 멤버의 rank 평균을 하나의 source로 취급)
GROUPS = {
    "plXL":   ["first_xgb_pl2", "first_lgbm_pl2"],                                    # hard pseudo XL (핵심)
    "pens":   ["first_xgb_pl2", "first_lgbm_pl2", "first_cat_pl2"],                   # 3모델 pseudo 앙상블
    "plXLc":  ["first_xgb_plc", "first_lgbm_plc"],                                    # consensus pseudo
    "combo":  ["first_xgb_pl2", "first_lgbm_pl2", "first_xgb_plc", "first_lgbm_plc"], # pl2+plc 4종
    "softXL": ["first_xgb_pls", "first_lgbm_pl2"],                                    # soft xgb + hard lgbm
    "softall":["first_xgb_pls", "first_xgb_pl2", "first_lgbm_pl2"],
    "cwXL":   ["first_xgb_plcw", "first_lgbm_pl2"],                                   # confidence-weight pseudo
    "cw3":    ["first_xgb_plcw", "first_xgb_pl2", "first_lgbm_pl2"],
    "fsXL":   ["first_xgb_fs", "first_lgbm_fs"],                                      # 교수힌트 feature-select
    "pleXL":  ["first_xgb_ple_pl2", "first_lgbm_ple_pl2"],                            # pseudo+임베딩
    "msXL":   ["first_xgb_pl2_ms", "first_lgbm_pl2_ms"],                              # 멀티시드 pseudo
    "plbXL":  ["first_xgb_plb", "first_lgbm_plb"],                                    # 강한 teacher pseudo
}

# 제출명 → [(source, weight), ...].  source = 'EXT73' | GROUPS키 | 단일 멤버명
RECIPES = {
    # --- 78 gamble: 73 + xgb_pl2 단독 (lgbm 나오기 전 첫 베팅) ---
    "78_gamble_pl2w20": [("EXT73", 0.80), ("first_xgb_pl2", 0.20)],
    "78_gamble_pl2w30": [("EXT73", 0.70), ("first_xgb_pl2", 0.30)],
    "78_gamble_pl2w40": [("EXT73", 0.60), ("first_xgb_pl2", 0.40)],
    # --- 80 pseudoens: 73 + 3모델 pseudo 앙상블 ---
    "80_pseudoens_w20": [("EXT73", 0.80), ("pens", 0.20)],
    "80_pseudoens_w25": [("EXT73", 0.75), ("pens", 0.25)],
    "80_pseudoens_w30": [("EXT73", 0.70), ("pens", 0.30)],
    "80_pseudoens_w35": [("EXT73", 0.65), ("pens", 0.35)],
    # --- 81/82 pseudoXL: 73 + (xgb_pl2+lgbm_pl2)/2.  cat 제외(약함). ---
    "81_pseudoXL_w15": [("EXT73", 0.85), ("plXL", 0.15)],
    "81_pseudoXL_w20": [("EXT73", 0.80), ("plXL", 0.20)],
    "81_pseudoXL_w25": [("EXT73", 0.75), ("plXL", 0.25)],
    "81_pseudoXL_w30": [("EXT73", 0.70), ("plXL", 0.30)],   # ⭐ = 제출92 = 팀 최고 LB 0.73540
    "81_pseudoXL_w35": [("EXT73", 0.65), ("plXL", 0.35)],
    "82_pseudoXL2_w30": [("EXT73", 0.70), ("plXL", 0.30)],  # pl2 재학습 후(멤버 동일) → 81_w30과 같은 레시피
    # --- 83 consensus pseudo / 84 combo ---
    "83_pseudoXLc_w30": [("EXT73", 0.70), ("plXLc", 0.30)],
    "84_pseudoCOMBO_w30": [("EXT73", 0.70), ("combo", 0.30)],
    "84_pseudoCOMBO_w35": [("EXT73", 0.65), ("combo", 0.35)],
    # --- 85 uncertainty 피처 추가 ---
    "85_best_plus_unc8":  [("EXT73", 0.62), ("plXL", 0.30), ("first_xgb_unc", 0.08)],
    "85_best_plus_unc12": [("EXT73", 0.58), ("plXL", 0.30), ("first_xgb_unc", 0.12)],
    "85_73_unc15":        [("EXT73", 0.85), ("first_xgb_unc", 0.15)],
    # --- 86 soft pseudo / 87 confidence-weight pseudo ---
    "86_softXL_w30":  [("EXT73", 0.70), ("softXL", 0.30)],
    "86_softall_w30": [("EXT73", 0.70), ("softall", 0.30)],
    "87_cwXL_w30": [("EXT73", 0.70), ("cwXL", 0.30)],
    "87_cw3_w30":  [("EXT73", 0.70), ("cw3", 0.30)],
    # --- 88/89 73 + pseudo + feature-select 삼각 ---
    "88_73_pl_fs":  [("EXT73", 0.62), ("plXL", 0.22), ("fsXL", 0.16)],
    "88_73_fs_w20": [("EXT73", 0.80), ("fsXL", 0.20)],
    "88_73_pl_fs2": [("EXT73", 0.58), ("plXL", 0.24), ("fsXL", 0.18)],
    "89_73_plfs_even": [("EXT73", 0.60), ("plXL", 0.20), ("fsXL", 0.20)],
    # --- 98 clean (73 + pseudo + 소량 fs) / 99 pseudo+임베딩 ---
    "98_clean_fs10": [("EXT73", 0.65), ("plXL", 0.25), ("fsXL", 0.10)],
    "98_clean_fs15": [("EXT73", 0.60), ("plXL", 0.25), ("fsXL", 0.15)],
    "99_ple_w25": [("EXT73", 0.75), ("pleXL", 0.25)],
    "99_ple_w30": [("EXT73", 0.70), ("pleXL", 0.30)],
    # --- 마지막 레버(6/8): 멀티시드 / 강한 teacher pseudo ---
    "ms_pl2_XL_w30": [("EXT73", 0.70), ("msXL", 0.30)],
    "plb_XL_w30":    [("EXT73", 0.70), ("plbXL", 0.30)],
}


def _rk(a):
    return rankdata(a) / len(a)


def _ext_test(idx):
    d = pd.read_csv(F73).set_index("custid")
    return _rk(d[d.columns[0]].reindex(idx).to_numpy())


def _Rt(m, idx):   # 멤버 test rank (73 custid 순서)
    return _rk(pd.Series(np.load(f"artifacts/oof/{m}__test.npy"), index=TE).reindex(idx).to_numpy())


def _Ro(m):        # 멤버 oof rank (train 순서)
    return _rk(np.load(f"artifacts/oof/{m}__oof.npy"))


def _src_test(src, idx):
    if src == "EXT73":
        return _ext_test(idx)
    if src in GROUPS:
        return np.mean([_Rt(m, idx) for m in GROUPS[src]], axis=0)
    return _Rt(src, idx)


def _src_oof(src):
    if src == "EXT73":
        return _Ro(OOF73_PROXY)            # 73 프록시
    if src in GROUPS:
        return np.mean([_Ro(m) for m in GROUPS[src]], axis=0)
    return _Ro(src)


def build_test(name, idx):
    return _rk(sum(w * _src_test(s, idx) for s, w in RECIPES[name]))


def oof_cv(name, y):
    """73→bestblend69 프록시로 블렌드 OOF-CV 계산(누수 갭테스트용). 멤버 OOF 없으면 None."""
    try:
        b = sum(w * _src_oof(s) for s, w in RECIPES[name])
    except FileNotFoundError:
        return None
    return float(roc_auc_score(y, b))


def _load_y():
    for p in ["y_train.csv", "data/y_train.csv", "artifacts/y_train.csv"]:
        if os.path.exists(p):
            return pd.read_csv(p).set_index("custid").reindex(TR).iloc[:, 0].to_numpy()
    return None


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    names = list(RECIPES) if which == "all" else [which]
    idx = pd.read_csv(F73).set_index("custid").index   # 73 custid 순서 = 제출 순서
    y = _load_y()
    os.makedirs(OUT, exist_ok=True)
    for nm in names:
        b = build_test(nm, idx)
        out = os.path.join(OUT, f"{nm}_hyunbin.csv")
        pd.DataFrame({"custid": idx, "gender": b}).to_csv(out, index=False)
        cv = oof_cv(nm, y) if y is not None else None
        cvs = f"  OOF-CV(73프록시)={cv:.5f}" if cv is not None else "  (멤버 OOF 일부 없음)"
        print(f"{nm:22s} -> {out}{cvs}")


if __name__ == "__main__":
    main()
