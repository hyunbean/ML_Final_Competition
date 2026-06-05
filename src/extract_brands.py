"""브랜드 목록 추출 (외부 메타데이터 라벨링용). 빈도순 정렬.

출력: artifacts/brands_to_label.csv (brd_nm, n_txn, n_cust, top_part)
→ 이 목록(특히 상위 빈도 브랜드)을 LLM에 줘서 성별/연령/가격대 라벨을 만들고,
  data/brand_meta.csv 로 저장하면 build_features가 자동으로 피처에 합침.

실행: python -m src.extract_brands
"""
import pandas as pd
from . import config as C


def main():
    cols = [C.ID_COL, "brd_nm", "part_nm"]
    tr = pd.read_csv(C.TRAIN_CSV, encoding=C.ENCODING, usecols=cols)
    te = pd.read_csv(C.TEST_CSV, encoding=C.ENCODING, usecols=cols)
    df = pd.concat([tr, te], ignore_index=True)
    df["brd_nm"] = df["brd_nm"].astype(str)

    g = df.groupby("brd_nm").agg(
        n_txn=("brd_nm", "size"),
        n_cust=(C.ID_COL, "nunique"),
        top_part=("part_nm", lambda s: s.mode().iat[0] if len(s.mode()) else ""),
    ).sort_values("n_txn", ascending=False).reset_index()

    out = C.ARTIFACTS / "brands_to_label.csv"
    g.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"saved: {out}  ({len(g)} brands)")
    print(f"상위 {min(30,len(g))}개:")
    print(g.head(30).to_string())
    cum = g["n_txn"].cumsum() / g["n_txn"].sum()
    n300 = (cum <= 0.95).sum()
    print(f"\n상위 {n300}개 브랜드가 전체 거래의 95% 커버 → 이만큼만 라벨링해도 충분")


if __name__ == "__main__":
    main()
