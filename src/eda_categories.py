"""카테고리별 성별 비율/빈도 — Day0 심화 EDA + 파생변수 v2(의미그룹) 설계용.

각 카테고리 컬럼의 값마다: gender 평균(거래수 가중) / 거래수 / 고유고객수 → 정렬해 출력·저장.
→ "어떤 part_nm/team_nm이 어느 성별로 기우는지" 보고 의미그룹(화장품/아동/식품 등) 매핑.

실행(로컬 anaconda):
  cd <프로젝트>
  & "C:\\ProgramData\\anaconda3\\python.exe" -m src.eda_categories
"""
import pandas as pd
from . import config as C

CAT_COLS = ["team_nm", "part_nm", "pc_nm", "corner_nm", "brd_nm"]
MIN_TXN = 200   # 너무 희소한 카테고리는 제외(노이즈)
EDA_DIR = C.ARTIFACTS / "eda"


def main():
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    usecols = [C.ID_COL] + CAT_COLS
    df = pd.read_csv(C.TRAIN_CSV, encoding=C.ENCODING, usecols=usecols)
    y = pd.read_csv(C.YTRAIN_CSV)
    df = df.merge(y, on=C.ID_COL, how="inner")

    g = df[C.TARGET].mean()
    print(f"전체 gender(=positive) 평균: {g:.4f} · 거래 {len(df):,} · 고객 {df[C.ID_COL].nunique():,}")

    for col in CAT_COLS:
        grp = df.groupby(col).agg(
            rate=(C.TARGET, "mean"),
            txn=(C.TARGET, "size"),
            cust=(C.ID_COL, "nunique"),
        )
        grp = grp[grp["txn"] >= MIN_TXN].sort_values("rate")
        grp.round(4).to_csv(EDA_DIR / f"{col}_gender_rate.csv", encoding="utf-8-sig")

        print(f"\n===== {col}  (전체 {df[col].nunique()}개 / txn>={MIN_TXN}: {len(grp)}개) =====")
        print("  ▼ positive 낮은 TOP 15")
        print(grp.head(15).round(3).to_string())
        print("  ▲ positive 높은 TOP 15")
        print(grp.tail(15).round(3).to_string())

    print(f"\n[저장] 전체 표 → {EDA_DIR}")


if __name__ == "__main__":
    main()
