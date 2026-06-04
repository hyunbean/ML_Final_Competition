# ML 실험노트 — KML Challenge 2026S (고객 성별 예측)

> 백화점 거래내역(transaction) → 고객(custid) 단위 집계 → 성별(gender) 이진분류
> 평가지표: **ROC-AUC** · 제출: positive class 확률

## 0. 데이터 개요
- train_transactions: 1,036,653 거래 / 30,000 고객 (1인 평균 ~35건)
- test_transactions: 689,777 거래 / 20,000 고객
- y_train: custid, gender (0/1)
- 인코딩: **CP949(euc-kr)** — `pd.read_csv(..., encoding='cp949')`
- 컬럼(16): custid, sales_datetime, str_nm, goodcd, brd_nm, corner_nm, pc_nm, part_nm, team_nm, buyer_nm, import_flg, tot_amt, dis_amt, net_amt, inst_mon, inst_fee

## 1. EDA 핵심 발견
| 항목 | 결과 |
|---|---|
| 클래스 분포 | gender 0 = 69.6% / **1 = 30.4% (소수)** |
| 결측치 | 0 (깨끗) |
| 거래 수(중앙값) | g0=20 / **g1=26** |
| 총구매액(중앙값) | g0=155만 / **g1=220만** |
| 사용 브랜드 수(중앙값) | g0=13 / g1=15 |
| 주말 비율 | 차이 없음(~0.40) → 변별력 낮음 |

**카테고리 신호(기준선 g1=0.364)** — 가장 강한 변별축
- g0 편향: 영캐릭터(0.22)·영플라자(0.24)·패션잡화(0.25)·여성캐주얼(0.29), 브랜드 비너스스타킹/바디/TBJ
- g1 편향: 남성정장스포츠(0.51)·아동복(0.52)·문화용품(0.52)·생식품(0.50), 브랜드 노스페이스/레고/베베
- 핵심 피처축: **part_nm, pc_nm, brd_nm** (team_nm은 3종뿐 → 약함)

**시간대 신호**: 오전·점심(10~11시) g1≈0.43 → 저녁(18~20시) g1≈0.28. 저녁구매=g0 성향. 월별 차이는 미미.

## 2. 베이스라인 (starter_code.ipynb)
- Manual feature: 집계 통계 + team/part/season/time/store 비중(crosstab) + 집중도
- Word2Vec: corner_nm 시퀀스 → 64d 임베딩 평균
- 모델: AutoGluon TabularPredictor (binary, roc_auc, good_quality, 300s)
- TODO: 베이스라인 점수 측정(미실행)

## 3. 1등 솔루션에서 가져올 개선안 (Top 5)
1. **브랜드 외부 메타데이터** (Gemini API): 브랜드별 타깃연령/가격대/성별타깃 → 강한 신호
2. **Adaptive-Sigma Target Encoding**: 카테고리 희소도에 따라 정규화 강도 자동조정
3. **Shannon 엔트로피 피처**: 브랜드/카테고리/시간대 구매 다양성 정량화
4. **Hill-Climbing 앙상블**: 단순평균 대비 AUC↑ (LGBM 0.43/CatBoost 0.22/Linear 0.21)
5. **GroupKFold Target Encoding**: 고객 단위 fold 분리로 누수 방지

**1등 성능 참고**: 단일 LGBM AUC≈0.714 → Hill-Climbing 앙상블 AUC≈0.718

## 4. 실험 로그
| # | 날짜 | 변경/피처 | 모델 | CV(AUC) | LB | 메모 |
|---|---|---|---|---|---|---|
| 0 | 2026-06-04 | EDA + 1등 분석 | - | - | - | 데이터/접근법 파악 |
| 1 |  | (베이스라인 실행) |  |  |  |  |

## 5. 다음 액션
- [ ] starter 베이스라인 실행 → 기준 점수 확보
- [ ] part_nm/pc_nm/brd_nm 타깃인코딩(GroupKFold) 추가
- [ ] 엔트로피·다양성 피처 추가
- [ ] 시간대(저녁비율) 피처 강화
- [ ] LGBM/CatBoost/XGB 앙상블(Hill-Climbing)
