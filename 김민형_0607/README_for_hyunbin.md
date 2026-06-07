# 📦 김민형 → 신현빈 핸드오프 (2026-06-07)

> 네 요청 4개 대응. **전부 [custid, pred] 2컬럼 csv, 네 `artifacts/folds.npy`로 생성 → OOF 직접 스태킹 가능.**
> train OOF 30000행 / test 19995행. custid로 정렬 맞추면 됨.

## 파일 목록 (_oof.csv = train OOF / _test.csv = test 예측)

| 모델 | CV(ROC-AUC) | corr_stack3 | 정체 / 가치 |
|---|---|---|---|
| **mh_bestblend69** | 0.72568 | 0.997 | 내 최강 leak-free 블렌드. ⚠️ corr 0.997 = 네 앵커로 만든 거라 **너한텐 중복**, 우선순위 낮음 |
| **mh_temporal_lgb** | 0.610 | **0.519** ⭐ | 시간행동만(언제/얼마나 자주/할부). **프로젝트 최저 corr** = 천장 뚫기 후보 1순위 |
| **mh_textsvd_lgb** | 0.664 | 0.800 | 카테고리 TF-IDF bigram→SVD256 (텍스트축 직교) |
| **mh_hypothesis_lgb** | 0.683 | 0.831 | 도메인가설 65피처(net_lean·남녀충돌·할인참여·환불) |

## 네 질문 4개 답

1. **최강 블렌드** = `mh_bestblend69` (단 corr 0.997, 너한텐 중복일 듯).
2. **새 단일모델** = temporal / textsvd / hypothesis (05·06·07·09·11 이후 신규). **이게 진짜 가치** — 약하지만 corr 0.52~0.83로 직교. interact처럼 스택서 살 가능성.
3. **CV + folds**: 내 원래 folds(seed2026) ≠ 네 folds.npy (일치율 0.20). **→ 위 파일은 전부 네 folds.npy로 재생성했으니 OOF 직접 스태킹 OK.** (재생성 안 했으면 test rank-avg만 가능했음)
4. **포맷**: [custid, pred] csv ✓

## 추천 사용법
- hillclimb/Ridge 스택 풀에 **temporal(0.52)·textsvd(0.80)** 우선 투입 → 직교 기여 테스트.
- `interact`가 +0.0004 줬듯, 저상관 멤버가 살 수 있음. 안 살면 버려도 됨(weight 0).
- bestblend69는 중복이라 굳이 안 써도 됨.

## ⚠️ 누수 주의 (내가 겪은 것 = 네가 distill 뺀 이유와 동일)
- **KD/distill on OOF = 항상 누수**(OOF 신기루 0.738). 네 커밋 `distill_lgbm 제거` 맞는 판단.
- 선형 메타(logreg/ridge)를 90개 상관 OOF에 풀로 돌리면 split-overfit으로 OOF 0.738 뜸 → **hillclimb 메타만 신뢰**(0.726대). LB가 진실.
