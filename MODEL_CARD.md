# Model Card — KML Challenge 2026S 고객 성별 예측 모델

> 이 문서는 이 repo가 구현한 모델(백화점 거래 로그 기반 고객 성별 이진분류)의 용도·데이터·성능·한계·금지 용도를 명시합니다.
> 상세 배경은 [`README.md`](README.md), [`docs/retrospective.md`](docs/retrospective.md), [`docs/eda-baseline-notes.md`](docs/eda-baseline-notes.md) 참고.

## 용도 (Intended Use)

- **과제 정의**: 백화점 거래 로그(transaction log)를 고객(`custid`) 단위로 집계한 피처로부터, 해당 고객의 **성별(0/1)을 예측**하는 이진 분류 모델입니다.
- **평가지표**: ROC-AUC (순위 기반 지표). 제출값은 positive class(남성, label 1) 확률/rank 정규화 점수입니다.
- **구현 범위**: 이 repo는 **대학원 수업(KML Challenge 2026S) 대회 제출물**로 구현되었습니다. 521개 집계 피처(`src/features.py`) → 약 120개 base 모델(XGBoost/LightGBM/CatBoost/AutoGluon/Optuna 튜닝 등) → 3-layer 메타스택(`src/stack_make73.py`, `src/blend_stack3.py`) → pseudo-labeling 블렌드(`src/train_pseudo_strict.py`) → rank 가중 최종 블렌드(`src/blend_rank.py`)로 이어지는 파이프라인입니다.
- **의도된 사용 맥락**: 대회 제출·포트폴리오·회고(모델링·앙상블·검증 방법론) 목적입니다. **실제 서비스나 의사결정 시스템에 배포하기 위해 만들어진 것이 아닙니다.**

## 학습 데이터 (Training Data)

`docs/eda-baseline-notes.md`, `docs/retrospective.md`(섹션 0)에 기록된 내용 기준:

| 항목 | 값 |
|---|---|
| 데이터 종류 | 백화점(현대백화점) 고객 거래 로그(transaction-level) |
| train 규모 | 거래 1,036,653건 / 고객(`custid`) 30,000명 (1인 평균 약 35건) |
| test 규모 | 거래 689,777건 / 고객 약 20,000명 (README 기준 최종 평가셋 19,995명) |
| 라벨 | `custid` 단위 `gender` (0/1) — **남성이 positive class(1)**, positive rate 약 0.304 (약 3:7 불균형) |
| 원본 컬럼(16개) | `custid, sales_datetime, str_nm, goodcd, brd_nm, corner_nm, pc_nm, part_nm, team_nm, buyer_nm, import_flg, tot_amt, dis_amt, net_amt, inst_mon, inst_fee` |
| 파생 피처 | 위 거래 로그를 고객 단위로 집계한 521개 피처(구매 카테고리·브랜드·시간대·금액 분포 등) |
| 인코딩 | CP949(EUC-KR) |
| 결측치 | 없음 (EDA 기준) |

**주의**: 신용(credit) 관련 필드(소득, 신용등급, 연체 이력 등)는 이 데이터셋에 **존재하지 않습니다**. 이 모델은 신용평가 모델이 아니라 **소매 구매 행동 → 성별 추정** 모델입니다.

## 성능 (Performance)

README와 `docs/retrospective.md`에 기록된 실측값 기준(리더보드 제출 점수):

| | Public LB | Private LB | 순위 |
|---|---|---|---|
| **본 팀 최종 제출** | **0.7385141 (1위)** | **0.7324252 (2위)** | 10팀 중 2위 |
| 1위 팀(참고) | 0.7379 | 0.7345426 (1위) | 1위 |

- 개별 base 모델 단독 성능은 ROC-AUC **0.71~0.72대**(천장, `docs/retrospective.md` 섹션 0)이며, 최종 제출 점수는 앙상블·스태킹·pseudo-labeling을 거친 결과입니다(README `앙상블 아키텍처` 표 참고).
- Public/Private 갭에 대한 정량 분석(과적합 진단)은 `docs/shakeup-analysis.md`와 `docs/retrospective.md` 섹션 17~19에 기록되어 있습니다 — 자체 데이터에 대한 갭 분석이지 신용평가 표준 지표는 아닙니다.

**신용평가 표준 지표 `[사후측정 2026-07-28, 본 레포 재현]`**

이 repo 코드 자체에는 KS/Gini/PSI/lift 계산이 구현되어 있지 않습니다. 아래 값은 `artifacts/oof/` 안의 저장된 OOF·test 예측(`first_xgb_pl2` = 단일 최고 base, `stack73full_L2_logreg` = 최종 L2 메타스택 — 두 파일 모두 `meta.json`의 기록된 `cv_auc`가 각각 0.72057 / 0.73402로, `docs/retrospective.md`의 D8 수치와 정확히 일치해 식별)를 `artifacts/train_custids.npy` 순서로 정렬하고, `C:\Users\guszh\kml_cust_agg.parquet`(30,000행, `custid` 인덱스 0~29999, `gender` 컬럼, positive rate 0.30393 — README의 0.304와 일치)의 라벨과 직접 대조해 이번 세션에서 **직접 재계산**했습니다. 재계산한 AUC(0.7205681491624072 / 0.7340207913489346)가 각 `meta.json`에 이미 기록돼 있던 `cv_auc`와 정렬 오차 없이 일치함을 먼저 확인한 뒤(sanity check), 같은 정렬로 Gini(=2·AUC−1)·KS·decile lift·PSI를 산출했습니다.

| 지표 | base(`first_xgb_pl2`) | 스택(`stack73full_L2_logreg`) | 기준선 | 판정 |
|---|---|---|---|---|
| Gini (AR) | 0.4411 | 0.4680 | 0.6 이상 = 우수 | 기준 미달 (우수 등급 아님) |
| KS | 31.62 | 33.30 | 20 이상 = 통과 | 통과 |
| Decile lift (상위 10%) | 2.16배 (정밀도 65.6%) | 2.19배 (정밀도 66.7%) | — | — |
| PSI (train-OOF 분포 vs test 예측 분포) | 0.0444 | 0.0004 | — | 둘 다 안정 구간 |

- 위 값은 career-ops 저장소 `data/practice-lens-audit.md`(D③·D7b)에 기록된 값과 일치합니다 — 다만 이 표는 그 문서를 베낀 것이 아니라, 이 repo에 실제로 존재하는 OOF/test npy와 라벨 parquet을 이번 세션에서 직접 로드해 pandas/scikit-learn으로 **독립 재현**한 결과입니다. CV 폴드 정의는 `src/config.py`·`src/folds.py`(seed 42, 5-fold), OOF 저장 포맷은 `src/oof_io.py` 규약을 따릅니다.
- **주의**: 이 값들은 train 5-fold **OOF**(교차검증 내부 held-out) 기준이며, README의 public/private 리더보드 점수(0.7385141/0.7324252)는 실제 test 라벨에 대한 제출 점수입니다. 두 수치 체계는 동일한 모델 계열이지만 서로 다른 평가 시점의 값이므로 직접 등치하지 않도록 구분해 표기했습니다.

## 금지 용도 (Prohibited Uses) — 가장 중요

이 모델과 관련 코드는 **실제 대출/신용 심사, 여신 의사결정, 고객 세그멘테이션에 기반한 차별적 처우**에 사용되어서는 안 됩니다. 구체적으로:

1. **실서비스 신용/여신 스코어링에 그대로 배포 금지.** 이 모델은 학술 대회용으로 학습·검증되었으며, 프로덕션 신용평가 파이프라인이 요구하는 규제 대응(공정성 감사, 모델 거버넌스, 설명가능성, 지속적 모니터링/드리프트 관리)을 전혀 거치지 않았습니다.
2. **공정성 감사(fairness audit) 없이 실제 대출/여신/보험/채용 등 사람에게 영향을 주는 결정에 사용 금지.** 성별을 예측 대상(target)으로 직접 학습한 모델이므로, 이를 다른 도메인(신용점수, 리스크 등급 등)의 대리 변수(proxy)로 전용할 경우 **성별 프로파일링(gender profiling)** 위험이 매우 큽니다 — 즉 구매 행동만으로 성별을 추정한 뒤 이를 신용·리스크 판단에 연결하면, 성별을 직접 심사 기준으로 쓰는 것과 실질적으로 동일한 차별적 효과를 낼 수 있습니다.
3. **개인 식별/추적 목적 금지.** `custid` 등 고객 식별자와 결합해 특정 개인의 성별을 추정·공개·마케팅 타기팅에 사용하는 것은 이 모델의 의도된 용도가 아닙니다.
4. **어떤 형태로든 실사용 전에는 반드시**: (a) 대상 도메인 데이터로 재검증, (b) 그룹별(성별 포함) 공정성 지표 재계산, (c) 법률/컴플라이언스 검토, (d) 신용평가라면 KS/Gini/PSI/lift 등 업계 표준 지표의 신규 산출 및 지속 모니터링 체계 구축이 선행되어야 합니다. 이 repo는 그 어느 것도 제공하지 않습니다.

## 한계 (Limitations)

- **세그먼트별(subgroup) 공정성 지표 미계산.** `gender` 자체가 예측 대상(target)이고, 데이터에 다른 세그멘트 가능 속성(연령·지역 등 인구통계 필드)이 존재하지 않아(`docs/eda-baseline-notes.md` 컬럼 16개 목록 기준), 세그먼트별 성능 격차를 계산할 축이 이 repo 데이터에는 없습니다. 이는 공정성 검증이 "충분해서 생략된" 것이 아니라, **애초에 검증할 재료가 없는 상태**임을 의미하며, 실사용 전에는 별도의 인구통계 데이터를 확보해 반드시 채워야 할 공백입니다.
- **Public–Private 분포차(shake-up)가 큰 대회 데이터.** `docs/retrospective.md` 섹션 0·17~19에 따르면 이 대회는 public/private LB 간 상관 0.84, 갭이 커서(shake-up 심함) public 리더보드 점수만으로 일반화 성능을 신뢰하기 어렵다고 이미 자체 분석되어 있습니다. 즉 위 표의 Public 점수는 과적합되어 있을 가능성이 있고, Private 점수가 더 신뢰할 수 있는 값입니다.
- **작은 데이터, 과포화 피처.** train 30,000행에 피처 521개로, 개별 신호가 약하고(단일 모델 AUC 0.71~0.72) 앙상블 의존도가 매우 높습니다. 다른 규모/분포의 데이터에 이식 시 성능이 재현되지 않을 수 있습니다.
- **Gini(AR) 기준 미달.** 성능 섹션의 사후측정 결과, 최종 스택 기준 Gini 0.4680으로 금융권 통상 기준선(0.6 이상 "우수")에 못 미칩니다. KS(33.30)는 통상 기준(20 이상)을 통과하지만, "판별력이 실무 배포에 충분하다"는 근거로 쓰기에는 부족한 수준이며 — 실제로 이 값들은 대회용 순위 지표(AUC) 최적화의 부산물일 뿐, 신용·리스크 목적에 맞춰 재설계·검증된 것이 아닙니다.
- **캘리브레이션·임계값·비용-편익 분석 미실시.** Brier score/ECE 등 확률 보정 지표, 그리고 실사용에 필요한 임계값 결정 절차(비용-편익 매트릭스)는 이번 사후측정 범위 밖이며 이 repo에 없습니다.
- **재현성 전제 조건.** 고정 seed(42)·고정 5-fold·`custid` 정렬 규약(`src/config.py`, `src/folds.py`)에 의존하며, 원본 CSV 데이터는 git에 포함되어 있지 않습니다(`KML_DATA` 환경변수로 별도 배치 필요).
