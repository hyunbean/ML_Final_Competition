# KML Challenge 2026S — 현대백화점 고객 성별 예측

> 백화점 거래 로그 103만 건을 고객(custid) 단위로 집계해 **고객 성별을 예측하는 이진 분류 대회 (ROC-AUC)**.
> 4인 팀 참가, **10팀 중 Public 1위 → Private 2위**. 순위가 뒤집힌 원인(public-overfit)까지 정량 분석한 회고가 이 repo의 핵심 자산입니다.

## 📊 결과

| | Public | Private | 최종 순위 |
|---|---|---|---|
| **우리 팀** | **0.7385 (1위)** | 0.7324 | **2위 / 10팀** |
| 1위 팀 | 0.7379 | 0.7345 | 1위 |

- 개별 모델 천장이 0.71~0.72인 대회에서 **4단계 앙상블로 0.738까지 +0.027**을 만들었습니다.
- 그러나 public 리더보드에 과적합된 구조(갭–public 상관 +0.63)로 private에서 역전당했고,
  **왜 졌는지를 1·3위 솔루션과 대조 분석**한 문서를 남겼습니다 → [`최종발표+분석/KML2026S_회고_1위.md`](최종발표+분석/KML2026S_회고_1위.md) (섹션 17~19)

## 🙋 내 역할 (신현빈, @hyunbean)

이 repo의 커밋 356건 중 대부분이 제 작업이며, 다음을 담당했습니다.

- **핵심 파이프라인 설계·구현**: 521개 집계 피처 생성(`features.py`), 고정 fold·rank 정규화·custid 정렬 규약(`config.py`, `folds.py`), 팀 공유 OOF 입출력 포맷(`oof_io.py`, 코드 전체 97곳에서 사용)
- **Pseudo-labeling 파이프라인**: teacher-student 격리, fold 수를 variance 레버로 활용해 public 5위 → 1위 돌파(`train_pseudo_strict.py`)
- **블렌딩·메타스택 운영**: rank 가중 블렌딩(`blend_rank.py`), 3-layer 메타스택(`stack_make73.py`, `blend_stack3.py`), corr 기반 final 2슬롯(공격/헤지) 선정
- **실험 운영·기록**: 모든 실험을 "가설 → CV → corr → LB → 즉시 기록" 루프로 관리, 커밋 메시지에 실험 의미를 남겨 git을 실험 이력 추적기로 활용
- **팀 협업 인프라**: OOF npy를 git으로 교환하는 워크플로우(`scripts/push_oof.sh`), 팀원 온보딩 문서(`setup/README_TEAM.md`)

팀 분담: 신현빈(파이프라인·pseudo·블렌드·제출 전략) / 김민형(base 메타스택 재구축·RunPod 학습) / 조아인(전처리) / 이두영(모델링). AI 페어프로그래밍을 실험 운영에 적극 활용했으며, 그 과정에서 얻은 **AI 협업 운영 노하우**(체크리스트 관리, 할루시네이션 방지, 교차검증)도 회고 섹션 9·13에 문서화했습니다.

## 🏗 앙상블 아키텍처 (개별 0.71 → 최종 0.738)

| 단계 | 구조 | 게인 |
|---|---|---|
| 0. base 멤버 ~120개 | 521 피처 × XGB/LGBM/CatBoost/AutoGluon/Optuna 튜닝 등 | 단일 0.713~0.719 |
| 1. L1 메타스택 | 결합 방식이 다른 메타러너 7종(LR·Ridge·ET·HGB·KNN·hillclimb)으로 5-fold 스택 | — |
| 2. L2 hillclimb | L1 출력 8개를 greedy 결합 | LB 0.7347 |
| 3. base 10-fold 교체 | 재폴드 가능한 27개 멤버를 10-fold판으로 교체 | +0.0015 |
| 4. pseudo 블렌드 | `rank(0.70·rank(base) + 0.30·rank(pseudo_50fold))` | **LB 0.7383** |

**기술 스택**: Python, pandas, scikit-learn, LightGBM, XGBoost, CatBoost, AutoGluon, Optuna, gensim(Word2Vec/item2vec)

## 📁 구조

```
최종발표+분석/         ★ 회고·발표·분석 (이 repo에서 가장 중요)
  ├ KML2026S_회고_1위.md    회고: 협업·돌파·검증·앙상블 구조·shake-up 분석·다음 대회 청사진
  ├ ML실험노트_초안.md      초반 EDA·베이스라인 노트
  ├ 실험노트_템플릿.md      일일 실험노트 템플릿 (다음 대회 재사용)
  └ 2등(우리팀)_KML성별예측_발표.html   팀 발표자료
setup/                 환경·실행 설정 (Colab/DLPC 셋업, 파이프라인, 팀 온보딩)
src/                   핵심 코드 (→ src/README.md 에 모듈별 안내)
experiments_archive/   실패/폐기 실험 (회고 근거용 보관 — 음수 결과도 기록)
artifacts/             고정 folds·custid 정렬 npy, 팀 공유 OOF 예측
```

## ⚙️ 재현 메모

- 데이터(`*.csv`)는 git 제외 — 각 머신에 배치 후 `KML_DATA` 환경변수로 경로 지정 (인코딩 cp949).
- 모든 예측은 **rank 정규화 + 고정 folds(seed 42) + custid 정렬 통일** (스태킹 누수 방지 규약).
- 재현 순서: `python -m src.folds` → `src.features` → 각 `train_*` → `src.blend_stack3`. 상세는 [`setup/PIPELINE.md`](setup/PIPELINE.md).

## 💡 이 대회에서 배운 것 (요약)

1. **Public 1위 ≠ 승리.** 우리는 갭–public 상관 +0.63(과적합형), 우승팀은 −0.19(robust형). 같은 public 점수대에서 갭 차이 0.003이 그대로 순위를 뒤집었다.
2. **제출 수는 과적합 지표.** 우리 140제출 vs private 3위 팀 47제출. public LB를 validation처럼 쓰면 갭이 커진다.
3. **CV−LB 갭 추이를 그려라.** 갭이 일정하면 robust, 벌어지면 public 튜닝 중단 신호.
4. **"천장"은 거의 항상 가설.** 며칠간 데이터 천장으로 단정했던 0.7354는 fold-5 variance 한계였다 — 안 건드린 축(fold/seed/앙상블 구조)부터 의심할 것.
5. **교훈에는 적용 범위가 있다.** pseudo 회피·pruning 등은 "작고 분포차 큰 이 대회"의 조건부 처방 — 다음 대회는 adversarial validation으로 분포차부터 확인 후 분기 (회고 섹션 19).
