# src/ 모듈 안내

80여 개 파일이 있지만 역할별로 아래 그룹으로 나뉩니다. 모든 모듈은 `config.py`의 고정 seed(42)·5-fold·경로 정의를 공유하고, 예측은 `oof_io.py` 포맷(rank 정규화 + custid 정렬)으로 저장해 서로 블렌딩 호환됩니다.

## 공통 기반 (모든 모듈의 기준)
| 파일 | 역할 |
|---|---|
| `config.py` | 경로(env로 덮어쓰기 가능)·시드·CV·컬럼 정의 |
| `data.py` / `folds.py` | 데이터 로드 / 고정 5-fold (StratifiedKFold seed 42) |
| `oof_io.py` | OOF·test 예측 저장/로드 — 팀 공유 포맷, 코드 전체 97곳에서 사용 |
| `features.py` | 521개 집계 피처 생성 (`build_all`) |
| `checkpoint.py` / `make_submission.py` | 학습 체크포인트 / 제출 파일 생성 |

## base 모델 학습 (`train_*`)
- **1등 피처셋(first) 계열**: `train_first.py` + `train_first_{xgb,lgbm,ftt,tabm,mlpplr,dart,bag,fs,optuna,...}.py`
- **mega(572 피처) 계열**: `train_mega_{lgbm,xgb,cat,ag}.py`
- **개별 모델**: `train_{catboost,xgb,lgbm,logreg,knn,et,nn,realmlp,tabm,...}.py`, Optuna 튜닝판은 `*_optuna.py`
- **특수 피처축**: `train_{tfidf,txn,copurchase,interact,multiagg,leaf,nmf,dae,...}.py`

## Pseudo-labeling
- `train_pseudo_strict.py` — 최종 채택판. teacher(student 계열 격리) → test 고신뢰 라벨 → student 재학습. `PL_KFOLD` 등 환경변수로 fold 수 제어(variance 레버).
- `train_pseudo.py`, `train_pseudo_soft.py`, `train_pseudo_first.py`, `train_pl_fusion.py` — 변형 실험.

## 블렌딩·스태킹
- `blend_rank.py` — rank 가중 블렌딩 `rank(Σwᵢ·rank(srcᵢ))` (최종 제출 레시피)
- `stack_make73.py` / `blend_stack3.py` — 3-layer 메타스택(LR·Ridge·ET·HGB·KNN + hillclimb)
- `blend_{caruana,ridge10,power,de,lab}.py`, `stack.py`, `ensemble.py` — 결합 방식 비교 실험

## 검증·진단
- `adv_validation.py` / `adversarial_validation.py` — train/test 분포차 검증
- `eda_categories.py`, `extract_brands.py` — EDA·카테고리 신호 분석
- `export_*.py`, `import_mh_*.py` — 팀원 간 OOF 예측 교환

실패/폐기된 실험(TabPFN, node2vec, BERT, sequence transformer 등)은 [`../experiments_archive/`](../experiments_archive)에 보관 — 어떤 축을 시도했고 왜 접었는지의 기록입니다.
