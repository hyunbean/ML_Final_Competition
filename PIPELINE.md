# 파이프라인 사용법 (KML Challenge 2026S)

성별 예측(ROC-AUC). 거래→고객 집계 → CV/OOF → 모델 → 앙상블 → 제출.

## 폴더 구조
```
ML_Final_Competition/
├─ train_transactions.csv / test_transactions.csv / y_train.csv   # 데이터 (CP949)
├─ src/
│  ├─ config.py       # 경로·시드·CV·컬럼 (환경변수 KML_ROOT/KML_DATA로 덮어쓰기 가능)
│  ├─ folds.py        # 정규 custid 순서 + 폴드 생성 (★최초 1회, 팀 공유)
│  ├─ data.py         # 로드 + 거래→고객 베이스라인 집계 (build_xy)
│  ├─ checkpoint.py   # 체크포인트/재시작 (원자적 저장)
│  ├─ oof_io.py       # OOF/test npy 저장·로드 (팀 공유 규격)
│  ├─ train_lgbm.py   # LGBM 베이스라인 (체크포인트 데모)
│  └─ ensemble.py     # 공유 OOF 모아 블렌딩 → 제출
└─ artifacts/
   ├─ train_custids.npy / test_custids.npy / folds.npy   # ★팀 전원 동일하게 공유
   ├─ checkpoints/    # 학습 중간(임시, 완료 후 자동 삭제)
   ├─ oof/            # {model}__oof.npy / __test.npy / __meta.json  ← 팀 공유 대상
   └─ submissions/    # 제출 csv
```

## 실행 순서
```bash
pip install -r requirements.txt
python -m src.folds         # 최초 1회 (정규순서/폴드 고정)
python -m src.train_lgbm    # 모델 학습 → artifacts/oof/ 에 npy 저장
python -m src.ensemble      # 공유 OOF 합쳐 제출 파일 생성
```

## 체크포인트 / 재시작
- `train_*.py`는 **fold 단위로 체크포인트** 저장(`artifacts/checkpoints/{model}.ckpt.pkl`).
- 중간에 서버가 끊겨도 **다시 실행하면 끝난 fold는 건너뛰고 이어서** 학습.
- 5 fold 전부 끝나면 OOF/test npy 저장 후 **체크포인트 파일 자동 삭제**.
- 저장은 원자적(os.replace)이라 저장 도중 죽어도 파일이 깨지지 않음.
- NN 학습은 같은 패턴으로 state에 `epoch`/optimizer를 넣어 epoch 단위로 확장.

## 팀 공유 규약 (★중요)
1. **`artifacts/train_custids.npy`, `test_custids.npy`, `folds.npy` 는 한 사람이 만들어 전원이 그대로 사용.**
   (각자 다른 폴드로 OOF를 만들면 스태킹이 누수/무효가 됨)
2. 각자 모델을 학습하면 `artifacts/oof/{model}__{oof,test}.npy` + `__meta.json` 3종이 생김.
   - 이 **oof 폴더 파일만 공유**하면 됨 (작음, custid·정답 미포함).
   - `model_name`은 겹치지 않게: 예) `lgbm_te_민수`, `catboost_엔트로피_지영`.
3. 누군가 `python -m src.ensemble` 돌리면 공유된 모든 OOF를 자동으로 합쳐 제출 파일 생성.

## CPU / GPU 가이드 (모델·작업별)
| 작업/모델 | 권장 | 플래그 / 비고 |
|---|---|---|
| 피처 엔지니어링(pandas 집계) | **CPU(멀티코어)** | RAM 바운드. 코어 많이. (RAPIDS/cuDF 쓰면 GPU 가능하나 불필요) |
| Word2Vec / item2vec (gensim) | **CPU** | `workers=코어수` |
| SVD / NMF (sklearn) | **CPU** | 큰 행렬은 `TruncatedSVD`(randomized) |
| **LightGBM** | **CPU** | 이 데이터(≈3만행)는 CPU가 더 빠르고 안정적. `n_jobs=-1`. GPU 이득은 대용량에서만 |
| **CatBoost** | **GPU** | `task_type="GPU", devices="0"` — GPU 최적화 우수(특히 범주형) |
| **XGBoost** | **GPU** | XGB≥2.0: `tree_method="hist", device="cuda"` / 구버전: `tree_method="gpu_hist"` |
| **AutoGluon** | CPU+GPU | 트리계열은 CPU, NN은 GPU. `fit(..., num_gpus=1)` 주면 NN 가속 |
| **NN**(TabNet/FT-Transformer/MLP/시퀀스) | **GPU 필수** | torch `.to("cuda")` |
| Optuna 탐색 | 모델 따라감 | LGBM=CPU 트라이얼 병렬 / CatBoost·XGB=GPU 트라이얼 순차(또는 멀티GPU) |

> **머신 풀가동 팁**: GPU엔 CatBoost·XGBoost·NN을, 동시에 CPU엔 LightGBM·피처·Word2Vec을 돌리면
> CPU/GPU를 **동시에** 써서 야간 처리량을 극대화할 수 있음.

## 다음 작업
- `features.py`: 카테고리/브랜드 OOF 타깃인코딩, 구성비, 엔트로피, SVD, 임베딩 등 (build_xy 확장).
- `train_catboost.py`(GPU) / `train_xgb.py`(GPU) / `train_nn.py`(GPU): train_lgbm.py 패턴 복사 후 모델만 교체.
