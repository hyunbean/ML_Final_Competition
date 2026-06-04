# DLPC 셋업 가이드 (KML Challenge 2026S)

> DLPC = 국민대 GPU 클라우드(본진). 접속 방식(SSH / JupyterLab 웹터미널)에 따라 살짝 다를 수 있음 — 아래는 **터미널 기준**.

## 0. 접속 & 환경 확인
```bash
nvidia-smi                                              # GPU 동작 확인
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python --version ; which conda ; nproc ; free -h ; df -h
```
→ GPU 이름/VRAM 알려주면 CatBoost/NN 배치사이즈 맞춰드림.

## 1. 코드 가져오기
**옵션 A — GitHub (권장)**
```bash
git clone <레포-URL> kml-2026s && cd kml-2026s
```
**옵션 B — 직접 업로드** (GitHub 안 쓸 때): 로컬 `src/` 폴더를 JupyterLab 업로드 또는
```bash
# (로컬에서) scp -r "ML_Final_Competition/src" 사용자@DLPC:~/kml-2026s/
```

## 2. conda 환경 (Python 3.11)
```bash
conda create -n kml python=3.11 -y
conda activate kml
```

## 3. 패키지 설치
```bash
pip install numpy pandas scikit-learn scipy lightgbm catboost xgboost gensim optuna
# PyTorch (GPU, cu128) — NN 단계에서
pip install torch --index-url https://download.pytorch.org/whl/cu128
# (선택) AutoGluon
pip install autogluon.tabular
```
설치 확인:
```bash
python -c "import lightgbm,catboost,xgboost,gensim,sklearn; print('ml ok')"
python -c "import torch; print('cuda', torch.cuda.is_available())"   # True 떠야 함
```

## 4. 데이터 업로드
`train_transactions.csv` / `test_transactions.csv` / `y_train.csv` (총 ~215MB)를 DLPC로.
```bash
mkdir -p ~/kml-2026s/data
# 방법1) JupyterLab 파일 업로드
# 방법2) (로컬에서) scp "*.csv" 사용자@DLPC:~/kml-2026s/data/
# 방법3) Google Drive에 있으면:  pip install gdown && gdown <파일ID> -O data/train_transactions.csv
```

## 5. 환경변수 (데이터 경로)
```bash
export KML_DATA=~/kml-2026s/data        # config.py가 이 경로에서 CSV를 읽음
echo 'export KML_DATA=~/kml-2026s/data' >> ~/.bashrc   # 영구 설정
```

## 6. 실행 (Day 0 →)
```bash
cd ~/kml-2026s
python -m src.folds          # ① 폴드 고정 (최초 1회, 결과 커밋해 팀 공유)
python -m src.features       # ② 피처 빌드 + 캐시
python -m src.train_lgbm     # ③ LGBM 학습 → artifacts/oof/ 에 npy
python -m src.ensemble       # ④ 공유 OOF 합쳐 제출 파일
```

## 7. 야간/장시간 잡 = tmux (세션 끊겨도 계속)
```bash
tmux new -s kml
conda activate kml && cd ~/kml-2026s
python -m src.train_lgbm           # 또는 AutoGluon 등
# Ctrl+b 누르고 d → detach (백그라운드로 계속 돎)
tmux attach -t kml                 # 다시 들어가기
# 대안:  nohup python -m src.train_xxx > logs/lgbm.txt 2>&1 &
```

## 8. GPU/CPU 플래그
- **CatBoost**: `task_type="GPU"`  ·  **XGBoost**: `tree_method="hist", device="cuda"`
- **LightGBM**: CPU 그대로(`n_jobs=-1`)
- 동시 활용: GPU엔 cat/xgb/nn, CPU엔 lgbm/features를 **동시에** 돌려 풀가동.

## 9. 체크포인트 / 재시작
- 학습 중단 → **같은 명령 다시 실행**하면 끝난 fold부터 이어서.
- 5 fold 완료되면 OOF/test npy 저장 후 체크포인트 자동 삭제.

## 10. 결과 공유 (GitHub)
```bash
git add artifacts/oof/*.npy artifacts/oof/*.json artifacts/*.npy
git commit -m "lgbm_full oof (cv=...)" && git push
# 팀원:  git pull → python -m src.ensemble
```
> 데이터 CSV·체크포인트는 `.gitignore`로 제외됨. 작은 OOF/폴드 npy만 공유됨.
