# Colab 셋업 (CPU 작업용)

DLPC GPU 대기 중이거나, CPU 모델(LightGBM)·피처·임베딩은 Colab에서.
DLPC는 GPU 모델(CatBoost/XGB/NN) 전용으로. (서버 1개 제약 → Colab으로 보완)

## 1) 노트북 첫 셀 (매 세션 1회)
```python
from google.colab import drive
drive.mount('/content/drive')

# 비공개 repo → 토큰으로 clone (<TOKEN>은 본인 PAT)
!git clone https://hyunbean:<TOKEN>@github.com/hyunbean/ML_Final_Competition.git
%cd ML_Final_Competition
!pip install -q -r requirements.txt
```

## 2) 데이터 (CSV 3개)
```python
# 옵션 A: Drive에 올려둔 CSV 복사 (권장 — 한 번 올려두면 재사용)
!mkdir -p data && cp /content/drive/MyDrive/ML_Final_Competition/*.csv data/
# 옵션 B: 왼쪽 파일창에 직접 업로드 → data/ 로
```

## 3) 실행 (CPU)
```python
import os; os.environ['KML_DATA'] = '/content/ML_Final_Competition/data'
!python -m src.folds        # seed=42 → DLPC와 동일 폴드
!python -m src.features     # W2V 임베딩 (몇 분)
!python -m src.train_lgbm   # LightGBM CV → OOF/test npy
```

## 4) 결과(OOF) GitHub로 공유
```python
!git config user.email "yyhhss0812@gmail.com"
!git config user.name "hyunbean"
!git add artifacts/oof/*.npy artifacts/oof/*.json
!git commit -m "feat: lgbm oof (colab)"
!git push https://hyunbean:<TOKEN>@github.com/hyunbean/ML_Final_Competition.git main
```
→ DLPC/팀원은 `git pull` 후 `python -m src.ensemble`로 합침.

## 주의
- **folds는 결정적**(seed=42): 데이터 동일하면 DLPC와 폴드 동일 → OOF 호환.
- Colab 디스크 **휘발** → 결과 npy는 꼭 **git push 또는 Drive 저장**.
- 12h/유휴 끊김 → 체크포인트로 재개(같은 명령 다시 실행).
- 모델 이름 겹치지 않게: Colab에서 만든 건 `lgbm_full` 그대로 두되, 여러 변형이면 접미사(`_colab` 등).
