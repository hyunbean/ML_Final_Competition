"""공통 설정 — 경로, 시드, CV, 컬럼 정의. 모든 모듈이 여기서 가져다 씀."""
from pathlib import Path
import os

# 프로젝트 루트 = 이 파일(src/config.py)의 상위 폴더. 환경변수로 덮어쓸 수 있음.
ROOT = Path(os.environ.get("KML_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR = Path(os.environ.get("KML_DATA", ROOT))          # csv들이 있는 곳
ARTIFACTS = Path(os.environ.get("KML_ARTIFACTS", ROOT / "artifacts"))

OOF_DIR = ARTIFACTS / "oof"             # 팀 공유용 예측 npy
CKPT_DIR = ARTIFACTS / "checkpoints"    # 학습 중간 체크포인트(임시, 완료 후 삭제)
SUB_DIR = ARTIFACTS / "submissions"     # 제출 파일

for _d in (ARTIFACTS, OOF_DIR, CKPT_DIR, SUB_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---- 데이터 ----
TRAIN_CSV = DATA_DIR / "train_transactions.csv"
TEST_CSV = DATA_DIR / "test_transactions.csv"
YTRAIN_CSV = DATA_DIR / "y_train.csv"
ENCODING = "cp949"

ID_COL = "custid"
TARGET = "gender"
POS_LABEL = 1

# ---- CV ----
SEED = 42
N_FOLDS = 5

# ---- 정규 순서/폴드 (팀 전원이 동일 파일을 써야 OOF 앙상블이 유효!) ----
TRAIN_IDS_NPY = ARTIFACTS / "train_custids.npy"
TEST_IDS_NPY = ARTIFACTS / "test_custids.npy"
FOLDS_NPY = ARTIFACTS / "folds.npy"
