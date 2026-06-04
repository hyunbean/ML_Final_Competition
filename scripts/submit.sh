#!/usr/bin/env bash
# Kaggle에 바로 제출 (다운로드 불필요). CSV 없으면 자동 생성.
# 사용:  bash scripts/submit.sh <model> "메시지"
#   예:  bash scripts/submit.sh xgb_full "xgb cv0.7016"
#        bash scripts/submit.sh lgbm_full "lgbm v2"
# 사전: Kaggle API 토큰(~/.kaggle/kaggle.json) + pip install kaggle  (1회)
cd "$(dirname "$0")/.." || exit 1

MODEL="${1:?모델명 필요 (예: xgb_full)}"
MSG="${2:-$MODEL}"
COMP="kml-challenge-2026-s"
F="artifacts/submissions/submission_${MODEL}.csv"

if [ ! -f "$F" ]; then
  echo "[submit] CSV 없음 → 생성: $MODEL"
  python -m src.make_submission "$MODEL"
fi

echo "[submit] '$F' → Kaggle ($COMP)"
kaggle competitions submit -c "$COMP" -f "$F" -m "$MSG"
echo "--- 최근 제출/점수 ---"
kaggle competitions submissions -c "$COMP" | head -8
