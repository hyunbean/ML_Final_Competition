#!/usr/bin/env bash
# 학습 결과(OOF/test npy)를 GitHub에 공유.
# 사용:  bash scripts/push_oof.sh "메시지"     (메시지 생략 가능)
cd "$(dirname "$0")/.." || exit 1

git config user.email "guszhd95@naver.com"
git config user.name "hyunbean"

git add artifacts/oof
git commit -m "${1:-results: update OOF}" || echo "(커밋할 새 결과 없음)"
# 바이너리 OOF 충돌 시 '내 최신본' 우선(merge, ours). rebase 충돌 지옥 방지.
git pull --no-rebase -X ours --no-edit origin main
git push origin main && echo "✅ OOF 공유 완료 → 다른 머신은 git pull 후 python -m src.ensemble"
