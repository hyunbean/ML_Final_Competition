#!/usr/bin/env bash
# 학습 결과(OOF/test npy)를 GitHub에 공유.
# 사용:  bash scripts/push_oof.sh "메시지"     (메시지 생략 가능)
cd "$(dirname "$0")/.." || exit 1

git config user.email "yyhhss0812@gmail.com"
git config user.name "hyunbean"

git add artifacts/oof
git commit -m "${1:-results: update OOF}" || echo "(커밋할 새 결과 없음)"
git pull --rebase origin main
git push origin main && echo "✅ OOF 공유 완료 → 팀원/다른머신은 git pull 후 python -m src.ensemble"
