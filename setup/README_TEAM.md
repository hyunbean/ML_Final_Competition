# 📌 팀 안내 (신현빈 → 김민형) — KML 2026S 성별예측

> 이 repo를 git pull 한 뒤 이 파일부터 읽으면 현황 파악됨. (Claude에게 이 파일 읽으라고 하면 됨)

## 🏆 현재 최고 (2026-06-06 밤 기준)
- **제출파일**: `artifacts/submissions/57_stack3_r10hierTE_hyunbin.csv`
- **OOF(CV) 0.72768 / LB 0.73369 → 현재 1위** (종전 1등 0.73363 추월)
- ⚠️ 제출 시 신현빈이 **번호를 팀 형식으로 바꿔서** 올림. **번호 말고 파일명·내용으로** 찾을 것.

## 🔑 핵심 인사이트 (뭐가 먹혔나)
- **금맥 = 작년 1등 피처셋(497) × 다양한 모델(CatBoost/LGBM/XGB)**. first_cat/first_lgbm/first_xgb 각각 블렌드 +0.0003~0.001.
- **features r10 새 FE가 큰 게인**(+0.00087): trajectory(전후반 delta+지출기울기), TE-shape(extreme_hi/lo/conf=선물형 구분), 계층/잔차TE(Bayesian shrinkage), goodcd item2vec, 전이확률TE.
- **weight 0 (효과없음)**: mega_cat/xgb/kitchen, leiden, e5, tabicl, interact_cat, first_et/rf/mlp/lr, power-mean, Ridge×10. → 비슷하거나 약한 멤버는 스택서 죽음. **저상관·강한 멤버만 삶.**

## 📂 파일 구조
- `artifacts/oof/<model>__oof.npy` + `__test.npy` + `__meta.json`(cv_auc/notes) — 모든 단일모델 OOF (블렌드 재료)
- `artifacts/submissions/` — 제출 CSV (번호_설명_이름.csv)
- `src/` — 학습/블렌드 스크립트. `src/blend_stack3.py`=메인 3-level 스택, `src/features.py`=피처(r10), `src/train_first*.py`=1등피처셋 모델.

## ▶️ 재현 (블렌드 최신화)
```bash
python -m src.folds            # ids/folds (1회)
python -m src.features         # r10 피처캐시 (W2V라 몇 분)
# 모델 재학습 필요시: python -m src.train_first / train_first_lgbm / train_first_xgb / train_catboost ...
python -m src.blend_stack3     # → artifacts/submissions/submission_stack3.csv (최신 블렌드)
```
새 모델 만들면 `bash scripts/push_oof.sh <모델명>` 으로 OOF 공유.

## 🔄 진행 중 (overnight)
- `src/train_first_optuna.py` — 1등피처셋 lgbm/xgb/cat Optuna 튜닝 → `first_*_opt` (강해지면 블렌드↑)

## 🎯 최종제출 전략 (마감 6/10 23:00)
Kaggle 2개 선택 → private 결정. 격차 ~0.0001 = 노이즈 → **셰이크업 확정**.
- **Slot1 [안정]**: OOF(CV) 최고 stack (=현재 57류)
- **Slot2 [공격]**: 직교 모델들 **rank-average** (scipy rankdata→평균, AUC 순위지표라 private 맷집)
- 금지: CV낮고 LB만 높은 "public 영웅" 단독
- 체크: adversarial validation(train/test AUC≈0.5), seed averaging

## 📝 기록
모든 실험은 Notion 공유 실험로그에 기록됨 (실험명/CV/LB/변경점).
