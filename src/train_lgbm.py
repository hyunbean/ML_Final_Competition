"""LGBM 베이스라인 트레이너 — 체크포인트/재시작 + OOF·test npy 저장 데모.

흐름:
  1) folds.py 출력(정규순서/폴드) 로드
  2) fold 마다 학습 → val=OOF, test=누적. fold 끝날 때마다 체크포인트 저장.
     (중단 후 다시 실행하면 끝난 fold는 건너뛰고 이어서)
  3) 전부 끝나면 CV AUC 출력 → OOF/test npy 저장 → 체크포인트 삭제

실행:
  python -m src.folds        # (최초 1회) 정규순서/폴드 생성
  python -m src.train_lgbm   # 학습
"""
import numpy as np
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

from . import config as C
from .data import build_xy
from .features import build_features
from .checkpoint import Checkpoint
from .oof_io import save_predictions

# True=full 피처(타깃인코딩+임베딩+다양성), False=베이스라인 집계만(빠른 기준점)
USE_FULL_FEATURES = True
MODEL_NAME = "lgbm_full" if USE_FULL_FEATURES else "lgbm_baseline"
CREATED_BY = "<이름>"   # 팀 공유 시 본인 이름으로

PARAMS = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.03,
    num_leaves=63,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    min_child_samples=50,
    n_estimators=5000,
    n_jobs=-1,
    verbosity=-1,
    # GPU 서버에서 가속하려면 아래 주석 해제:
    # device_type="gpu",
)


def main():
    X, y, Xtest = build_features() if USE_FULL_FEATURES else build_xy()
    folds = np.load(C.FOLDS_NPY)
    n_tr, n_te = len(X), len(Xtest)
    print(f"X_train={X.shape}  X_test={Xtest.shape}  features={X.shape[1]}")

    ckpt = Checkpoint(MODEL_NAME, C.CKPT_DIR)
    state = ckpt.load(default=dict(
        done=[],
        oof=np.full(n_tr, np.nan, dtype=np.float64),
        test_sum=np.zeros(n_te, dtype=np.float64),
        fold_auc={},
    ))

    for f in range(C.N_FOLDS):
        if f in state["done"]:
            continue
        tr_idx = np.where(folds != f)[0]
        va_idx = np.where(folds == f)[0]

        model = lgb.LGBMClassifier(**PARAMS)
        model.fit(
            X.iloc[tr_idx], y[tr_idx],
            eval_set=[(X.iloc[va_idx], y[va_idx])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(200, verbose=False)],
        )
        state["oof"][va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
        state["test_sum"] += model.predict_proba(Xtest)[:, 1]
        auc = roc_auc_score(y[va_idx], state["oof"][va_idx])
        state["fold_auc"][str(f)] = float(auc)
        state["done"].append(f)
        ckpt.save(state)   # ← fold 단위 체크포인트
        print(f"[fold {f}] AUC={auc:.5f}  best_iter={model.best_iteration_}  done={state['done']}")

    oof = state["oof"]
    test_pred = state["test_sum"] / C.N_FOLDS
    cv_auc = float(roc_auc_score(y, oof))
    print(f"\n==== {MODEL_NAME}  CV AUC = {cv_auc:.5f} ====")

    save_predictions(MODEL_NAME, oof, test_pred, meta=dict(
        cv_auc=cv_auc,
        fold_auc=state["fold_auc"],
        seed=C.SEED,
        n_folds=C.N_FOLDS,
        params=PARAMS,
        feature_set="full(te+emb+div)" if USE_FULL_FEATURES else "baseline_agg",
        created_by=CREATED_BY,
        notes="LGBM",
    ))
    ckpt.cleanup()   # ← 전부 학습 완료 → 체크포인트 삭제


if __name__ == "__main__":
    main()
