"""CatBoost 트레이너 (GPU) — train_lgbm.py와 동일 패턴: 체크포인트/재시작 + OOF·test npy.

참고: 우리 피처는 '고객 단위 집계 수치행렬'이라 CatBoost의 범주형 native 기능은 해당 없음
(범주 신호는 features.py의 타깃인코딩/구성비/임베딩이 이미 담음). CatBoost는 같은 수치행렬을
'다른 알고리즘'으로 학습해 앙상블 다양성을 준다. 이 데이터의 강력한 단일 후보.

실행:
  python -m src.folds            # (선행) 정규순서/폴드
  python -m src.train_catboost   # GPU 학습
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

from . import config as C
from .data import build_xy
from .features import build_features
from .checkpoint import Checkpoint
from .oof_io import save_predictions

USE_FULL_FEATURES = True
TASK_TYPE = "GPU"        # GPU 없으면 "CPU" 로 변경
MODEL_NAME = "catboost_full" if USE_FULL_FEATURES else "catboost_baseline"
CREATED_BY = "<이름>"    # 팀 공유 시 본인 이름으로

PARAMS = dict(
    loss_function="Logloss",
    eval_metric="AUC",
    iterations=6000,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=3.0,
    random_seed=C.SEED,
    task_type=TASK_TYPE,
    devices="0",
    verbose=False,
)


def main():
    X, y, Xtest = build_features() if USE_FULL_FEATURES else build_xy()
    folds = np.load(C.FOLDS_NPY)
    n_tr, n_te = len(X), len(Xtest)
    print(f"X_train={X.shape}  X_test={Xtest.shape}  features={X.shape[1]}  task={TASK_TYPE}")

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

        model = CatBoostClassifier(**PARAMS)
        model.fit(
            X.iloc[tr_idx], y[tr_idx],
            eval_set=(X.iloc[va_idx], y[va_idx]),
            early_stopping_rounds=200,
            verbose=False,
        )
        state["oof"][va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
        state["test_sum"] += model.predict_proba(Xtest)[:, 1]
        auc = roc_auc_score(y[va_idx], state["oof"][va_idx])
        state["fold_auc"][str(f)] = float(auc)
        state["done"].append(f)
        ckpt.save(state)   # ← fold 단위 체크포인트
        print(f"[fold {f}] AUC={auc:.5f}  best_iter={model.get_best_iteration()}  done={state['done']}")

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
        notes=f"CatBoost ({TASK_TYPE})",
    ))
    ckpt.cleanup()   # ← 전부 완료 → 체크포인트 삭제


if __name__ == "__main__":
    main()
