"""AutoGluon (auto-ensemble) — per-fold OOF로 우리 스태킹에 합류. 강력+다양. 야간 추천.

각 fold에서 TabularPredictor를 학습해 val=OOF, test=누적 → 다른 모델과 동일 규격 OOF 생성.
CPU로도 잘 돌아서 DLPC 야간(tmux)에 적합. time_limit으로 길이 조절.

설치: pip install autogluon.tabular
실행: python -m src.train_autogluon [fold당_초]   (기본 600초)
"""
import sys
import numpy as np
from sklearn.metrics import roc_auc_score
from autogluon.tabular import TabularPredictor

from . import config as C
from .features import build_features
from .checkpoint import Checkpoint
from .oof_io import save_predictions

MODEL_NAME = "autogluon_full"
TIME = int(sys.argv[1]) if len(sys.argv) > 1 else 600
PRESET = sys.argv[2] if len(sys.argv) > 2 else "good_quality"   # good_quality | best_quality


def main():
    X, y, Xtest = build_features()
    folds = np.load(C.FOLDS_NPY)
    n_tr, n_te = len(X), len(Xtest)
    print(f"X={X.shape}  time/fold={TIME}s  preset={PRESET}")

    ckpt = Checkpoint(MODEL_NAME, C.CKPT_DIR)
    state = ckpt.load(default=dict(
        done=[], oof=np.full(n_tr, np.nan, dtype=np.float64),
        test_sum=np.zeros(n_te, dtype=np.float64), fold_auc={}))

    for f in range(C.N_FOLDS):
        if f in state["done"]:
            continue
        tr, va = np.where(folds != f)[0], np.where(folds == f)[0]
        tdf = X.iloc[tr].copy()
        tdf[C.TARGET] = y[tr]
        pred = TabularPredictor(
            label=C.TARGET, problem_type="binary", eval_metric="roc_auc",
            verbosity=1, path=str(C.ARTIFACTS / f"ag_fold{f}"),
        ).fit(tdf, presets=PRESET, time_limit=TIME)
        pos = pred.positive_class
        state["oof"][va] = pred.predict_proba(X.iloc[va])[pos].values
        state["test_sum"] += pred.predict_proba(Xtest)[pos].values
        auc = roc_auc_score(y[va], state["oof"][va])
        state["fold_auc"][str(f)] = float(auc)
        state["done"].append(f)
        ckpt.save(state)
        print(f"[fold {f}] AUC={auc:.5f}  done={state['done']}")

    oof = state["oof"]
    test_pred = state["test_sum"] / C.N_FOLDS
    cv = float(roc_auc_score(y, oof))
    print(f"\n==== {MODEL_NAME}  CV AUC = {cv:.5f} ====")

    save_predictions(MODEL_NAME, oof, test_pred, meta=dict(
        cv_auc=cv, fold_auc=state["fold_auc"], seed=C.SEED, n_folds=C.N_FOLDS,
        feature_set="full(te+emb+div+grp)", created_by="hyunbean",
        notes=f"AutoGluon {PRESET} {TIME}s/fold"))
    ckpt.cleanup()


if __name__ == "__main__":
    main()
