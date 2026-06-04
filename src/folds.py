"""정규 custid 순서 + StratifiedKFold 폴드 생성 (한 번만 실행).

출력: train_custids.npy / test_custids.npy / folds.npy
⚠️ 이 3개 파일은 반드시 팀원 전원이 '동일하게' 공유해서 써야 함.
   (각자 다른 폴드로 OOF를 만들면 스태킹 앙상블이 누수/무효가 됨)

실행: python -m src.folds
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from . import config as C


def main():
    y = pd.read_csv(C.YTRAIN_CSV)
    test_ids = pd.read_csv(C.TEST_CSV, encoding=C.ENCODING, usecols=[C.ID_COL])[C.ID_COL].unique()

    # 정규 순서 = custid 오름차순 (전원 동일하게 재현되도록)
    y = y.sort_values(C.ID_COL).reset_index(drop=True)
    train_ids = y[C.ID_COL].to_numpy()
    test_ids = np.sort(test_ids)
    target = y[C.TARGET].to_numpy()

    folds = np.full(len(y), -1, dtype=np.int8)
    skf = StratifiedKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=C.SEED)
    for f, (_, val_idx) in enumerate(skf.split(train_ids, target)):
        folds[val_idx] = f

    np.save(C.TRAIN_IDS_NPY, train_ids)
    np.save(C.TEST_IDS_NPY, test_ids)
    np.save(C.FOLDS_NPY, folds)

    print(f"saved: {len(train_ids)} train / {len(test_ids)} test, {C.N_FOLDS} folds, seed={C.SEED}")
    print(f"overall pos_rate = {target.mean():.4f}")
    for f in range(C.N_FOLDS):
        m = folds == f
        print(f"  fold {f}: n={m.sum():5d}  pos_rate={target[m].mean():.4f}")


if __name__ == "__main__":
    main()
