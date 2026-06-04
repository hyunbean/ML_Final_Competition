"""제너릭 체크포인트 — 원자적 저장 + 재시작 시 이어서 학습.

- GBM 계열은 'fold' 단위로 체크포인트(가장 자연스러운 단위).
- 학습 중 중단(서버 끊김/끄기)되어도, 끝난 fold는 다시 안 함.
- 원자적 교체(os.replace)라 저장 도중 죽어도 파일이 깨지지 않음.
- NN(epoch 단위)도 같은 패턴으로 state에 epoch/optimizer 넣어 확장 가능.
"""
import os
import pickle
import tempfile
from pathlib import Path


def _atomic_write(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)   # 원자적 교체
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


class Checkpoint:
    def __init__(self, name: str, ckpt_dir):
        self.name = name
        self.path = Path(ckpt_dir) / f"{name}.ckpt.pkl"

    def load(self, default: dict) -> dict:
        if self.path.exists():
            with open(self.path, "rb") as f:
                state = pickle.load(f)
            print(f"[ckpt] resume '{self.name}' — 완료 fold {state.get('done', [])}")
            return state
        return default

    def save(self, state: dict):
        _atomic_write(self.path, pickle.dumps(state, protocol=4))

    def cleanup(self):
        """전부 학습 완료 후 체크포인트 삭제."""
        if self.path.exists():
            self.path.unlink()
            print(f"[ckpt] '{self.name}' 완료 → 체크포인트 삭제")
