"""
Minimal MNIST loader with a safe synthetic fallback.

Rules honored:
- NumPy-only core (stdlib urllib/gzip/struct for download/parse)
- Returns flattened X in [0,1], dtype=float32, labels int64
"""

from __future__ import annotations
import os, gzip, struct, urllib.request
import numpy as np
from typing import Tuple, Optional

MNIST_URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images":  "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels":  "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}


def _download(url: str, path: str, timeout: int = 30) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)  # no extra deps

def _parse_idx_images(gz_path: str) -> np.ndarray:
    with gzip.open(gz_path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError("bad magic for images")
        buf = f.read(rows * cols * num)
        X = np.frombuffer(buf, dtype=np.uint8).reshape(num, rows * cols)
        return (X.astype(np.float32) / 255.0)

def _parse_idx_labels(gz_path: str) -> np.ndarray:
    with gzip.open(gz_path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError("bad magic for labels")
        buf = f.read(num)
        y = np.frombuffer(buf, dtype=np.uint8)
        return y.astype(np.int64)

def load_mnist(data_dir: str = "assets/mnist") -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    os.makedirs(data_dir, exist_ok=True)
    paths = {
        k: os.path.join(data_dir, os.path.basename(v))
        for k, v in MNIST_URLS.items()
    }
    # download if missing (best-effort)
    for key, url in MNIST_URLS.items():
        try:
            _download(url, paths[key])
        except Exception:
            # leave file missing; caller can fallback
            pass

    if all(os.path.exists(paths[k]) for k in paths):
        Xtr = _parse_idx_images(paths["train_images"])
        ytr = _parse_idx_labels(paths["train_labels"])
        Xte = _parse_idx_images(paths["test_images"])
        yte = _parse_idx_labels(paths["test_labels"])
        # standard small validation carve-out
        n_val = 5000 if Xtr.shape[0] > 10000 else max(1000, Xtr.shape[0] // 10)
        Xv, yv = Xtr[:n_val], ytr[:n_val]
        Xtr, ytr = Xtr[n_val:], ytr[n_val:]
        return (Xtr, ytr), (Xv, yv), (Xte, yte)

    raise RuntimeError("MNIST files missing and download failed")

def load_mnist_subset(max_train=2000, max_val=500, max_test=1000, seed: int = 0):
    # convenience thin wrapper
    (Xtr, ytr), (Xv, yv), (Xte, yte) = load_mnist()
    rng = np.random.default_rng(seed)
    def take(X, y, n):
        if n is None or n >= X.shape[0]: return X, y
        idx = rng.choice(X.shape[0], size=n, replace=False)
        return X[idx], y[idx]
    return take(Xtr, ytr, max_train), take(Xv, yv, max_val), take(Xte, yte, max_test)

def _synthetic_block_digits(n: int, seed: int = 123) -> Tuple[np.ndarray, np.ndarray]:
    """
    784-dim vectors with class-specific "block" activations + noise.
    Trainable by a tiny MLP; used for smoke tests / offline runs.
    """
    rng = np.random.default_rng(seed)
    C = 10
    H, W = 28, 28
    block_h, block_w = 7, 7
    X = np.zeros((n, H, W), dtype=np.float32)
    y = rng.integers(0, C, size=(n,), dtype=np.int64)
    for i in range(n):
        k = int(y[i])
        r = (k // 5) * block_h
        c = (k % 5) * block_w
        X[i, r:r+block_h, c:c+block_w] = 1.0
    X += 0.25 * rng.standard_normal(size=X.shape).astype(np.float32)
    X = np.clip(X, 0.0, 1.0).reshape(n, H*W)
    return X.astype(np.float32), y

def load_dataset(name: str = "auto",
                 n_train: int = 1024,
                 n_val: int = 256,
                 n_test: int = 256,
                 seed: int = 0):
    """
    name: "mnist", "synthetic", or "auto" (try MNIST then fallback to synthetic)
    """
    if name == "synthetic":
        Xtr, ytr = _synthetic_block_digits(n_train, seed=seed)
        Xv,  yv  = _synthetic_block_digits(n_val,   seed=seed+1)
        Xte, yte = _synthetic_block_digits(n_test,  seed=seed+2)
        return (Xtr, ytr), (Xv, yv), (Xte, yte)

    try:
        (Xtr, ytr), (Xv, yv), (Xte, yte) = load_mnist_subset(
            max_train=n_train, max_val=n_val, max_test=n_test, seed=seed
        )
        return (Xtr, ytr), (Xv, yv), (Xte, yte)
    except Exception:
        if name == "mnist":
            raise
        # auto fallback
        return load_dataset("synthetic", n_train, n_val, n_test, seed)
