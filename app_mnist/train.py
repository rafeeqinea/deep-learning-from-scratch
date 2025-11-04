from __future__ import annotations
import argparse, time
import numpy as np
from typing import Dict, Tuple

from app_mnist.data_mnist import load_dataset
from app_mnist.model_mlp import init_mlp, forward_mlp, backward_mlp, ce_loss_from_logits
from app_mnist.metrics import accuracy

from chapters.ch05_ml_basics.api import set_all_seeds, batch_iterator
from chapters.ch07_regularization.api import l2_value, l2_grad, apply_weight_decay
from chapters.ch08_optimization.api import sgd_update, momentum_update

Params = Dict[str, np.ndarray]
Grads  = Dict[str, np.ndarray]

def evaluate(params: Params, X: np.ndarray, y: np.ndarray, batch_size: int = 256) -> Tuple[float, float]:
    total_loss, correct, seen = 0.0, 0, 0
    for Xb, yb in batch_iterator(X, y, batch_size=batch_size, shuffle=False):
        logits, _ = forward_mlp(Xb, params)
        total_loss += ce_loss_from_logits(logits, yb) * Xb.shape[0]
        correct += int(np.sum(np.argmax(logits, axis=1) == yb))
        seen += Xb.shape[0]
    return total_loss / max(1, seen), correct / max(1, seen)

def train_epoch(params: Params,
                X: np.ndarray, y: np.ndarray,
                lr: float,
                batch_size: int,
                optimizer: str = "sgd",
                state = None,
                beta: float = 0.9,
                nesterov: bool = False,
                l2_lambda: float = 0.0,
                seed: int = 0):
    rng = np.random.default_rng(seed)
    losses = []
    for Xb, yb in batch_iterator(X, y, batch_size=batch_size, shuffle=True, seed=int(rng.integers(0, 1<<31))):
        logits, cache = forward_mlp(Xb, params)
        loss = ce_loss_from_logits(logits, yb)
        # grads from CE
        grads = backward_mlp(logits, yb, cache)
        # add L2 grad if any (exclude biases)
        if l2_lambda > 0.0:
            wd = l2_grad(params, l2_lambda, exclude_biases=True)
            for k in grads:
                grads[k] = grads[k] + wd[k]
            loss = loss + l2_value(params, l2_lambda, exclude_biases=True)
        # update
        if optimizer == "sgd":
            sgd_update(params, grads, lr=lr)
        elif optimizer == "momentum":
            state = momentum_update(params, grads, state, lr=lr, beta=beta, nesterov=nesterov)
        else:
            raise ValueError("optimizer must be 'sgd' or 'momentum'")
        losses.append(loss)
    return float(np.mean(losses)), state

def train_main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["auto", "mnist", "synthetic"], default="auto")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--opt", choices=["sgd", "momentum"], default="momentum")
    ap.add_argument("--beta", type=float, default=0.9)
    ap.add_argument("--nesterov", action="store_true")
    ap.add_argument("--l2", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train", type=int, default=2048)
    ap.add_argument("--val", type=int, default=512)
    ap.add_argument("--test", type=int, default=512)
    args = ap.parse_args()

    set_all_seeds(args.seed)
    (Xtr, ytr), (Xv, yv), (Xte, yte) = load_dataset(args.dataset, args.train, args.val, args.test, seed=args.seed)

    D_in, C = Xtr.shape[1], int(np.max(ytr) + 1)
    params = init_mlp(D_in, args.hidden, C, seed=args.seed)

    state = None
    best_val = 0.0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, state = train_epoch(
            params, Xtr, ytr, lr=args.lr, batch_size=args.batch,
            optimizer=args.opt, state=state, beta=args.beta,
            nesterov=args.nesterov, l2_lambda=args.l2, seed=args.seed + ep
        )
        val_loss, val_acc = evaluate(params, Xv, yv, batch_size=args.batch)
        te_loss, te_acc   = evaluate(params, Xte, yte, batch_size=args.batch)
        best_val = max(best_val, val_acc)
        dt = time.time() - t0
        print(f"epoch {ep:02d} | train {train_loss:.4f} | val {val_loss:.4f} ({val_acc*100:.1f}%) | "
              f"test {te_loss:.4f} ({te_acc*100:.1f}%) | {dt:.2f}s")

if __name__ == "__main__":
    train_main()
