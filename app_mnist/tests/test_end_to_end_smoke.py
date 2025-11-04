import numpy as np
from app_mnist.data_mnist import load_dataset
from app_mnist.model_mlp import init_mlp, forward_mlp, backward_mlp, ce_loss_from_logits
from app_mnist.metrics import accuracy
from app_mnist.train import train_epoch, evaluate
from chapters.ch05_ml_basics.api import set_all_seeds

def test_smoke_training_improves_on_synthetic():
    set_all_seeds(7)
    # small synthetic to keep CI fast and deterministic
    (Xtr, ytr), (Xv, yv), _ = load_dataset("synthetic", n_train=512, n_val=256, n_test=256, seed=7)
    Din, C = Xtr.shape[1], int(ytr.max() + 1)
    params = init_mlp(Din, hidden_dim=64, output_dim=C, seed=7)

    init_loss, init_acc = evaluate(params, Xv, yv, batch_size=128)
    # train a couple of epochs with momentum + small L2
    state = None
    for ep in range(3):
        _, state = train_epoch(params, Xtr, ytr, lr=0.2, batch_size=128,
                               optimizer="momentum", state=state, beta=0.9,
                               nesterov=True, l2_lambda=1e-4, seed=7+ep)
    final_loss, final_acc = evaluate(params, Xv, yv, batch_size=128)

    # smoke-level assertions: loss down, accuracy up a bit
    assert final_loss <= init_loss * 0.9  # at least 10% drop
    assert final_acc >= init_acc + 0.05   # at least +5% absolute
