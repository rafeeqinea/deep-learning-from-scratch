# 🧩 Deep Learning Book (2016) — Chapter-Scoped, NumPy-Only Lab

A faithful, hands-on rebuild of core ideas from **Goodfellow, Bengio, Courville (2016)** — but engineered as **chapter mini-packages**.  
Each chapter is a clean, importable API; the final **MNIST MLP app** composes *only* from these APIs.

> **Rule #1:** NumPy-only. No torch / tf / jax / autograd frameworks.  
> **Rule #2:** Small, testable modules. Chapter → section → tiny helpers.  
> **Rule #3:** The MNIST app does not bypass chapter APIs.

---

## 🗺️ Repository Layout (Top Level)

```
dl-notes-lab/
├─ chapters/              # chapter-scoped, reusable mini-packages
│  ├─ ch02_linear_algebra/
│  ├─ ch03_probability/
│  ├─ ch04_numerical/
│  ├─ ch05_ml_basics/
│  ├─ ch06_feedforward/
│  ├─ ch07_regularization/
│  ├─ ch08_optimization/
│  └─ __init__.py
├─ app_mnist/            # final MNIST MLP that uses chapter APIs
│  ├─ data_mnist.py
│  ├─ model_mlp.py
│  ├─ train.py
│  ├─ metrics.py
│  └─ tests/
├─ main.py               # CLI router (run chapter demos / train app)
└─ README.md
```

---

## 🧱 Chapter Packages (Full Blueprint)

Each chapter folder contains **section folders** (`sX_Y/`).  
Each section has a **section main** (`sX_Y.py`) that re-exports tiny helpers so imports stay tidy.

### ch02_linear_algebra/  *(all utilities used later; no external libs)*

```
chapters/ch02_linear_algebra/
├─ s2_1/                 # 2.1 Scalars, Vectors, Matrices and Tensors
│  ├─ s2_1.py
│  ├─ shapes.py          # infer_shape, ensure_vector/matrix/tensor_nd
│  ├─ dtypes.py          # as_float32/64, is_integer_array
│  └─ basic_validators.py # same_shape, broadcastable, finite
├─ s2_2/                 # 2.2 Multiplying Matrices and Vectors
│  ├─ s2_2.py
│  ├─ matvec.py          # mm, mv, vv (dot), safe_matmul
│  └─ linear_comb.py     # lincomb, affine_map
├─ s2_3/                 # 2.3 Identity and Inverse
│  ├─ s2_3.py
│  ├─ identity.py        # eye, projector(U)=U U^T
│  └─ inverse_solve.py   # solve (preferred), is_invertible, maybe_inverse
├─ s2_4/                 # 2.4 Linear Dependence and Span
│  ├─ s2_4.py
│  ├─ span_rank.py       # rank, is_full_rank, span_contains
│  └─ basis.py           # orthonormal_basis via SVD/QR
├─ s2_5/                 # 2.5 Norms
│  ├─ s2_5.py
│  └─ norms.py           # l1, l2, linf, Frobenius, op_norm_2 (σ_max)
├─ s2_6/                 # 2.6 Special Matrices/Vectors
│  ├─ s2_6.py
│  └─ special.py         # is_diagonal, is_symmetric, is_orthogonal, is_psd
├─ s2_7/                 # 2.7 Eigendecomposition
│  ├─ s2_7.py
│  └─ eigendecomp.py     # eig_sym, spectral_decomp, is_positive_definite
├─ s2_8/                 # 2.8 SVD
│  ├─ s2_8.py
│  └─ svd_tools.py       # svd_thin, cond_number, low_rank_approx
├─ s2_9/                 # 2.9 Moore–Penrose Pseudoinverse
│  ├─ s2_9.py
│  └─ pseudoinverse.py   # pinv_svd, min_norm_solve
├─ s2_10/                # 2.10 Trace
│  ├─ s2_10.py
│  └─ trace_ops.py       # trace, trace_cyclic_equal
├─ s2_11/                # 2.11 Determinant
│  ├─ s2_11.py
│  └─ determinant.py     # det, logdet_safe, volume_scale
├─ s2_12/                # 2.12 PCA
│  ├─ s2_12.py
│  └─ pca.py             # pca_svd, explained_variance, project_to_k
├─ api.py                # chapter re-exports
└─ tests/                # per-section tests
```

**Ch02 exports used later:** shape/dtype checks, safe matmul/affine, norms, SVD/cond, pinv, logdet, PCA helpers.

---

### ch03_probability/

```
chapters/ch03_probability/
├─ s3_5/
│  ├─ s3_5.py
│  └─ conditionals.py    # discrete/continuous slice-then-renormalize
├─ s3_6/
│  ├─ s3_6.py
│  └─ factorization.py   # chain-rule utilities (factorize joints)
├─ s3_7/
│  ├─ s3_7.py
│  └─ expectation.py     # empirical_expectation, variance, covariance
├─ s3_8/
│  ├─ s3_8.py
│  └─ information.py     # entropy, cross_entropy, KL (discrete)
├─ s3_9/
│  ├─ s3_9.py
│  └─ ml_map.py          # NLL from logits (one-hot), λ ↔ σ² (MAP prior)
├─ s3_10/
│  ├─ s3_10.py
│  └─ prob_numerics.py   # CE shift-invariance assertions, safety checks
├─ api.py
└─ tests/
```

**Ch03 exports used later:** CE/entropy/KL semantics for README plots and invariance tests (the MNIST app computes CE via Ch04/Ch06).

---

### ch04_numerical/

```
chapters/ch04_numerical/
├─ s4_1/
│  ├─ s4_1.py
│  └─ floating_point.py  # eps, overflow/underflow demos
├─ s4_2/
│  ├─ s4_2.py
│  └─ conditioning.py    # cond2_via_svd
├─ s4_3/
│  ├─ s4_3.py
│  └─ grad_theory.py     # descent-lemma demos (didactic)
├─ s4_4/
│  ├─ s4_4.py
│  └─ stable_reductions.py # logsumexp, log_softmax, softplus
├─ s4_5/
│  ├─ s4_5.py
│  └─ gradcheck.py       # central-difference gradient check
├─ api.py
└─ tests/
```

**Ch04 exports used later:** `log_softmax` (stable CE path) and `gradcheck` (backprop verification).

---

### ch05_ml_basics/

```
chapters/ch05_ml_basics/
├─ s5_1/
│  ├─ s5_1.py
│  └─ learning_setup.py  # task T, performance P, experience E
├─ s5_2/
│  ├─ s5_2.py
│  └─ capacity.py        # basic under/overfitting diagnostics
├─ s5_3/
│  ├─ s5_3.py
│  └─ splits_hparams.py  # train/val/test split
├─ s5_9/
│  ├─ s5_9.py
│  └─ sgd_basics.py      # batch_iterator, seeding
├─ s5_10/
│  ├─ s5_10.py
│  └─ system_design.py   # simple CSV logging helpers
├─ api.py
└─ tests/
```

**Ch05 exports used later:** seeding, splits, minibatches, accuracy, lightweight logging.

---

### ch06_feedforward/

```
chapters/ch06_feedforward/
├─ s6_1/
│  ├─ s6_1.py
│  └─ layers.py          # linear_forward/backward (cache X)
├─ s6_2/
│  ├─ s6_2.py
│  └─ activations.py     # relu_forward/backward (mask from Z)
├─ s6_3/
│  ├─ s6_3.py
│  └─ init.py            # glorot_uniform
├─ s6_4/
│  ├─ s6_4.py
│  └─ losses.py          # cross_entropy_from_logits (calls ch04.log_softmax)
├─ s6_5/
│  ├─ s6_5.py
│  └─ backprop_identities.py # ce_grad_wrt_logits = (softmax - onehot)/B
├─ api.py
└─ tests/
```

**Ch06 exports used later:** all building blocks of the MLP forward/backward and stable CE.

---

### ch07_regularization/

```
chapters/ch07_regularization/
├─ s7_1/
│  ├─ s7_1.py
│  └─ weight_decay.py    # l2_value (λ/2 Σ||W||²), l2_grad (λW), exclude_biases
├─ api.py
└─ tests/
```

**Ch07 exports used later:** L2 value + gradients for weights (biases excluded).

---

### ch08_optimization/

```
chapters/ch08_optimization/
├─ s8_1/
│  ├─ s8_1.py
│  └─ sgd.py             # sgd_update (in-place)
├─ s8_2/
│  ├─ s8_2.py
│  └─ momentum.py        # momentum_update (in-place)
├─ s8_3/
│  ├─ s8_3.py
│  └─ schedules.py       # cosine_lr (optional)
├─ api.py
└─ tests/
```

**Ch08 exports used later:** parameter update rules, optional LR schedules.

---

## 🧪 MNIST App (Composes Chapters Only)

```
app_mnist/
├─ data_mnist.py         # uses ch05: splits, batches, seeding
├─ model_mlp.py          # uses ch06: layers/activations/loss; ch07: L2; ch04: log_softmax (via ch06)
├─ train.py              # uses ch08: sgd/momentum; ch05: logging, accuracy
├─ metrics.py            # uses ch05: accuracy_from_logits
└─ tests/
   ├─ test_softmax_ce.py # stability & shift invariance
   ├─ test_gradcheck.py  # tiny batch finite-diff check
   └─ test_end_to_end_smoke.py # 1-epoch smoke: CE↓, acc↑
```

**Wiring (data/grad flow):**

```
batches → forward → CE (log-softmax) + L2 → backward (dZ2 etc.) → sgd/momentum → val/test metrics
```

---

## 🔗 Dependency Graph (who imports whom)

```
ch02_linear_algebra  ┐
ch03_probability     │  (semantics/tests; optional at runtime)
                     ├──▶ ch04_numerical ──▶ ch06_feedforward (loss uses ch04.log_softmax)
ch05_ml_basics  ─────┘                        ▲
                                              │
ch07_regularization ──────────────────────────┘ (L2 value + grads)
ch08_optimization ───────────────────────────▶ app_mnist/train.py

app_mnist/model_mlp.py  ◀── ch06 (forward/backward) + ch07 (L2)
app_mnist/data_mnist.py ◀── ch05 (splits/batching)
app_mnist/metrics.py    ◀── ch05 (accuracy)
```

---

## 🧭 Build Order (keeps the repo runnable at all times)

1) **Ch04** stable reductions (`logsumexp`, `log_softmax`) + **gradcheck**  
2) **Ch06** layers, ReLU, Glorot; CE from logits; CE grad identity  
3) **Ch07** L2 value/grad (weights only)  
4) **Ch05** splits, batching, accuracy, seeding, basic logging  
5) **App**: model wrapper, training loop, metrics; smoke tests  
6) **Ch03** probability semantics & invariance tests for README plots  
7) **Ch02** linear algebra utilities (used for diagnostics & extensions)

---

## ✅ Testing

- Use `pytest` (or `python -m pytest`) to run everything in `chapters/**/tests` and `app_mnist/tests`.
- Core invariants:
  - Softmax rows sum to 1; CE is **shift-invariant** (add constant per row of logits → unchanged).
  - Gradcheck (central difference) agrees with backprop within tolerance (float64 on tiny batch).
  - L2 penalizes **weights only**.

---

## 🛠️ Conventions

- **NumPy-only** (`numpy` is the only array library; no autograd).
- Prefer **float32** for training; switch to **float64** only for gradcheck.
- Shapes explicit in docstrings (e.g., `X: (B,784)`).
- Deterministic runs via seeded RNG (chapter 5 API).
- Keep functions **small**; section `sX_Y.py` re-exports helpers so import sites stay clean.

---

## 🧩 Example Imports (no frameworks, only chapter APIs)

```python
# MNIST app uses only chapter APIs
from chapters.ch06_feedforward.api   import linear_forward, relu_forward, cross_entropy_from_logits, ce_grad_wrt_logits, glorot_uniform
from chapters.ch07_regularization.api import l2_value, l2_grad
from chapters.ch08_optimization.api  import sgd_update, momentum_update
from chapters.ch05_ml_basics.api     import train_val_test_split, batch_iterator, accuracy_from_logits, set_all_seeds
```

---

## 📄 License

MIT — free to use, fork, and remix (for learning and research).
