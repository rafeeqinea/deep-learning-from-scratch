# s4_1 – floating point utilities
from .s4_1.s4_1 import (
    machine_eps, ulp, exp_safe_bounds, kahan_sum, tiny_perturbation_is_ignored
)

# s4_2 – conditioning
from .s4_2.s4_2 import (
    cond2_via_svd,
)

# s4_3 – gradient theory demos
from .s4_3.s4_3 import (
    make_ls_quadratic, lipschitz_const_from_A, descent_lemma_gap
)

# s4_4 – stable reductions
from .s4_4.s4_4 import (
    logsumexp, log_softmax, softplus
)

# s4_5 – gradcheck
from .s4_5.s4_5 import (
    central_diff_grad, check_grad
)

__all__ = [
    # s4_1
    "machine_eps", "ulp", "exp_safe_bounds", "kahan_sum", "tiny_perturbation_is_ignored",
    # s4_2
    "cond2_via_svd",
    # s4_3
    "make_ls_quadratic", "lipschitz_const_from_A", "descent_lemma_gap",
    # s4_4
    "logsumexp", "log_softmax", "softplus",
    # s4_5
    "central_diff_grad", "check_grad",
]
