"""
Chapter 2 — Linear Algebra (public API, NumPy-only)

One-stop import point for all section utilities: s2_1 … s2_12

Usage
-----
from chapters.ch02_linear_algebra.api import (
    # examples
    ensure_matrix, ensure_vector,            # §2.1
    mm, mv, vv, affine_map,                  # §2.2
    solve, maybe_inverse, eye, projector,    # §2.3
    rank, orthonormal_basis,                 # §2.4
    norm_l2, frobenius, op_norm_2,           # §2.5
    is_symmetric, is_orthogonal, is_psd,     # §2.6
    eig_sym, spectral_decomp,                # §2.7
    svd_thin, cond_number, low_rank_approx,  # §2.8
    pinv_svd, min_norm_solve,                # §2.9
    trace, trace_cyclic_equal,               # §2.10
    det, logdet_safe, volume_scale,          # §2.11
    center, pca_svd, project_to_k,           # §2.12
)
"""

# Re-export all public names from each section package.
# Each s2_X package already re-exports its own public functions via __all__.

from .s2_1 import *   # shapes dtypes validators
from .s2_2 import *   # mm mv vv lincomb affine_map add_affine safe_matmul
from .s2_3 import *   # eye projector is_invertible solve maybe_inverse
from .s2_4 import *   # rank is_full_rank span_contains orthonormal_basis
from .s2_5 import *   # norm_l1 norm_l2 norm_linf frobenius op_norm_2
from .s2_6 import *   # is_diagonal is_symmetric is_orthogonal is_psd
from .s2_7 import *   # eig_sym spectral_decomp is_positive_definite
from .s2_8 import *   # svd_thin cond_number low_rank_approx
from .s2_9 import *   # pinv_svd min_norm_solve
from .s2_10 import *  # trace trace_cyclic_equal
from .s2_11 import *  # det logdet_safe sign_logabsdet volume_scale
from .s2_12 import *  # center pca_svd explained_variance project_to_k reconstruct_from_k

# Friendly __all__: export everything that is not private
__all__ = [name for name in globals().keys() if not name.startswith("_")]

# Re-export all public names from each section package.
# Each s2_X package already re-exports its own public functions.

from .s2_1 import *   # shapes, dtypes, validators
from .s2_2 import *   # mm, mv, vv, lincomb, affine_map, add_affine, safe_matmul
from .s2_3 import *   # eye, projector, is_invertible, solve, maybe_inverse
from .s2_4 import *   # rank, is_full_rank, span_contains, orthonormal_basis
from .s2_5 import *   # norm_l1, norm_l2, norm_linf, frobenius, op_norm_2
from .s2_6 import *   # is_diagonal, is_symmetric, is_orthogonal, is_psd
from .s2_7 import *   # eig_sym, spectral_decomp, is_positive_definite
from .s2_8 import *   # svd_thin, cond_number, low_rank_approx
from .s2_9 import *   # pinv_svd, min_norm_solve
from .s2_10 import *  # trace, trace_cyclic_equal
from .s2_11 import *  # det, logdet_safe, sign_logabsdet, volume_scale
from .s2_12 import *  # center, pca_svd, explained_variance, project_to_k, reconstruct_from_k

# Create a friendly __all__ so "from ...api import *" only exports public symbols.
# (We rely on each section’s own __all__ to have already limited what got imported.)
__all__ = [name for name in globals().keys() if not name.startswith("_")]
