# Determinant helpers (NumPy only), written simply with clear checks.
import numpy as np
from chapters.ch02_linear_algebra.s2_1 import ensure_matrix


def _require_square(A, name="A"):
    """Raise an error if A is not square."""
    ensure_matrix(A)
    n, m = A.shape
    if n != m:
        raise ValueError("%s must be square, got shape %r" % (name, A.shape))


def det(A):
    """
    Determinant of a square matrix.

    Notes:
      - This uses NumPy's LU-based determinant internally.
      - Can over/underflow for large n; prefer slogdet for stability when
        you only need log|det(A)|.
    """
    _require_square(A, name="A")
    return float(np.linalg.det(A))


def sign_logabsdet(A):
    """
    Numerically stable sign and log-absolute-determinant via slogdet:

        sign, logabs = np.linalg.slogdet(A)

    Meaning:
      - If det(A) > 0, sign = +1 and logabs = log(det(A))
      - If det(A) < 0, sign = -1 and logabs = log(|det(A)|)
      - If det(A) = 0, sign = 0  and logabs = -inf

    Returns:
      (sign: float in {-1, 0, +1}, logabs: float)
    """
    _require_square(A, name="A")
    sign, logabs = np.linalg.slogdet(A)
    # Force plain Python floats for friendliness
    return float(sign), float(logabs)


def logdet_safe(A, assume_spd=False):
    """
    Return a *finite* log-determinant when it exists safely.

    - If A is symmetric positive definite (SPD), log(det(A)) is well-defined
      and positive; set assume_spd=True to enforce/assume this and just return
      the scalar logdet.

    - If `assume_spd=False`, we still compute sign/log|det| using slogdet, but:
        * If sign <= 0 (det <= 0), we raise a ValueError because log(det(A))
          is undefined (negative or zero).
        * If sign > 0, we return log|det(A)|.

    This keeps the function beginner-friendly: you either explicitly assume SPD
    or you get a clean error when log(det(A)) would not make sense.

    Returns:
      float logdet  (never returns -inf; raises instead).
    """
    _require_square(A, name="A")
    sign, logabs = np.linalg.slogdet(A)

    if assume_spd:
        # For SPD matrices, sign should be +1. If not, tell the user.
        if sign <= 0:
            raise ValueError(
                "logdet_safe: assume_spd=True but det(A) <= 0. "
                "Matrix may not be positive definite."
            )
        return float(logabs)

    # Generic case: only return a real log(det(A)) if det(A) > 0
    if sign > 0:
        return float(logabs)
    elif sign == 0:
        raise ValueError("logdet_safe: det(A) = 0 (singular), logdet = -inf.")
    else:
        raise ValueError("logdet_safe: det(A) < 0, log(det(A)) is not real.")


def volume_scale(A):
    """
    Return |det(A)|, which acts as the volume scale factor of the linear map A.

    Uses slogdet for numerical stability:
        |det(A)| = exp( log|det(A)| )

    This is always nonnegative (0 for singular matrices).
    """
    _require_square(A, name="A")
    sign, logabs = np.linalg.slogdet(A)
    if sign == 0:
        return 0.0
    # exp(log|det|) gives |det|
    return float(np.exp(logabs))
