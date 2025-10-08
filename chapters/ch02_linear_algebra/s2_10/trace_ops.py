# Trace operator utilities (NumPy only), written simply with clear checks.
import numpy as np
from chapters.ch02_linear_algebra.s2_1 import ensure_matrix


def _require_square(M, name="matrix"):
    """
    Raise an error if M is not square.
    In this project, we use the mathematical definition of trace (square only),
    not NumPy's permissive sum of diagonal for non-square matrices.
    """
    ensure_matrix(M)
    m, n = M.shape
    if m != n:
        raise ValueError("%s must be square, got shape %r" % (name, M.shape))


def trace(A):
    """
    Return the trace of a square matrix A:
        tr(A) = sum of diagonal entries.

    We *require* A to be square (n x n). This matches the math definition used
    throughout the book and later derivations.
    """
    _require_square(A, name="A")
    # Use NumPy's trace (now that A is square, this matches the math definition)
    return float(np.trace(A))


def _trace_of_product_two(A, B):
    """
    Helper: compute tr(AB) with shape checks.

    Shapes:
      A: (m, n), B: (n, m)  -> AB is (m, m), OK
      BA is (n, n)

    We verify that both AB and BA are square as required to take trace.
    """
    ensure_matrix(A)
    ensure_matrix(B)
    m, n = A.shape
    n2, m2 = B.shape
    if n != n2:
        raise ValueError("Inner dimensions must match for AB: %r @ %r" % (A.shape, B.shape))
    # AB is (m, m) only if B has shape (n, m)
    if m2 != m:
        raise ValueError("For tr(AB), B must be (n, m) so AB is square; got B.shape=%r" % (B.shape,))
    AB = A @ B
    _require_square(AB, name="AB")
    return float(np.trace(AB))


def _trace_of_product_three(A, B, C):
    """
    Helper: compute tr(ABC) with shape checks.

    Shapes:
      A: (m, n), B: (n, p), C: (p, m)  -> ABC is (m, m), OK
    """
    ensure_matrix(A)
    ensure_matrix(B)
    ensure_matrix(C)
    m, n = A.shape
    n2, p = B.shape
    p2, m2 = C.shape
    if n != n2 or p != p2 or m != m2:
        raise ValueError(
            "For tr(ABC), shapes must be A:(m,n), B:(n,p), C:(p,m). "
            "Got A=%r, B=%r, C=%r" % (A.shape, B.shape, C.shape)
        )
    ABC = (A @ B) @ C
    _require_square(ABC, name="ABC")
    return float(np.trace(ABC))


def trace_cyclic_equal(A, B, C=None, tol=1e-8):
    """
    Check the cyclic property of trace.

    If C is None:
      - Verify tr(AB) == tr(BA), given shapes A:(m,n), B:(n,m).

    If C is not None:
      - Verify tr(ABC) == tr(BCA) == tr(CAB),
        given shapes A:(m,n), B:(n,p), C:(p,m).

    Returns
    -------
    True if all required equalities hold within absolute tolerance 'tol',
    otherwise False. Raises helpful shape errors if products are invalid.
    """
    tol = float(tol)

    if C is None:
        # Two-factor cyclic property: tr(AB) = tr(BA)
        # Compute both with explicit checks
        t_ab = _trace_of_product_two(A, B)

        # For BA: B:(n,m), A:(m,n) -> BA is (n,n) square
        # The helper expects first arg (m,n), second (n,m),
        # so swap roles by reinterpreting shapes: use B as "A", A as "B".
        t_ba = _trace_of_product_two(B, A)
        return abs(t_ab - t_ba) <= tol

    else:
        # Three-factor cyclic property: tr(ABC) = tr(BCA) = tr(CAB)
        t_abc = _trace_of_product_three(A, B, C)

        # BCA: B:(n,p), C:(p,m), A:(m,n)
        t_bca = _trace_of_product_three(B, C, A)

        # CAB: C:(p,m), A:(m,n), B:(n,p)
        t_cab = _trace_of_product_three(C, A, B)

        return (abs(t_abc - t_bca) <= tol) and (abs(t_abc - t_cab) <= tol)
