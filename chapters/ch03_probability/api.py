"""
Chapter 3 — Probability & Information Theory (public API, NumPy-only)

One-stop import point for §3.5..§3.10.
"""

from .s3_5 import *   # conditionals, marginals, law of total probability
from .s3_6 import *   # marginalize_nd, chain rule checker
from .s3_7 import *   # expectations, variances, covariance
from .s3_8 import *   # entropy, cross-entropy, KL
from .s3_9 import *   # NLL (categorical/binary), L2, MAP helper
from .s3_10 import *  # logsumexp, softmax/log_softmax, CE/BCE, sigmoid utils

__all__ = [name for name in globals().keys() if not name.startswith("_")]
