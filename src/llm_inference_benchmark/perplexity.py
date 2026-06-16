"""Perplexity computation for backends with token-level log-probability access.

Definition: corpus-level perplexity = exp(total_negative_log_likelihood / total_tokens),
pooling negative log-likelihood across all evaluated tokens rather than averaging
per-text perplexities. This is the standard definition and avoids biasing the result
toward short completions.

Scope: this measures self-perplexity — how confident the model is in the tokens it
itself generated (teacher-forced). It is an intrinsic fluency signal, not a task
correctness check, and is only comparable across runs sharing the same tokenizer
and model family. See docs/metrics.md for the full caveats.
"""

from __future__ import annotations

import math


def perplexity_from_nll(total_nll: float, total_tokens: int) -> float:
    """Return corpus-level perplexity from a summed negative log-likelihood.

    Raises ValueError when total_tokens <= 0 (nothing was scored).
    """
    if total_tokens <= 0:
        raise ValueError("total_tokens must be positive to compute perplexity")
    return math.exp(total_nll / total_tokens)
