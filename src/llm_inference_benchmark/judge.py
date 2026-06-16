"""LLM-as-judge scoring for backends with token-level logit access.

Definition: for each (prompt, completion) pair, the model under test is asked a
fixed yes/no question about its own completion in one forward pass. The judge
score for that pair is P(yes), recovered from the competing "Yes"/"No" logits at
the final position via a 2-way softmax. The corpus-level judge score is the mean
P(yes) across all scored pairs.

Scope: the model judges its own output with a single fixed question. This is not
a calibrated preference model, not a substitute for human or third-party
evaluation, and not comparable across different judge models. See
docs/metrics.md for the full caveats.
"""

from __future__ import annotations

import math


def probability_from_yes_no_logits(yes_logit: float, no_logit: float) -> float:
    """Return P(yes) from competing 'Yes'/'No' logits via a 2-way softmax.

    Equivalent to sigmoid(yes_logit - no_logit).
    """
    return 1.0 / (1.0 + math.exp(no_logit - yes_logit))


def judge_score_from_probabilities(probabilities: list[float]) -> float:
    """Return the corpus-level judge score: mean P(yes) across scored completions.

    Raises ValueError when probabilities is empty (nothing was scored).
    """
    if not probabilities:
        raise ValueError("probabilities must be non-empty to compute a judge score")
    return sum(probabilities) / len(probabilities)
