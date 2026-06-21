"""Reasoning token parser for thinking-model outputs.

Splits completion text into a reasoning portion (inside start/end tags such as
``<think>…</think>``) and an answer portion (everything outside).  Token counts
are estimated from the char-length fraction of the backend-reported output_tokens
so no tokenizer dependency is introduced.

Usage::

    from llm_inference_benchmark.reasoning import parse_reasoning, reasoning_stats

    thinking, answer = parse_reasoning(text, "<think>", "</think>")
    mean_r, mean_a, frac = reasoning_stats(texts, token_counts, "<think>", "</think>")
"""

from __future__ import annotations

import statistics


def parse_reasoning(text: str, start_tag: str, end_tag: str) -> tuple[str, str]:
    """Split *text* into ``(reasoning, answer)`` using *start_tag* / *end_tag*.

    Rules:
    - If *start_tag* is not found: ``("", text)``.
    - If *start_tag* is found but *end_tag* is absent: entire text after the
      opening tag is treated as reasoning and answer is ``""``.
    - Multiple occurrences: all text between any matching tag pair is
      accumulated into *reasoning*; everything outside is accumulated into
      *answer*.
    """
    if not start_tag or start_tag not in text:
        return "", text

    reasoning_parts: list[str] = []
    answer_parts: list[str] = []
    pos = 0

    while pos < len(text):
        s = text.find(start_tag, pos)
        if s == -1:
            answer_parts.append(text[pos:])
            break
        # Text before the opening tag is answer text.
        if s > pos:
            answer_parts.append(text[pos:s])
        inner_start = s + len(start_tag)
        e = text.find(end_tag, inner_start)
        if e == -1:
            # No closing tag — rest of text is reasoning.
            reasoning_parts.append(text[inner_start:])
            pos = len(text)
        else:
            reasoning_parts.append(text[inner_start:e])
            pos = e + len(end_tag)

    return "".join(reasoning_parts), "".join(answer_parts)


def _reasoning_fraction(text: str, start_tag: str, end_tag: str) -> float:
    """Return the fraction [0, 1] of content chars that belong to the reasoning portion.

    Computed as ``reasoning_chars / (reasoning_chars + answer_chars)`` so tag
    bytes are excluded from the denominator, giving an intuitive content ratio.
    Returns 0.0 when there is no content at all.
    """
    reasoning, answer = parse_reasoning(text, start_tag, end_tag)
    total = len(reasoning) + len(answer)
    return len(reasoning) / total if total > 0 else 0.0


def reasoning_stats(
    texts: list[str],
    output_token_counts: list[int],
    start_tag: str,
    end_tag: str,
) -> tuple[float, float, float]:
    """Compute per-run reasoning token statistics.

    Returns ``(mean_reasoning_tokens, mean_answer_tokens, reasoning_fraction)``
    where token counts are estimated by applying the char-length fraction to
    the backend-reported *output_token_counts*.

    When *texts* and *output_token_counts* have different lengths, or either is
    empty, returns ``(0.0, 0.0, 0.0)``.
    """
    if not texts or len(texts) != len(output_token_counts):
        return 0.0, 0.0, 0.0

    r_tokens: list[float] = []
    a_tokens: list[float] = []
    fracs: list[float] = []

    for text, n_out in zip(texts, output_token_counts, strict=True):
        frac = _reasoning_fraction(text, start_tag, end_tag)
        r_tok = n_out * frac
        a_tok = n_out * (1.0 - frac)
        r_tokens.append(r_tok)
        a_tokens.append(a_tok)
        fracs.append(frac)

    return (
        statistics.mean(r_tokens),
        statistics.mean(a_tokens),
        statistics.mean(fracs),
    )
