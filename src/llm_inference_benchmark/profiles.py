"""Named workload profiles for reproducible optimization experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkloadProfile:
    name: str
    prompts_file: str
    input_length: str
    output_length: str
    description: str


_PROFILES: dict[str, WorkloadProfile] = {
    "short_chat": WorkloadProfile(
        name="short_chat",
        prompts_file="data/prompts/short_chat.txt",
        input_length="short",
        output_length="short",
        description=(
            "Short conversational Q&A. Typical chat-style input with a brief answer expected. "
            "Use this profile to benchmark latency for interactive use cases."
        ),
    ),
    "summarization": WorkloadProfile(
        name="summarization",
        prompts_file="data/prompts/summarization.txt",
        input_length="medium",
        output_length="short",
        description=(
            "Medium-length input passages with a request to summarize to 2-3 sentences. "
            "Input is longer than short_chat; output is concise."
        ),
    ),
    "code_completion": WorkloadProfile(
        name="code_completion",
        prompts_file="data/prompts/code_completion.txt",
        input_length="short",
        output_length="medium",
        description=(
            "Function-signature completions. Short prompts, medium-length code output expected. "
            "Tests throughput on code generation tasks."
        ),
    ),
    "long_context_smoke": WorkloadProfile(
        name="long_context_smoke",
        prompts_file="data/prompts/long_context_smoke.txt",
        input_length="long",
        output_length="short",
        description=(
            "Long context passages with a comprehension question. "
            "Stresses the prefill pass; output is a short answer."
        ),
    ),
}

PROFILE_NAMES: frozenset[str] = frozenset(_PROFILES)


def get_profile(name: str) -> WorkloadProfile:
    """Return the named profile or raise ValueError with valid options listed."""
    if name not in _PROFILES:
        valid = ", ".join(sorted(_PROFILES))
        raise ValueError(f"Unknown workload profile {name!r}. Valid profiles: {valid}")
    return _PROFILES[name]
