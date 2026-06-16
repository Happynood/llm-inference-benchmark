"""LLM inference benchmark — reproducible backend comparisons."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("llm-inference-benchmark")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
