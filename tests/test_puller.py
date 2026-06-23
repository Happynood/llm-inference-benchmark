from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_inference_benchmark.puller import (
    _find_gguf_sibling,
    _sha256_file,
    pull_gguf,
    pull_transformers,
)


def _sibling(filename: str, size: int = 0, sha256: str | None = None) -> MagicMock:
    s = MagicMock()
    s.rfilename = filename
    s.size = size
    s.lfs = MagicMock(sha256=sha256) if sha256 else None
    return s


def _model_info(*siblings: MagicMock) -> MagicMock:
    info = MagicMock()
    info.siblings = list(siblings)
    return info


class TestSha256File:
    def test_matches_hashlib(self, tmp_path: Path) -> None:
        content = b"hello world"
        f = tmp_path / "file.bin"
        f.write_bytes(content)
        assert _sha256_file(f) == hashlib.sha256(content).hexdigest()


class TestFindGgufSibling:
    def test_finds_matching_quant(self) -> None:
        info = _model_info(
            _sibling("model-Q4_K_M.gguf", 100, "abc123"),
            _sibling("model-Q8_0.gguf", 200),
        )
        with patch("llm_inference_benchmark.puller.hf_model_info", return_value=info):
            name, size, sha256 = _find_gguf_sibling("owner/repo", "Q4_K_M", None)
        assert name == "model-Q4_K_M.gguf"
        assert size == 100
        assert sha256 == "abc123"

    def test_raises_when_no_match(self) -> None:
        info = _model_info(_sibling("model-Q8_0.gguf"))
        with patch("llm_inference_benchmark.puller.hf_model_info", return_value=info):
            with pytest.raises(ValueError, match="No GGUF file matching"):
                _find_gguf_sibling("owner/repo", "Q4_K_M", None)

    def test_prefers_shortest_filename_over_shards(self) -> None:
        info = _model_info(
            _sibling("model-Q4_K_M-00001-of-00003.gguf"),
            _sibling("model-Q4_K_M.gguf"),
        )
        with patch("llm_inference_benchmark.puller.hf_model_info", return_value=info):
            name, _, _ = _find_gguf_sibling("owner/repo", "Q4_K_M", None)
        assert name == "model-Q4_K_M.gguf"

    def test_case_insensitive_quant_match(self) -> None:
        info = _model_info(_sibling("model-q4_k_m.gguf", 50))
        with patch("llm_inference_benchmark.puller.hf_model_info", return_value=info):
            name, _, _ = _find_gguf_sibling("owner/repo", "Q4_K_M", None)
        assert name == "model-q4_k_m.gguf"


class TestPullGguf:
    def test_size_limit_aborts_before_download(self, tmp_path: Path) -> None:
        info = _model_info(_sibling("model-Q4_K_M.gguf", size=int(15 * 1024**3)))
        with patch("llm_inference_benchmark.puller.hf_model_info", return_value=info):
            with patch("llm_inference_benchmark.puller.hf_hub_download") as mock_dl:
                with pytest.raises(ValueError, match="exceeds --max-size-gb"):
                    pull_gguf("owner/repo", "Q4_K_M", dest_dir=tmp_path, max_size_gb=10.0)
        mock_dl.assert_not_called()

    def test_skip_when_file_exists_with_correct_hash(self, tmp_path: Path) -> None:
        content = b"model weights"
        expected = hashlib.sha256(content).hexdigest()
        local = tmp_path / "model-Q4_K_M.gguf"
        local.write_bytes(content)

        info = _model_info(_sibling("model-Q4_K_M.gguf", size=len(content), sha256=expected))
        with patch("llm_inference_benchmark.puller.hf_model_info", return_value=info):
            with patch("llm_inference_benchmark.puller.hf_hub_download") as mock_dl:
                result = pull_gguf("owner/repo", "Q4_K_M", dest_dir=tmp_path)

        mock_dl.assert_not_called()
        assert result.skipped is True
        assert result.path == local

    def test_hash_mismatch_deletes_downloaded_file(self, tmp_path: Path) -> None:
        content = b"downloaded content"
        bad_hash = "a" * 64
        downloaded = tmp_path / "model-Q4_K_M.gguf"

        def fake_download(**_: object) -> str:
            downloaded.write_bytes(content)
            return str(downloaded)

        info = _model_info(_sibling("model-Q4_K_M.gguf", size=len(content), sha256=bad_hash))
        with patch("llm_inference_benchmark.puller.hf_model_info", return_value=info):
            with patch("llm_inference_benchmark.puller.hf_hub_download", side_effect=fake_download):
                with pytest.raises(ValueError, match="Hash mismatch"):
                    pull_gguf("owner/repo", "Q4_K_M", dest_dir=tmp_path)

        assert not downloaded.exists()

    def test_successful_download_returns_result(self, tmp_path: Path) -> None:
        content = b"valid model weights"
        sha = hashlib.sha256(content).hexdigest()
        downloaded = tmp_path / "model-Q4_K_M.gguf"

        def fake_download(**_: object) -> str:
            downloaded.write_bytes(content)
            return str(downloaded)

        info = _model_info(_sibling("model-Q4_K_M.gguf", size=len(content), sha256=sha))
        with patch("llm_inference_benchmark.puller.hf_model_info", return_value=info):
            with patch("llm_inference_benchmark.puller.hf_hub_download", side_effect=fake_download):
                result = pull_gguf("owner/repo", "Q4_K_M", dest_dir=tmp_path)

        assert result.skipped is False
        assert result.backend == "gguf"
        assert result.path == downloaded

    def test_no_hash_metadata_skips_verification(self, tmp_path: Path) -> None:
        content = b"model without lfs hash"
        downloaded = tmp_path / "model-Q4_K_M.gguf"

        def fake_download(**_: object) -> str:
            downloaded.write_bytes(content)
            return str(downloaded)

        info = _model_info(_sibling("model-Q4_K_M.gguf", size=len(content), sha256=None))
        with patch("llm_inference_benchmark.puller.hf_model_info", return_value=info):
            with patch("llm_inference_benchmark.puller.hf_hub_download", side_effect=fake_download):
                result = pull_gguf("owner/repo", "Q4_K_M", dest_dir=tmp_path)

        assert result.skipped is False
        assert downloaded.exists()


class TestPullTransformers:
    def test_size_limit_aborts(self) -> None:
        info = _model_info(_sibling("model.safetensors", size=int(12 * 1024**3)))
        with patch("llm_inference_benchmark.puller.hf_model_info", return_value=info):
            with patch("llm_inference_benchmark.puller.snapshot_download") as mock_snap:
                with pytest.raises(ValueError, match="exceeds --max-size-gb"):
                    pull_transformers("owner/repo", max_size_gb=10.0)
        mock_snap.assert_not_called()

    def test_successful_snapshot_returns_result(self, tmp_path: Path) -> None:
        info = _model_info(_sibling("model.safetensors", size=int(1 * 1024**3)))
        with patch("llm_inference_benchmark.puller.hf_model_info", return_value=info):
            with patch(
                "llm_inference_benchmark.puller.snapshot_download",
                return_value=str(tmp_path),
            ):
                result = pull_transformers("owner/repo")

        assert result.backend == "transformers"
        assert result.path == tmp_path
        assert result.skipped is False
