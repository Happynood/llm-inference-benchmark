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
    suggest_fitting_variants,
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


class TestSuggestFittingVariants:
    def _make_info(self, *files: tuple[str, int]) -> MagicMock:
        siblings = [_sibling(name, size) for name, size in files]
        return _model_info(*siblings)

    def test_returns_files_that_fit(self) -> None:
        gb = 1024**3
        info = self._make_info(
            ("model-Q4_K_M.gguf", int(1.5 * gb)),
            ("model-Q8_0.gguf", int(4.0 * gb)),
        )
        with patch("llm_inference_benchmark.puller.hf_model_info", return_value=info):
            result = suggest_fitting_variants("owner/repo", vram_gb=3.0)
        names = [r[0] for r in result]
        assert "model-Q4_K_M.gguf" in names
        assert "model-Q8_0.gguf" not in names

    def test_sorted_descending_by_size(self) -> None:
        gb = 1024**3
        info = self._make_info(
            ("model-Q4_K_M.gguf", int(1.5 * gb)),
            ("model-Q5_K_M.gguf", int(2.0 * gb)),
            ("model-Q8_0.gguf", int(4.0 * gb)),
        )
        with patch("llm_inference_benchmark.puller.hf_model_info", return_value=info):
            result = suggest_fitting_variants("owner/repo", vram_gb=3.0)
        sizes = [r[1] for r in result]
        assert sizes == sorted(sizes, reverse=True)

    def test_excludes_unknown_size(self) -> None:
        info = self._make_info(("model-Q4_K_M.gguf", 0))
        with patch("llm_inference_benchmark.puller.hf_model_info", return_value=info):
            result = suggest_fitting_variants("owner/repo", vram_gb=10.0)
        assert result == []

    def test_all_fit(self) -> None:
        gb = 1024**3
        info = self._make_info(
            ("model-Q4_K_M.gguf", int(1.0 * gb)),
            ("model-Q8_0.gguf", int(2.0 * gb)),
        )
        with patch("llm_inference_benchmark.puller.hf_model_info", return_value=info):
            result = suggest_fitting_variants("owner/repo", vram_gb=8.0)
        assert len(result) == 2


def _make_hw_profile(vram_gb: float | None) -> MagicMock:
    profile = MagicMock()
    profile.vram_gb = vram_gb
    return profile


class TestWarnVram:
    """Integration-style tests for the VRAM warning printed by pull_gguf."""

    def _run_pull(
        self,
        tmp_path: Path,
        file_size_bytes: int,
        vram_gb: float | None,
        all_files: list[tuple[str, int]],
        capsys: pytest.CaptureFixture[str],
    ) -> str:
        content = b"x"
        sha = hashlib.sha256(content).hexdigest()
        downloaded = tmp_path / "model-Q8_0.gguf"

        def fake_download(**_: object) -> str:
            downloaded.write_bytes(content)
            return str(downloaded)

        siblings = [_sibling(name, size) for name, size in all_files]
        info = _model_info(*siblings, _sibling("model-Q8_0.gguf", file_size_bytes, sha))

        hw_patch = patch(
            "llm_inference_benchmark.puller._hw.detect",
            return_value=_make_hw_profile(vram_gb),
        )
        dl_patch = patch(
            "llm_inference_benchmark.puller.hf_hub_download",
            side_effect=fake_download,
        )
        with patch("llm_inference_benchmark.puller.hf_model_info", return_value=info):
            with hw_patch, dl_patch:
                pull_gguf("owner/repo", "Q8_0", dest_dir=tmp_path)

        return capsys.readouterr().out

    def test_no_gpu_no_warning(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        size = int(3.0 * 1024**3)
        out = self._run_pull(tmp_path, size, None, [], capsys)
        assert "Warning" not in out

    def test_fits_no_warning(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        size = int(2.0 * 1024**3)
        out = self._run_pull(tmp_path, size, 8.0, [], capsys)
        assert "Warning" not in out

    def test_exceeds_vram_prints_warning(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        size = int(5.0 * 1024**3)
        out = self._run_pull(tmp_path, size, 4.0, [], capsys)
        assert "Warning" in out
        assert "VRAM" in out

    def test_exceeds_vram_lists_alternatives(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        size = int(5.0 * 1024**3)
        alternatives = [("model-Q4_K_M.gguf", int(2.0 * 1024**3))]
        out = self._run_pull(tmp_path, size, 4.0, alternatives, capsys)
        assert "Q4_K_M" in out
        assert "llm-bench pull" in out

    def test_no_fitting_alternative_no_suggestion_list(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        size = int(5.0 * 1024**3)
        # All other files are also too large
        alternatives = [("model-Q6_K.gguf", int(4.5 * 1024**3))]
        out = self._run_pull(tmp_path, size, 4.0, alternatives, capsys)
        assert "Warning" in out
        assert "llm-bench pull" not in out


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
