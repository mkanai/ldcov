import io

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch

from ldcov.io.bcor_reader import BcorReader
from ldcov.io.bcor_writer import save_bcor


@pytest.fixture
def local_bcor_bytes(tmp_path):
    n = 12
    rng = np.random.default_rng(42)
    corr = rng.uniform(-0.5, 0.5, size=(n, n))
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1.0)
    variant_info = pd.DataFrame({
        "rsid": [f"rs{i}" for i in range(n)],
        "chrom": ["1"] * n,
        "pos": list(range(1, n + 1)),
        "ref": ["A"] * n,
        "alt": ["G"] * n,
    })
    out = tmp_path / "fixture.bcor"
    save_bcor(corr, str(out), variant_info=variant_info, n_samples=50)
    return {
        "bcor_bytes": out.read_bytes(),
        "idx_bytes": Path(str(out) + ".idx").read_bytes(),
        "corr": corr,
        "variant_info": variant_info,
    }


class _FakeGCSFile:
    """Minimal gcsfs-compatible file object backed by an in-memory buffer."""
    def __init__(self, data: bytes):
        self._buf = io.BytesIO(data)

    def read(self, n: int = -1) -> bytes:
        return self._buf.read(n) if n != -1 else self._buf.read()

    def seek(self, offset: int, whence: int = 0):
        self._buf.seek(offset, whence)

    def tell(self) -> int:
        return self._buf.tell()

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class _FakeGCSFS:
    def __init__(self, contents: dict):
        # contents: {gs://...: bytes}
        self._contents = contents

    def info(self, path):
        return {"size": len(self._contents[path])}

    def open(self, path, mode="rb"):
        assert mode == "rb"
        return _FakeGCSFile(self._contents[path])

    def exists(self, path):
        return path in self._contents


def test_reader_loads_sidecar_when_present(local_bcor_bytes, tmp_path):
    out = tmp_path / "with_idx.bcor"
    out.write_bytes(local_bcor_bytes["bcor_bytes"])
    (tmp_path / "with_idx.bcor.idx").write_bytes(local_bcor_bytes["idx_bytes"])

    reader = BcorReader(str(out))
    assert reader.has_index
    assert reader.index.n_snps == len(local_bcor_bytes["variant_info"])


def test_reader_works_without_sidecar(local_bcor_bytes, tmp_path):
    out = tmp_path / "no_idx.bcor"
    out.write_bytes(local_bcor_bytes["bcor_bytes"])
    # Intentionally no .idx file.

    reader = BcorReader(str(out))
    assert not reader.has_index
    assert reader.index is None
    # Still fully functional for index-based reads.
    loaded = reader.read_corr()
    np.testing.assert_array_almost_equal(loaded, local_bcor_bytes["corr"], decimal=4)


def test_reader_rejects_sidecar_when_parent_truncated(local_bcor_bytes, tmp_path, caplog):
    """If the .bcor has been truncated after the sidecar was written, the on-disk file
    size mismatches the sidecar's recorded size, so the sidecar must be rejected."""
    bcor_path = tmp_path / "trunc.bcor"
    bcor_bytes = local_bcor_bytes["bcor_bytes"]
    # Truncate the last 100 bytes of the correlation block.
    bcor_path.write_bytes(bcor_bytes[: len(bcor_bytes) - 100])
    (tmp_path / "trunc.bcor.idx").write_bytes(local_bcor_bytes["idx_bytes"])

    import logging
    with caplog.at_level(logging.WARNING):
        reader = BcorReader(str(bcor_path))

    assert not reader.has_index, "sidecar must be rejected when parent is truncated"
    assert any("file_size mismatch" in r.message.lower() for r in caplog.records)


def test_reader_rejects_stale_sidecar(local_bcor_bytes, tmp_path, caplog):
    """A sidecar belonging to a different .bcor (header mismatch) is ignored,
    and the reader logs a warning rather than silently returning wrong rows."""
    n = local_bcor_bytes["corr"].shape[0]
    rng = np.random.default_rng(0)
    other_corr = rng.uniform(-0.5, 0.5, size=(n, n))
    other_corr = (other_corr + other_corr.T) / 2
    np.fill_diagonal(other_corr, 1.0)
    other_vi = pd.DataFrame({
        "rsid": [f"OTHERnameLong{i}" for i in range(n)],  # different rsids AND different lengths
        "chrom": ["1"] * n, "pos": list(range(1, n + 1)),
        "ref": ["A"] * n, "alt": ["G"] * n,
    })
    other_path = tmp_path / "other.bcor"
    save_bcor(other_corr, str(other_path), variant_info=other_vi, n_samples=50)

    foreign_idx_bytes = (tmp_path / "other.bcor.idx").read_bytes()
    bcor_path = tmp_path / "mismatch.bcor"
    bcor_path.write_bytes(local_bcor_bytes["bcor_bytes"])
    (tmp_path / "mismatch.bcor.idx").write_bytes(foreign_idx_bytes)

    import logging
    with caplog.at_level(logging.WARNING):
        reader = BcorReader(str(bcor_path))

    assert not reader.has_index, "stale sidecar must be rejected"
    assert any("mismatch" in rec.message.lower() for rec in caplog.records), (
        f"expected mismatch warning; got: {[r.message for r in caplog.records]}"
    )
    loaded = reader.read_corr()
    np.testing.assert_array_almost_equal(loaded, local_bcor_bytes["corr"], decimal=4)


def test_read_corr_by_rsid_local(local_bcor_bytes, tmp_path):
    out = tmp_path / "idx.bcor"
    out.write_bytes(local_bcor_bytes["bcor_bytes"])
    (tmp_path / "idx.bcor.idx").write_bytes(local_bcor_bytes["idx_bytes"])

    reader = BcorReader(str(out))
    rsids = ["rs1", "rs3", "rs7", "rs11"]
    subset, subset_meta = reader.read_corr_by_rsid(rsids)

    vi = local_bcor_bytes["variant_info"]
    row_idx = [int(vi.index[vi["rsid"] == r][0]) for r in rsids]
    expected = local_bcor_bytes["corr"][np.ix_(row_idx, row_idx)]
    np.testing.assert_array_almost_equal(subset, expected, decimal=4)
    assert list(subset_meta["rsid"]) == rsids


def test_read_corr_by_rsid_missing_raise(local_bcor_bytes, tmp_path):
    out = tmp_path / "idx.bcor"
    out.write_bytes(local_bcor_bytes["bcor_bytes"])
    (tmp_path / "idx.bcor.idx").write_bytes(local_bcor_bytes["idx_bytes"])

    reader = BcorReader(str(out))
    with pytest.raises(KeyError, match="not found in bcor"):
        reader.read_corr_by_rsid(["rs1", "rs_missing"])


def test_read_corr_by_rsid_missing_skip(local_bcor_bytes, tmp_path):
    out = tmp_path / "idx.bcor"
    out.write_bytes(local_bcor_bytes["bcor_bytes"])
    (tmp_path / "idx.bcor.idx").write_bytes(local_bcor_bytes["idx_bytes"])

    reader = BcorReader(str(out))
    subset, subset_meta = reader.read_corr_by_rsid(
        ["rs1", "rs_missing", "rs3"], missing="skip"
    )
    assert list(subset_meta["rsid"]) == ["rs1", "rs3"]
    assert subset.shape == (2, 2)


def test_read_corr_by_rsid_two_lists(local_bcor_bytes, tmp_path):
    """rsids2 should produce a non-square (len(rsids), len(rsids2)) matrix."""
    out = tmp_path / "two.bcor"
    out.write_bytes(local_bcor_bytes["bcor_bytes"])
    (tmp_path / "two.bcor.idx").write_bytes(local_bcor_bytes["idx_bytes"])

    reader = BcorReader(str(out))
    rsids_a = ["rs1", "rs3"]
    rsids_b = ["rs5", "rs7", "rs9"]
    subset, meta_a = reader.read_corr_by_rsid(rsids_a, rsids2=rsids_b)
    assert subset.shape == (len(rsids_a), len(rsids_b))
    assert list(meta_a["rsid"]) == rsids_a

    vi = local_bcor_bytes["variant_info"]
    rows_a = [int(vi.index[vi["rsid"] == r][0]) for r in rsids_a]
    rows_b = [int(vi.index[vi["rsid"] == r][0]) for r in rsids_b]
    expected = local_bcor_bytes["corr"][np.ix_(rows_a, rows_b)]
    np.testing.assert_array_almost_equal(subset, expected, decimal=4)


def test_read_corr_by_rsid_extended_format(tmp_path):
    """Extended-format (non-unit diagonal) partial read by rsid."""
    n = 8
    rng = np.random.default_rng(7)
    corr = rng.uniform(-0.4, 0.4, size=(n, n))
    corr = (corr + corr.T) / 2
    diag = 0.7 + 0.3 * rng.random(n)
    np.fill_diagonal(corr, diag)
    variant_info = pd.DataFrame({
        "rsid": [f"rsX{i}" for i in range(n)],
        "chrom": ["1"] * n,
        "pos": list(range(1, n + 1)),
        "ref": ["A"] * n,
        "alt": ["G"] * n,
    })
    out = tmp_path / "ext.bcor"
    save_bcor(corr, str(out), variant_info=variant_info, n_samples=42)

    reader = BcorReader(str(out))
    assert reader.is_extended
    assert reader.has_index

    rsids = ["rsX0", "rsX3", "rsX6"]
    subset, _ = reader.read_corr_by_rsid(rsids)
    rows = [0, 3, 6]
    expected = corr[np.ix_(rows, rows)]
    np.testing.assert_array_almost_equal(subset, expected, decimal=4)
    np.testing.assert_array_almost_equal(np.diag(subset), diag[rows], decimal=4)


def test_read_corr_by_rsid_sparse_query_merging(local_bcor_bytes, tmp_path):
    """Sparse rsid query should not blow up; result should still be correct."""
    out = tmp_path / "sparse.bcor"
    out.write_bytes(local_bcor_bytes["bcor_bytes"])
    (tmp_path / "sparse.bcor.idx").write_bytes(local_bcor_bytes["idx_bytes"])

    reader = BcorReader(str(out))
    rsids = ["rs0", "rs11"]  # row 0 and row 11 (max span)
    subset, _ = reader.read_corr_by_rsid(rsids, range_merge_gap=0)
    vi = local_bcor_bytes["variant_info"]
    rows = [int(vi.index[vi["rsid"] == r][0]) for r in rsids]
    expected = local_bcor_bytes["corr"][np.ix_(rows, rows)]
    np.testing.assert_array_almost_equal(subset, expected, decimal=4)


def test_gcs_meta_block_not_downloaded_at_open_when_sidecar_present(local_bcor_bytes):
    """With a sidecar, the reader should NOT eagerly download the meta block on open."""
    gcs_bcor = "gs://fake-bucket/big.bcor"
    gcs_idx = "gs://fake-bucket/big.bcor.idx"

    fetched = []

    class RecordingFake(_FakeGCSFS):
        def open(self, path, mode="rb"):
            f = super().open(path, mode)
            orig_read = f.read

            def _wrapped_read(n=-1):
                pos = f.tell()
                data = orig_read(n)
                fetched.append((path, pos, len(data)))
                return data

            f.read = _wrapped_read  # type: ignore[assignment]
            return f

    contents = {
        gcs_bcor: local_bcor_bytes["bcor_bytes"],
        gcs_idx: local_bcor_bytes["idx_bytes"],
    }

    with patch("gcsfs.GCSFileSystem", return_value=RecordingFake(contents)):
        reader = BcorReader(gcs_bcor)

    bcor_bytes_read = sum(n for p, _, n in fetched if p == gcs_bcor)
    # On open: exactly the 32-byte header. The meta block must NOT be touched.
    assert bcor_bytes_read == 32, (
        f"Expected header-only fetch (32 bytes) on open when sidecar is present, "
        f"got {bcor_bytes_read} bytes"
    )


def test_gcs_get_meta_uses_single_contiguous_read(local_bcor_bytes):
    """When get_meta() is finally called over GCS with a sidecar, it should issue ONE
    big ranged read for the meta block — not one per record."""
    gcs_bcor = "gs://fake-bucket/big.bcor"
    gcs_idx = "gs://fake-bucket/big.bcor.idx"

    fetched = []

    class RecordingFake(_FakeGCSFS):
        def open(self, path, mode="rb"):
            f = super().open(path, mode)
            orig_read = f.read

            def _wrapped_read(n=-1):
                pos = f.tell()
                data = orig_read(n)
                fetched.append((path, pos, len(data)))
                return data

            f.read = _wrapped_read  # type: ignore[assignment]
            return f

    contents = {
        gcs_bcor: local_bcor_bytes["bcor_bytes"],
        gcs_idx: local_bcor_bytes["idx_bytes"],
    }

    with patch("gcsfs.GCSFileSystem", return_value=RecordingFake(contents)):
        reader = BcorReader(gcs_bcor)
        # Drain accumulated reads from open.
        before = list(fetched)
        meta = reader.get_meta()

    new_reads = [r for r in fetched if r not in before]
    # The eager meta path should issue exactly 1 read against the .bcor for the meta block.
    bcor_reads = [r for r in new_reads if r[0] == gcs_bcor]
    assert len(bcor_reads) == 1, (
        f"Expected exactly 1 ranged read for the meta block; got {len(bcor_reads)}: {bcor_reads}"
    )
    assert len(meta) == reader.n_snps


@pytest.mark.integration
def test_bcor_reader_against_public_gcs_fixture():
    """Read the public bcor + sidecar fixture from gs://gcs-anndata-test/ldcov/data/.

    Mirrors the test_load_bgen_from_gcs pattern: skips on any GCS access failure
    so it doesn't break developers without credentials. Run with `pytest -m integration`.

    Verifies end-to-end:
      - sidecar auto-loads
      - full matrix read returns expected shape
      - read_corr_by_rsid returns a submatrix consistent with the full read
    """
    gcs_path = "gs://gcs-anndata-test/ldcov/data/data.bcor"
    try:
        reader = BcorReader(gcs_path)
    except Exception as e:
        pytest.skip(f"public GCS fixture not accessible: {e}")

    assert reader.has_index, "sidecar must be present alongside the public fixture"
    assert reader.n_snps == 55  # known fixture shape

    full = reader.read_corr()
    assert full.shape == (55, 55)

    rsids = ["rs1", "rs5", "rs55"]
    subset, meta = reader.read_corr_by_rsid(rsids)
    assert subset.shape == (3, 3)
    assert list(meta["rsid"]) == rsids

    # Self-consistency: the partial read must equal the corresponding slice of the full read.
    rows = [reader.index.rsid_to_row(r) for r in rsids]
    np.testing.assert_array_equal(subset, full[np.ix_(rows, rows)])
