import os
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("LDCOV_RUN_GCS_TESTS") != "1",
    reason="set LDCOV_RUN_GCS_TESTS=1 to run network-bound gnomAD tests",
)

GNOMAD_BM = (
    "gs://gcp-public-data--gnomad/release/2.1.1/ld/" "gnomad.genomes.r2.1.1.nfe.common.adj.ld.bm"
)


def test_read_one_block_from_gnomad():
    from ldcov.io.blockmatrix.reader import HailBlockMatrixReader

    reader = HailBlockMatrixReader(GNOMAD_BM, block_cache=1)
    assert reader.n_rows == 14207204
    assert reader.block_size == 4096
    block = reader.read_block(0, 0)  # diagonal block, always stored
    assert block is not None
    assert block.shape == (4096, 4096)
    # Diagonal block of an LD matrix: diagonal must be 1 (self-correlation).
    diag = np.diag(block)
    assert np.all(np.isfinite(diag))
    assert np.allclose(diag, 1.0, atol=1e-6)
    # gnomAD stores the upper triangle; off-diagonal upper entries should be non-zero.
    assert np.any(block[0, 1:] != 0.0)


def test_assemble_region_symmetric_from_gnomad():
    from ldcov.io.blockmatrix.reader import HailBlockMatrixReader
    from ldcov.ld_bm.extract import _assemble

    reader = HailBlockMatrixReader(GNOMAD_BM, block_cache=2)
    m = _assemble(reader, list(range(0, 50)))  # contiguous range inside block (0,0)
    assert m.shape == (50, 50)
    assert np.allclose(m, m.T, equal_nan=True)  # symmetric
    assert np.allclose(np.diag(m), 1.0, atol=1e-6)  # LD diagonal is 1.0
    assert m[0, 1] == pytest.approx(m[1, 0], abs=1e-9)  # mirrored off-diagonal
