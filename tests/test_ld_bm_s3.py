import os

import numpy as np
import pytest

pytest.importorskip("s3fs")  # skip if the s3 extra is not installed

pytestmark = pytest.mark.skipif(
    os.environ.get("LDCOV_RUN_S3_TESTS") != "1",
    reason="set LDCOV_RUN_S3_TESTS=1 to run network-bound Pan-UKB tests",
)

PANUKB_BM = "s3://pan-ukb-us-east-1/ld_release/UKBB.EUR.ldadj.bm"


def test_read_metadata_from_panukb():
    from ldcov.io.blockmatrix.reader import HailBlockMatrixReader

    reader = HailBlockMatrixReader(PANUKB_BM, block_cache=1)  # anon by default
    assert reader.n_rows == 23960350
    assert reader.block_size == 4096
    block = reader.read_block(0, 0)  # diagonal block is always stored
    assert block is not None
    assert block.shape == (4096, 4096)


def test_assemble_small_range_symmetric_from_panukb():
    from ldcov.io.blockmatrix.reader import HailBlockMatrixReader
    from ldcov.ld_bm.extract import _assemble

    reader = HailBlockMatrixReader(PANUKB_BM, block_cache=2)
    m = _assemble(reader, list(range(0, 50)))  # contiguous range inside block (0, 0)
    assert m.shape == (50, 50)
    assert np.allclose(m, m.T, equal_nan=True)
    assert np.all(np.isfinite(np.diag(m)))
