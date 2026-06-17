import numpy as np
from ldcov.io.blockmatrix.reader import HailBlockMatrixReader
from tests.helpers import write_synthetic_bm


def test_reader_metadata(tmp_path):
    b00 = np.full((4, 4), 1.0)
    write_synthetic_bm(str(tmp_path / "m.bm"), {(0, 0): b00}, 4, 4, 4)
    r = HailBlockMatrixReader(str(tmp_path / "m.bm"))
    assert r.n_rows == 4 and r.n_cols == 4 and r.block_size == 4
    assert r.partitioner.n_block_rows == 1


def test_reader_read_block(tmp_path):
    b00 = np.arange(16, dtype=np.float64).reshape(4, 4)
    b01 = np.arange(16, 32, dtype=np.float64).reshape(4, 4)
    write_synthetic_bm(str(tmp_path / "m.bm"), {(0, 0): b00, (0, 1): b01}, 8, 8, 4)
    r = HailBlockMatrixReader(str(tmp_path / "m.bm"))
    np.testing.assert_array_equal(r.read_block(0, 0), b00)
    np.testing.assert_array_equal(r.read_block(0, 1), b01)


def test_reader_missing_block_returns_none(tmp_path):
    b00 = np.zeros((4, 4))
    write_synthetic_bm(str(tmp_path / "m.bm"), {(0, 0): b00}, 8, 8, 4)
    r = HailBlockMatrixReader(str(tmp_path / "m.bm"))
    assert r.read_block(1, 1) is None  # not stored


def test_reader_block_cache(tmp_path):
    b00 = np.ones((4, 4))
    write_synthetic_bm(str(tmp_path / "m.bm"), {(0, 0): b00}, 4, 4, 4)
    r = HailBlockMatrixReader(str(tmp_path / "m.bm"), block_cache=2)
    a = r.read_block(0, 0)
    b = r.read_block(0, 0)
    assert a is b  # second read served from cache (same object)


def test_reader_read_block_thread_safe(tmp_path):
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    from ldcov.io.blockmatrix.reader import HailBlockMatrixReader
    from tests.helpers import write_synthetic_bm

    # 4 distinct blocks along the diagonal+upper of a 2x2 block grid
    blocks = {
        (0, 0): np.full((4, 4), 1.0),
        (0, 1): np.full((4, 4), 2.0),
        (1, 1): np.full((4, 4), 3.0),
    }
    write_synthetic_bm(str(tmp_path / "m.bm"), blocks, 8, 8, 4)
    r = HailBlockMatrixReader(str(tmp_path / "m.bm"), block_cache=8)
    pairs = [(0, 0), (0, 1), (1, 1)] * 4
    with ThreadPoolExecutor(max_workers=4) as ex:
        got = list(ex.map(lambda ij: (ij, r.read_block(*ij)), pairs))
    for ij, block in got:
        np.testing.assert_array_equal(block, blocks[ij])
