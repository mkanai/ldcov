import pytest
from ldcov.io.blockmatrix.partitioner import GridPartitioner


def test_linear_id_column_major():
    # 10000x10000, block 4096 -> n_block_rows = 3
    gp = GridPartitioner(block_size=4096, n_rows=10000, n_cols=10000, maybe_filtered=None)
    assert gp.n_block_rows == 3
    assert gp.n_block_cols == 3
    assert gp.linear_id(0, 0) == 0
    assert gp.linear_id(1, 0) == 1
    assert gp.linear_id(0, 1) == 3
    assert gp.linear_id(2, 1) == 5


def test_dense_part_slot_is_identity():
    gp = GridPartitioner(block_size=4096, n_rows=10000, n_cols=10000, maybe_filtered=None)
    assert gp.part_slot(2, 1) == 5


def test_sparse_band_membership_and_slot():
    # Upper-triangular band: store (0,0),(0,1),(1,1),(1,2),(2,2) -> column-major linear ids
    # n_block_rows=3: (0,0)=0,(0,1)=3,(1,1)=4,(1,2)=7,(2,2)=8
    gp = GridPartitioner(
        block_size=4096, n_rows=10000, n_cols=10000, maybe_filtered=[0, 3, 4, 7, 8]
    )
    assert gp.part_slot(0, 0) == 0
    assert gp.part_slot(1, 1) == 2
    assert gp.part_slot(2, 2) == 4
    assert gp.part_slot(0, 2) is None  # not stored (outside band)
    assert gp.part_slot(1, 0) is None  # lower triangle never stored


def test_block_of():
    gp = GridPartitioner(block_size=4096, n_rows=10000, n_cols=10000, maybe_filtered=None)
    assert gp.block_of(0) == 0
    assert gp.block_of(4095) == 0
    assert gp.block_of(4096) == 1
    assert gp.block_of(9999) == 2
