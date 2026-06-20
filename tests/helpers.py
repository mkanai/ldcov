"""Shared test helpers for the ldcov test suite.

A regular importable module (unlike conftest.py, which pytest reserves for
fixtures/hooks) so test modules can `from tests.helpers import ...` without
reaching into conftest or each other.
"""

import json
import os
import struct

import lz4.block
import numpy as np

from ldcov.io.blockmatrix.reader import HailBlockMatrixReader


def _bm_frame(body: bytes) -> bytes:
    comp = lz4.block.compress(body, store_size=False)
    return struct.pack("<i", len(comp) + 4) + struct.pack("<i", len(body)) + comp


def _bm_encode_block(mat: np.ndarray) -> bytes:
    rows, cols = mat.shape
    body = (
        struct.pack("<i", rows)
        + struct.pack("<i", cols)
        + struct.pack("<b", 0)
        + np.ascontiguousarray(mat, dtype="<f8").tobytes(order="F")
    )
    return _bm_frame(body)


def write_synthetic_bm(path, blocks, n_rows, n_cols, block_size, dense=False):
    """Write a Hail-format BlockMatrix directory.

    blocks: dict mapping (block_i, block_j) -> 2D float64 ndarray.
    dense: if True, write maybe_filtered=None (every listed block treated as full grid).
    """
    n_block_rows = (n_rows + block_size - 1) // block_size
    items = sorted(blocks.keys(), key=lambda ij: ij[0] + ij[1] * n_block_rows)
    os.makedirs(os.path.join(path, "parts"), exist_ok=True)
    part_files = []
    for slot, ij in enumerate(items):
        pf = "part-%05d" % slot
        part_files.append(pf)
        with open(os.path.join(path, "parts", pf), "wb") as fh:
            fh.write(_bm_encode_block(blocks[ij]))
    meta = {
        "blockSize": block_size,
        "nRows": n_rows,
        "nCols": n_cols,
        "maybeFiltered": None if dense else [i + j * n_block_rows for (i, j) in items],
        "partFiles": part_files,
    }
    with open(os.path.join(path, "metadata.json"), "w") as fh:
        json.dump(meta, fh)
    return path


def make_symmetric_bm(tmp_path):
    """Write a 6x6 symmetric BM (block_size=3 -> 2x2 blocks, upper-tri stored).

    Returns (HailBlockMatrixReader, full 6x6 ndarray).
    """
    rng = np.random.default_rng(0)
    full = rng.standard_normal((6, 6))
    full = (full + full.T) / 2.0
    np.fill_diagonal(full, 1.0)
    bs = 3
    blocks = {
        (0, 0): np.triu(full[0:3, 0:3]).copy(),
        (0, 1): full[0:3, 3:6].copy(),
        (1, 1): np.triu(full[3:6, 3:6]).copy(),
    }
    write_synthetic_bm(str(tmp_path / "m.bm"), blocks, 6, 6, bs)
    return HailBlockMatrixReader(str(tmp_path / "m.bm")), full
