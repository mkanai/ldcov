import struct
import numpy as np
import lz4.block
from ldcov.io.blockmatrix.block_codec import decode_block


def _frame(body: bytes) -> bytes:
    comp = lz4.block.compress(body, store_size=False)
    return struct.pack("<i", len(comp) + 4) + struct.pack("<i", len(body)) + comp


def encode_block(mat: np.ndarray, is_transpose: bool = False) -> bytes:
    """Encode a 2D float64 matrix the way Hail writes a BlockMatrix block."""
    rows, cols = mat.shape
    order = "C" if is_transpose else "F"
    body = (
        struct.pack("<i", rows)
        + struct.pack("<i", cols)
        + struct.pack("<b", 1 if is_transpose else 0)
        + np.ascontiguousarray(mat, dtype="<f8").tobytes(order=order)
    )
    return _frame(body)


def test_decode_block_roundtrip_column_major():
    mat = np.arange(12, dtype=np.float64).reshape(3, 4)
    out = decode_block(encode_block(mat, is_transpose=False))
    assert out.shape == (3, 4)
    np.testing.assert_array_equal(out, mat)


def test_decode_block_roundtrip_row_major():
    mat = np.arange(12, dtype=np.float64).reshape(3, 4)
    out = decode_block(encode_block(mat, is_transpose=True))
    np.testing.assert_array_equal(out, mat)


def test_decode_block_multiframe():
    # Force >1 frame by chunking the body across frames.
    mat = np.arange(20, dtype=np.float64).reshape(4, 5)
    body = (
        struct.pack("<i", 4)
        + struct.pack("<i", 5)
        + struct.pack("<b", 0)
        + np.ascontiguousarray(mat, dtype="<f8").tobytes(order="F")
    )
    raw = _frame(body[:17]) + _frame(body[17:])
    out = decode_block(raw)
    np.testing.assert_array_equal(out, mat)
