"""Decode a Hail BlockMatrix part file (LZ4-framed) into a numpy array."""

import struct
import numpy as np
import lz4.block


def _decompress_frames(raw: bytes) -> bytes:
    """Concatenate all LZ4 block frames in a part file into one byte buffer.

    Decompressed chunks are collected and joined once (single allocation) rather than
    grown incrementally, and the compressed input is sliced via a memoryview to avoid
    copying each frame's bytes out of ``raw``.
    """
    mv = memoryview(raw)
    chunks = []
    pos = 0
    n = len(raw)
    while pos + 4 <= n:
        (total_len,) = struct.unpack_from("<i", mv, pos)
        pos += 4
        if total_len <= 0:
            break
        (decomp_len,) = struct.unpack_from("<i", mv, pos)
        pos += 4
        chunks.append(
            lz4.block.decompress(mv[pos : pos + total_len - 4], uncompressed_size=decomp_len)
        )
        pos += total_len - 4
    return b"".join(chunks)


def decode_block(raw: bytes) -> np.ndarray:
    """Decode one BlockMatrix block file's bytes into a 2D float64 array.

    The returned array is a read-only view over the decompressed buffer in the block's
    native memory order (column-major when ``is_transpose`` is False). We deliberately do
    NOT force a C-contiguous copy: for a 4096x4096 float64 block that copy is a ~134 MB
    cache-unfriendly transpose costing ~370 ms — more than the LZ4 decompression itself —
    and downstream ``np.ix_`` sub-block selection in the assembler works on either order
    and only materializes the small selected region.
    """
    buf = _decompress_frames(raw)
    rows, cols = struct.unpack_from("<ii", buf, 0)
    is_transpose = struct.unpack_from("<b", buf, 8)[0] != 0
    data = np.frombuffer(buf, dtype="<f8", count=rows * cols, offset=9)
    order = "C" if is_transpose else "F"
    return data.reshape((rows, cols), order=order)
