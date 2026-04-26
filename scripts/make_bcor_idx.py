#!/usr/bin/env python3
"""Generate a `.bcor.idx` sidecar for an existing `.bcor` file.

Usage:
    python scripts/make_bcor_idx.py path/to/file.bcor

Writes path/to/file.bcor.idx alongside the input. Useful for adding sidecars
to legacy .bcor files produced before ldcov gained sidecar support, or by
external tools (e.g., LDstore).

Requires unique rsids in the .bcor (the sidecar format requires rsid->row
to be a function). Aborts with a clear error otherwise.
"""

import argparse
import os
import struct
import sys

import numpy as np

from ldcov.io.bcor_reader import BcorReader
from ldcov.io.bcor_index import BcorIndexWriter


def make_bcor_idx(bcor_path: str, force: bool = False) -> str:
    """Emit a .bcor.idx sidecar next to the given .bcor. Returns the sidecar path."""
    if not os.path.exists(bcor_path):
        raise FileNotFoundError(bcor_path)

    idx_path = bcor_path + ".idx"
    if os.path.exists(idx_path) and not force:
        raise FileExistsError(
            f"{idx_path} already exists. Pass --force to overwrite."
        )

    reader = BcorReader(bcor_path)
    try:
        n = reader.n_snps
        header_size = 7 + 8 + 4 + 4 + 1 + 8  # = 32
        meta_size = reader._corr_block_offset - header_size

        # Walk the meta block to determine per-record byte offsets within the buffer.
        meta_data = bytes(reader._handle.read_range(header_size, meta_size))
        buffer_offsets = [0]
        off = 0
        for _ in range(n):
            L_buffer = struct.unpack_from("<I", meta_data, off)[0]
            off += L_buffer + 4
            buffer_offsets.append(off)
        if off != meta_size:
            raise RuntimeError(
                f"meta block walk mismatch: ended at {off}, expected {meta_size}. "
                "The .bcor file may be corrupt."
            )

        absolute_offsets = np.asarray(
            [o + header_size for o in buffer_offsets], dtype=np.uint64
        )
        variant_info = reader.get_meta()  # has 'rsid' column from BcorReader
    finally:
        reader.close()

    bcor_file_size = os.path.getsize(bcor_path)
    bcor_corr_block_offset = header_size + meta_size  # equivalent to absolute_offsets[-1]

    BcorIndexWriter(idx_path).write(
        variant_info=variant_info,
        meta_record_offsets=absolute_offsets,
        bcor_meta_start=header_size,
        bcor_file_size=bcor_file_size,
        bcor_corr_block_offset=bcor_corr_block_offset,
    )
    return idx_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("bcor", help="Path to input .bcor file")
    parser.add_argument(
        "--force", action="store_true", help="Overwrite an existing .bcor.idx"
    )
    args = parser.parse_args(argv)

    try:
        out = make_bcor_idx(args.bcor, force=args.force)
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
