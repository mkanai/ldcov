"""BCOR sidecar index (.bcor.idx) — rsid → row lookup + meta offsets."""

import struct
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

BCOR_IDX_MAGIC = b"bcoridx1"
BCOR_IDX_VERSION = 1
# 8 magic + 4 version + 4 n_snps + 4 flags + 4 bcor_meta_start
# + 8 bcor_file_size + 8 bcor_corr_block_offset
BCOR_IDX_HEADER_SIZE = 40

_HEADER_STRUCT = struct.Struct("<8sIIIIQQ")


def encode_idx_header(
    n_snps: int,
    bcor_meta_start: int,
    bcor_file_size: int,
    bcor_corr_block_offset: int,
    flags: int = 0,
) -> bytes:
    return _HEADER_STRUCT.pack(
        BCOR_IDX_MAGIC,
        BCOR_IDX_VERSION,
        n_snps,
        flags,
        bcor_meta_start,
        bcor_file_size,
        bcor_corr_block_offset,
    )


def decode_idx_header(buf: bytes) -> Dict[str, int]:
    """Return {n_snps, flags, bcor_meta_start, bcor_file_size, bcor_corr_block_offset}.
    Raises ValueError on bad magic/version/truncation."""
    if len(buf) < BCOR_IDX_HEADER_SIZE:
        raise ValueError("bcor index header truncated")
    (
        magic,
        version,
        n_snps,
        flags,
        bcor_meta_start,
        bcor_file_size,
        bcor_corr_block_offset,
    ) = _HEADER_STRUCT.unpack(buf[:BCOR_IDX_HEADER_SIZE])
    if magic != BCOR_IDX_MAGIC:
        raise ValueError(f"not a valid bcor index file (magic={magic!r})")
    if version != BCOR_IDX_VERSION:
        raise ValueError(
            f"unsupported bcor index version {version}; this library speaks {BCOR_IDX_VERSION}"
        )
    return {
        "n_snps": n_snps,
        "flags": flags,
        "bcor_meta_start": bcor_meta_start,
        "bcor_file_size": bcor_file_size,
        "bcor_corr_block_offset": bcor_corr_block_offset,
    }


class BcorIndexWriter:
    def __init__(self, output_path: str):
        self.output_path = output_path

    def write(
        self,
        variant_info: pd.DataFrame,
        meta_record_offsets: np.ndarray,
        bcor_meta_start: int,
        bcor_file_size: int,
        bcor_corr_block_offset: int,
    ) -> None:
        if "rsid" not in variant_info.columns:
            raise ValueError("variant_info must have an 'rsid' column")

        n = len(variant_info)

        if meta_record_offsets.shape != (n + 1,):
            raise ValueError(
                f"meta_record_offsets must have length n+1 ({n + 1}), got {meta_record_offsets.shape}"
            )
        offsets_u64 = np.ascontiguousarray(meta_record_offsets, dtype=np.uint64)
        # Endpoint invariants must hold for ALL n (including n=0, where [0] == [-1]
        # is a single slot that must simultaneously equal both bounds).
        if int(offsets_u64[0]) != int(bcor_meta_start):
            raise ValueError(
                f"meta_record_offsets[0] must equal bcor_meta_start "
                f"({int(offsets_u64[0])} != {int(bcor_meta_start)})"
            )
        if int(offsets_u64[-1]) != int(bcor_corr_block_offset):
            raise ValueError(
                f"meta_record_offsets[-1] must equal bcor_corr_block_offset "
                f"({int(offsets_u64[-1])} != {int(bcor_corr_block_offset)})"
            )
        if n > 0 and np.any(np.diff(offsets_u64.astype(np.int64)) < 0):
            raise ValueError("meta_record_offsets must be monotonically non-decreasing")

        rsids = variant_info["rsid"].astype(str).to_numpy()

        # Reject duplicate rsids: rsid → row must be a function for partial-read-by-rsid.
        # If duplicates ever need to be supported, the format must be extended.
        unique_rsids, counts = np.unique(rsids, return_counts=True)
        if len(unique_rsids) != n:
            dup_rsids = unique_rsids[counts > 1].tolist()
            raise ValueError(
                f"variant_info contains duplicate rsids: {dup_rsids[:10]}"
                f"{'...' if len(dup_rsids) > 10 else ''}. "
                "The .bcor.idx format requires unique rsids."
            )

        # rsids stay in row order (no sort). Build rsid_offsets in int64 then cast to uint32.
        rsid_bytes = [s.encode("utf-8") for s in rsids]
        lengths = np.fromiter((len(b) for b in rsid_bytes), dtype=np.int64, count=n)
        total_rsid_bytes = int(lengths.sum())
        if total_rsid_bytes > 2**32 - 1:
            raise ValueError("rsid block exceeds 4 GiB; uint32 offset range exhausted")
        cum_int64 = np.empty(n + 1, dtype=np.int64)
        cum_int64[0] = 0
        if n > 0:
            np.cumsum(lengths, out=cum_int64[1:])
        rsid_offsets = cum_int64.astype(np.uint32)

        with open(self.output_path, "wb") as fh:
            fh.write(
                encode_idx_header(
                    n_snps=n,
                    bcor_meta_start=bcor_meta_start,
                    bcor_file_size=bcor_file_size,
                    bcor_corr_block_offset=bcor_corr_block_offset,
                )
            )
            fh.write(rsid_offsets.tobytes())
            fh.write(offsets_u64.tobytes())
            fh.write(b"".join(rsid_bytes))


class BcorIndexReader:
    """Loads a .bcor.idx sidecar. Fully in-memory once constructed.

    Lookup is via an in-memory dict[str, int] built once at load time — O(1) per query,
    O(n) construction. No sort/permutation indirection.
    """

    def __init__(
        self,
        n_snps: int,
        bcor_meta_start: int,
        bcor_file_size: int,
        bcor_corr_block_offset: int,
        rsid_offsets: np.ndarray,  # uint32[n+1]
        meta_record_offsets: np.ndarray,  # uint64[n+1]
        rsids: list,  # length-n list of UTF-8-decoded rsids in row order
    ):
        self._n_snps = int(n_snps)
        self._bcor_meta_start = int(bcor_meta_start)
        self._bcor_file_size = int(bcor_file_size)
        self._bcor_corr_block_offset = int(bcor_corr_block_offset)
        self._rsid_offsets = rsid_offsets
        self._meta_record_offsets = meta_record_offsets
        self._rsids = rsids

        # Build rsid → row dict once. Reject duplicates discovered here too (defense in
        # depth; the writer also rejects them).
        self._rsid_to_row_map: dict = {}
        for row, r in enumerate(rsids):
            if r in self._rsid_to_row_map:
                raise ValueError(
                    f"bcor index contains duplicate rsid {r!r} at rows "
                    f"{self._rsid_to_row_map[r]} and {row}"
                )
            self._rsid_to_row_map[r] = row

    @classmethod
    def from_stream(cls, fh, size: Optional[int] = None) -> "BcorIndexReader":
        fh.seek(0)
        header_buf = fh.read(BCOR_IDX_HEADER_SIZE)
        hdr = decode_idx_header(header_buf)
        n_snps = hdr["n_snps"]
        bcor_meta_start = hdr["bcor_meta_start"]

        rsid_offsets_bytes = 4 * (n_snps + 1)
        meta_offsets_bytes = 8 * (n_snps + 1)

        rsid_offsets_buf = fh.read(rsid_offsets_bytes)
        if len(rsid_offsets_buf) != rsid_offsets_bytes:
            raise ValueError("bcor index truncated: rsid_offsets section short")
        rsid_offsets = np.frombuffer(rsid_offsets_buf, dtype=np.uint32).copy()

        meta_offsets_buf = fh.read(meta_offsets_bytes)
        if len(meta_offsets_buf) != meta_offsets_bytes:
            raise ValueError("bcor index truncated: meta_record_offsets section short")
        meta_record_offsets = np.frombuffer(meta_offsets_buf, dtype=np.uint64).copy()

        rsid_block_len = int(rsid_offsets[-1]) if n_snps > 0 else 0
        rsid_block = fh.read(rsid_block_len)
        if len(rsid_block) != rsid_block_len:
            raise ValueError(
                f"bcor index truncated: expected {rsid_block_len}-byte rsid block, got {len(rsid_block)}"
            )

        cls._validate_sections(
            n_snps=n_snps,
            rsid_offsets=rsid_offsets,
            meta_record_offsets=meta_record_offsets,
            rsid_block_len=rsid_block_len,
            bcor_meta_start=bcor_meta_start,
            bcor_corr_block_offset=hdr["bcor_corr_block_offset"],
        )

        # Decode rsid block eagerly so any UTF-8 errors surface as ValueError at load time.
        rsids = []
        for i in range(n_snps):
            start = int(rsid_offsets[i])
            end = int(rsid_offsets[i + 1])
            try:
                rsids.append(rsid_block[start:end].decode("utf-8"))
            except UnicodeDecodeError as e:
                raise ValueError(
                    f"bcor index rsid_block contains invalid UTF-8 at row {i}: {e}"
                ) from e

        return cls(
            n_snps=n_snps,
            bcor_meta_start=bcor_meta_start,
            bcor_file_size=hdr["bcor_file_size"],
            bcor_corr_block_offset=hdr["bcor_corr_block_offset"],
            rsid_offsets=rsid_offsets,
            meta_record_offsets=meta_record_offsets,
            rsids=rsids,
        )

    @staticmethod
    def _validate_sections(
        n_snps: int,
        rsid_offsets: np.ndarray,
        meta_record_offsets: np.ndarray,
        rsid_block_len: int,
        bcor_meta_start: int,
        bcor_corr_block_offset: int,
    ) -> None:
        if rsid_offsets.shape != (n_snps + 1,):
            raise ValueError("rsid_offsets has wrong length")
        if meta_record_offsets.shape != (n_snps + 1,):
            raise ValueError("meta_record_offsets has wrong length")

        if int(rsid_offsets[0]) != 0:
            raise ValueError("rsid_offsets[0] must be 0")

        # Endpoint invariants on meta_record_offsets must hold for ALL n. For n=0 the
        # array has a single slot that simultaneously is [0] and [-1].
        if int(meta_record_offsets[0]) != int(bcor_meta_start):
            raise ValueError(
                f"meta_record_offsets[0] (={int(meta_record_offsets[0])}) must equal "
                f"bcor_meta_start (={int(bcor_meta_start)})"
            )
        if int(meta_record_offsets[-1]) != int(bcor_corr_block_offset):
            raise ValueError(
                f"meta_record_offsets[-1]={int(meta_record_offsets[-1])} must equal "
                f"bcor_corr_block_offset={int(bcor_corr_block_offset)}"
            )

        if n_snps == 0:
            return

        rsid_diff = np.diff(rsid_offsets.astype(np.int64))
        if np.any(rsid_diff < 0):
            raise ValueError("rsid_offsets must be monotonically non-decreasing")
        if int(rsid_offsets[-1]) != rsid_block_len:
            raise ValueError(
                f"rsid_offsets[-1]={int(rsid_offsets[-1])} does not match "
                f"rsid_block length {rsid_block_len}"
            )

        meta_diff = np.diff(meta_record_offsets.astype(np.int64))
        if np.any(meta_diff < 0):
            raise ValueError("meta_record_offsets must be monotonically non-decreasing")

    @property
    def n_snps(self) -> int:
        return self._n_snps

    @property
    def bcor_meta_start(self) -> int:
        return self._bcor_meta_start

    @property
    def bcor_file_size(self) -> int:
        return self._bcor_file_size

    @property
    def bcor_corr_block_offset(self) -> int:
        return self._bcor_corr_block_offset

    def rsid_to_row(self, rsid: str) -> Optional[int]:
        return self._rsid_to_row_map.get(rsid)

    def rsids_to_rows(self, rsids: Sequence[str]) -> np.ndarray:
        # Per-item dict lookup. Faster than np.searchsorted for typical K (10-10k) since
        # there is no setup or indirection cost. Materialize first so generators / map
        # objects work too, not only Sized sequences.
        rsids_list = list(rsids)
        get = self._rsid_to_row_map.get
        return np.fromiter((get(r, -1) for r in rsids_list), dtype=np.int64, count=len(rsids_list))

    def meta_byte_range(self, row: int) -> Tuple[int, int]:
        """Return (start, length) byte range for variant `row`'s meta record."""
        if row < 0 or row >= self._n_snps:
            raise IndexError(row)
        start = int(self._meta_record_offsets[row])
        length = int(self._meta_record_offsets[row + 1] - self._meta_record_offsets[row])
        return start, length

    def meta_record_range(self, row: int) -> Tuple[int, int]:
        """Return (start, end) absolute byte offsets for variant `row`'s meta record.
        end is exclusive."""
        if row < 0 or row >= self._n_snps:
            raise IndexError(row)
        return (
            int(self._meta_record_offsets[row]),
            int(self._meta_record_offsets[row + 1]),
        )

    @property
    def meta_block_end(self) -> int:
        """Absolute byte offset of the first byte after the meta block in the parent .bcor.
        Equals bcor_corr_block_offset by invariant."""
        return int(self._meta_record_offsets[self._n_snps])
