"""
BCOR (Binary CORrelation) file reader.

This module provides functionality for reading bcor files, which store
correlation matrices in a binary format compatible with LDstore.

Extended format: Supports reading extended bcor files (magic: "bcor1.1x")
that include diagonal values.
"""

import bisect
import io

import numpy as np
import pandas as pd
import struct
from typing import List, Optional, Tuple
import logging

from .bcor_file_handle import BcorFileHandle

logger = logging.getLogger(__name__)


def _decode_meta_record(chunk) -> Tuple[str, int, str, str, str]:
    """Parse one .bcor meta record. `chunk` MUST cover exactly one record:
    [L_buffer:u32][index:u32][L_rsid:u16][rsid][pos:u32][L_chrom:u16][chrom]
    [L_a1:u32][a1][L_a2:u32][a2]
    Returns (rsid, position, chromosome, allele1, allele2). Strict on length:
    rejects both truncated AND oversized slices."""
    if len(chunk) < 4:
        raise ValueError(f"bcor meta record too short to read L_buffer ({len(chunk)} bytes)")
    L_buffer = struct.unpack_from("<I", chunk, 0)[0]
    expected_len = L_buffer + 4
    if len(chunk) != expected_len:
        raise ValueError(
            f"bcor meta record length mismatch: have {len(chunk)} bytes, expected {expected_len}"
        )
    offset = 8  # skip L_buffer (u32) + index (u32)
    L_rsid = struct.unpack_from("<H", chunk, offset)[0]
    offset += 2
    rsid = bytes(chunk[offset : offset + L_rsid]).decode("utf-8")
    offset += L_rsid
    position = struct.unpack_from("<I", chunk, offset)[0]
    offset += 4
    L_chrom = struct.unpack_from("<H", chunk, offset)[0]
    offset += 2
    chromosome = bytes(chunk[offset : offset + L_chrom]).decode("utf-8")
    offset += L_chrom
    L_a1 = struct.unpack_from("<I", chunk, offset)[0]
    offset += 4
    allele1 = bytes(chunk[offset : offset + L_a1]).decode("utf-8")
    offset += L_a1
    L_a2 = struct.unpack_from("<I", chunk, offset)[0]
    offset += 4
    allele2 = bytes(chunk[offset : offset + L_a2]).decode("utf-8")
    offset += L_a2
    if offset != len(chunk):
        raise ValueError(
            f"bcor meta record has {len(chunk) - offset} trailing bytes after parse; corrupt"
        )
    return rsid, position, chromosome, allele1, allele2


def _merge_ranges(offsets: np.ndarray, value_size: int, gap: int):
    """Given a sorted array of byte offsets (each value `value_size` bytes), merge
    adjacent ones if gap between them is <= `gap`. Returns list of (start, length)."""
    if offsets.size == 0:
        return []
    merged = []
    run_start = int(offsets[0])
    run_end = run_start + value_size  # exclusive
    for off in offsets[1:]:
        off_i = int(off)
        if off_i - run_end <= gap:
            run_end = max(run_end, off_i + value_size)
        else:
            merged.append((run_start, run_end - run_start))
            run_start = off_i
            run_end = off_i + value_size
    merged.append((run_start, run_end - run_start))
    return merged


class BcorReader:
    """
    Reader for bcor (binary correlation) files.

    The bcor format is a binary format for storing correlation matrices,
    originally developed for LDstore. This reader provides a pure Python
    implementation for reading bcor files.

    Extended format: Supports reading extended bcor files (magic: "bcor1.1x")
    that include diagonal values.

    Attributes:
        filename (str): Path to the bcor file
        n_samples (int): Number of samples
        n_snps (int): Number of SNPs/variants
        compression (int): Compression level (0-3)
        is_extended (bool): Whether this is an extended format file

    Example:
        >>> reader = BcorReader('data.bcor')
        >>> matrix = reader.read_corr()  # Read full correlation matrix
        >>> meta = reader.get_meta()     # Get variant metadata
    """

    # Magic numbers
    MAGIC_STANDARD = b"bcor1.1"
    MAGIC_EXTENDED = b"bcor1.x"  # Extended format with diagonal values (7 bytes)

    # Precompiled struct formats for better performance
    _HEADER_FORMATS = {
        "file_size": struct.Struct("<Q"),
        "n_samples": struct.Struct("<I"),
        "n_snps": struct.Struct("<I"),
        "compression": struct.Struct("<B"),
        "corr_offset": struct.Struct("<Q"),
        "meta_buffer": struct.Struct("<I"),
        "meta_index": struct.Struct("<I"),
        "meta_rsid": struct.Struct("<H"),
        "meta_pos": struct.Struct("<I"),
        "meta_chrom": struct.Struct("<H"),
        "meta_allele": struct.Struct("<I"),
    }

    # Precompiled correlation value formats
    _CORR_FORMATS = {
        1: struct.Struct("<B"),
        2: struct.Struct("<H"),
        4: struct.Struct("<I"),
        8: struct.Struct("<Q"),
    }

    # NA values for each byte size
    _NA_VALUES = {1: 208, 2: 53248, 4: 3489660928, 8: 14987979559889010688}

    # Compression to bytes mapping
    _COMPRESSION_BYTES = [2, 4, 8, 1]

    def __init__(self, filename: str, use_mmap: bool = None):
        """Initialize bcor reader.

        Parameters:
        -----------
        filename : str
            Path to the bcor file to read
        use_mmap : bool, optional
            Whether to use memory mapping for large files.
            If None, automatically decided based on file size.
        """
        self._filename = filename
        self._handle = BcorFileHandle(filename, use_mmap=use_mmap)
        self._handle.__enter__()  # eagerly open
        self._use_mmap = self._handle.mmap is not None

        # Header is always read eagerly — it's tiny (32 bytes) and every later step needs it.
        self._read_header()

        # Sidecar (if present) gates whether we eagerly load the meta block.
        self._index = None
        self._meta = None
        self._load_index_if_present()

        # Eager meta load when there is no sidecar, or for any local file (where the I/O is
        # cheap and existing tests/callers expect self._meta to be populated immediately).
        if self._index is None or not self._handle.is_remote():
            self._read_meta()
        # else: defer — get_meta() will lazily fetch using sidecar offsets in _lazy_populate_meta.

        # Cached per-compression-level values
        self._bytes_per_value = self._COMPRESSION_BYTES[self._compression]
        self._corr_format = self._CORR_FORMATS[self._bytes_per_value]
        self._na_value = self._NA_VALUES[self._bytes_per_value]
        self._shift_bits = -1 * (8 * self._bytes_per_value - 2)

        if self._is_extended:
            self._read_diagonal_values()

    def __del__(self):
        """Close file handle on deletion."""
        self.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def close(self):
        """Close file handles and memory map."""
        if hasattr(self, "_handle") and self._handle is not None:
            self._handle.close()
            self._handle = None

    def _read_header(self):
        """Read bcor file header."""
        self._handle.seek(0)
        magic = self._handle.read(7)

        if magic == self.MAGIC_STANDARD:
            self._is_extended = False
        elif magic == self.MAGIC_EXTENDED:
            self._is_extended = True
        else:
            raise ValueError(f"File '{self._filename}' is not a valid bcor file")

        self._magic_size = 7
        logger.debug(f"Detected {'extended' if self._is_extended else 'standard'} bcor format")

        self._file_size = self._HEADER_FORMATS["file_size"].unpack(self._handle.read(8))[0]
        self._n_samples = self._HEADER_FORMATS["n_samples"].unpack(self._handle.read(4))[0]
        self._n_snps = self._HEADER_FORMATS["n_snps"].unpack(self._handle.read(4))[0]
        self._compression = self._HEADER_FORMATS["compression"].unpack(self._handle.read(1))[0]
        self._corr_block_offset = self._HEADER_FORMATS["corr_offset"].unpack(self._handle.read(8))[
            0
        ]

        logger.debug(
            f"BCOR header: n_samples={self._n_samples}, n_snps={self._n_snps}, "
            f"compression={self._compression}, corr_offset={self._corr_block_offset}"
        )

    def _read_meta(self):
        """Read metadata block efficiently. Reuses _decode_meta_record per variant."""
        n = self._n_snps
        if n == 0:
            self._meta = pd.DataFrame(
                columns=["rsid", "position", "chromosome", "allele1", "allele2"]
            )
            return

        header_size = 7 + 8 + 4 + 4 + 1 + 8  # = 32
        meta_size = self._corr_block_offset - header_size
        meta_data = bytes(self._handle.read_range(header_size, meta_size))

        rsid, chromosome, allele1, allele2 = [], [], [], []
        position = np.empty(n, dtype=np.int32)

        if self._index is not None:
            # Use the sidecar's per-variant byte offsets to slice each record exactly.
            for i in range(n):
                abs_start, abs_end = self._index.meta_record_range(i)
                r, p, c, a1, a2 = _decode_meta_record(
                    meta_data[abs_start - header_size : abs_end - header_size]
                )
                rsid.append(r)
                position[i] = p
                chromosome.append(c)
                allele1.append(a1)
                allele2.append(a2)
        else:
            # Walk records sequentially using the L_buffer length from each header.
            offset = 0
            for i in range(n):
                L_buffer = struct.unpack_from("<I", meta_data, offset)[0]
                rec_len = L_buffer + 4
                r, p, c, a1, a2 = _decode_meta_record(meta_data[offset : offset + rec_len])
                rsid.append(r)
                position[i] = p
                chromosome.append(c)
                allele1.append(a1)
                allele2.append(a2)
                offset += rec_len

        self._meta = pd.DataFrame(
            {
                "rsid": rsid,
                "position": position,
                "chromosome": chromosome,
                "allele1": allele1,
                "allele2": allele2,
            }
        )

    def _read_diagonal_values(self):
        """Read diagonal values for extended format."""
        n = self._n_snps
        diag_bytes = n * self._bytes_per_value

        buf = self._handle.read_range(self._corr_block_offset, diag_bytes)
        dtype = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[self._bytes_per_value]
        diag_int = np.frombuffer(buf, dtype=dtype, count=n)

        # Vectorized conversion to float
        mask = diag_int != self._na_value
        self._diagonal_values = np.empty(n, dtype=np.float32)
        self._diagonal_values[mask] = (
            np.ldexp(diag_int[mask].astype(np.float64), self._shift_bits) - 1.0
        )
        self._diagonal_values[~mask] = np.nan

        # Update correlation data offset
        self._corr_data_offset = self._corr_block_offset + n * self._bytes_per_value

    def _load_index_if_present(self):
        from .bcor_index import BcorIndexReader

        idx_path = f"{self._filename}.idx"
        bcor_gcs_fs = self._handle.gcs_fs  # None for local handles
        exists = False
        if self._handle.is_remote():
            if bcor_gcs_fs is not None:
                try:
                    exists = bcor_gcs_fs.exists(idx_path)
                except Exception as e:
                    logger.debug(
                        f"bcor index existence check failed for {idx_path}: {e}; " "assuming absent"
                    )
                    exists = False
        else:
            import os as _os

            exists = _os.path.exists(idx_path)

        if not exists:
            logger.debug(f"No bcor index sidecar at {idx_path}")
            return

        # Reuse the parent's gcsfs.GCSFileSystem to avoid re-authenticating for the sidecar.
        idx_handle = BcorFileHandle(idx_path, gcs_fs=bcor_gcs_fs)
        try:
            idx_handle.__enter__()
            # For GCS, reading the whole sidecar once is cheap (typ. tens of MB) and
            # avoids issuing many small ranged reads during BcorIndexReader.from_stream.
            idx_bytes = bytes(idx_handle.read_range(0, idx_handle.size))
        except Exception as e:  # network errors, missing file, permission denied, etc.
            logger.warning(f"Could not read bcor index sidecar {idx_path}: {e}; falling back.")
            return
        finally:
            idx_handle.close()

        try:
            candidate = BcorIndexReader.from_stream(io.BytesIO(idx_bytes), size=len(idx_bytes))
        except (ValueError, struct.error, UnicodeDecodeError) as e:
            logger.warning(f"Ignoring malformed bcor index sidecar {idx_path}: {e}")
            return

        # Header-level cross-check (Section 2.1). Cheap; no extra ranged reads.
        mismatch_reason = self._sidecar_mismatch_reason(candidate)
        if mismatch_reason is not None:
            logger.warning(
                f"Ignoring bcor index sidecar {idx_path}: {mismatch_reason}. "
                "If you regenerated the .bcor, regenerate the sidecar too "
                "(or pass --no-bcor-idx to suppress)."
            )
            return

        self._index = candidate
        logger.debug(f"Loaded bcor index sidecar from {idx_path}")

    def _sidecar_mismatch_reason(self, candidate) -> Optional[str]:
        """Return None if header-level fields match this .bcor, else a human-readable reason.

        Note on `bcor_file_size`: we deliberately compare the sidecar's recorded value
        against the .bcor's *actual on-disk size* (`self._handle.size`), NOT the value
        embedded in the .bcor's own header. The existing `_read_header` overwrites
        `self._file_size` with the embedded value, so a truncated parent whose embedded
        header still claims the original size would otherwise pass.
        """
        if candidate.n_snps != self._n_snps:
            return f"n_snps mismatch (sidecar={candidate.n_snps}, parent={self._n_snps})"
        if candidate.bcor_meta_start != 32:  # current header size
            return f"bcor_meta_start mismatch (sidecar={candidate.bcor_meta_start}, parent=32)"
        if candidate.bcor_corr_block_offset != self._corr_block_offset:
            return (
                f"bcor_corr_block_offset mismatch "
                f"(sidecar={candidate.bcor_corr_block_offset}, parent={self._corr_block_offset})"
            )
        actual_size = self._handle.size
        if candidate.bcor_file_size != actual_size:
            return (
                f"bcor_file_size mismatch "
                f"(sidecar={candidate.bcor_file_size}, parent on-disk size={actual_size})"
            )
        return None

    @property
    def has_index(self) -> bool:
        return self._index is not None

    @property
    def index(self):
        return self._index

    @property
    def n_samples(self) -> int:
        """Get number of samples."""
        return self._n_samples

    @property
    def n_snps(self) -> int:
        """Get number of SNPs."""
        return self._n_snps

    @property
    def compression(self) -> int:
        """Get compression level."""
        return self._compression

    @property
    def is_extended(self) -> bool:
        """Check if this is an extended format file."""
        return self._is_extended

    def get_n_samples(self) -> int:
        """Get number of samples (deprecated, use n_samples property)."""
        return self._n_samples

    def get_n_snps(self) -> int:
        """Get number of SNPs (deprecated, use n_snps property)."""
        return self._n_snps

    def get_meta(self) -> pd.DataFrame:
        """Get metadata dataframe. Lazily fetches from .bcor if not already loaded."""
        if self._meta is None:
            self._lazy_populate_meta()
        return self._meta.copy()

    def _lazy_populate_meta(self):
        """Fetch the full meta block in ONE ranged read (the meta block is contiguous), then
        slice per-variant records using the sidecar's meta_record_offsets."""
        assert self._index is not None
        n = self._n_snps

        if n == 0:
            self._meta = pd.DataFrame(
                columns=["rsid", "position", "chromosome", "allele1", "allele2"]
            )
            return

        block_start, _ = self._index.meta_record_range(0)
        block_end = self._index.meta_block_end  # = bcor_corr_block_offset
        block = bytes(self._handle.read_range(block_start, block_end - block_start))

        rsid, chromosome, allele1, allele2 = [], [], [], []
        position = np.empty(n, dtype=np.int32)
        for i in range(n):
            abs_start, abs_end = self._index.meta_record_range(i)
            r, p, c, a1, a2 = _decode_meta_record(
                block[abs_start - block_start : abs_end - block_start]
            )
            rsid.append(r)
            position[i] = p
            chromosome.append(c)
            allele1.append(a1)
            allele2.append(a2)

        self._meta = pd.DataFrame(
            {
                "rsid": rsid,
                "position": position,
                "chromosome": chromosome,
                "allele1": allele1,
                "allele2": allele2,
            }
        )

    def _get_triangular_index(self, snp_x: int, snp_y: int) -> int:
        """Get linear index for strictly lower triangular matrix."""
        if snp_x <= snp_y:
            snp_x, snp_y = snp_y, snp_x

        # Simplified calculation
        n = self._n_snps
        return (n * (n - 1) // 2) - ((n - snp_y) * (n - snp_y - 1) // 2) + (snp_x - snp_y - 1)

    def _read_corr_pair(self, snp_x: int, snp_y: int, seek: bool = True) -> float:
        """Read correlation for a pair of SNPs."""
        # Handle diagonal for extended format
        if snp_x == snp_y and self._is_extended:
            return self._diagonal_values[snp_x]

        # Get offset for correlation data
        base_offset = self._corr_data_offset if self._is_extended else self._corr_block_offset
        offset = base_offset + self._get_triangular_index(snp_x, snp_y) * self._bytes_per_value
        buf = self._handle.read_range(offset, self._bytes_per_value)
        int_val = self._corr_format.unpack_from(buf, 0)[0]
        if int_val == self._na_value:
            return np.nan
        return np.ldexp(int_val, self._shift_bits) - 1.0

    def _read_full_matrix(self) -> np.ndarray:
        """Read full correlation matrix with bulk I/O."""
        n = self._n_snps
        corr = np.zeros((n, n), dtype=np.float32)

        # Set diagonal values
        if self._is_extended:
            np.fill_diagonal(corr, self._diagonal_values)
        else:
            np.fill_diagonal(corr, 1.0)

        # Calculate total number of off-diagonal values to read
        n_values = n * (n - 1) // 2
        total_bytes = n_values * self._bytes_per_value

        # Get base offset for correlation data
        base_offset = self._corr_data_offset if self._is_extended else self._corr_block_offset

        buf = self._handle.read_range(base_offset, total_bytes)
        dtype = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[self._bytes_per_value]
        values = np.frombuffer(buf, dtype=dtype, count=n_values)

        # Vectorized conversion
        mask = values != self._na_value
        float_values = np.empty(n_values, dtype=np.float32)
        float_values[mask] = np.ldexp(values[mask].astype(np.float64), self._shift_bits) - 1.0
        float_values[~mask] = np.nan

        # Fill the upper triangle using vectorized indexing
        # The writer extracts values using np.triu_indices in row-major order
        row_indices, col_indices = np.triu_indices(n, k=1)
        corr[row_indices, col_indices] = float_values

        # Make symmetric (but preserve diagonal)
        return corr + corr.T - np.diag(np.diag(corr))

    def read_corr(self, snps1: List[int] = None, snps2: List[int] = None) -> np.ndarray:
        """
        Read correlation matrix or subset.

        Parameters:
        -----------
        snps1 : list of int, optional
            Indices of SNPs for rows. If None or empty, read all.
        snps2 : list of int, optional
            Indices of SNPs for columns. If None or empty, use snps1.

        Returns:
        --------
        np.ndarray
            Correlation matrix or subset

        Examples:
        ---------
        >>> reader = BcorReader('data.bcor')
        >>> # Read full matrix
        >>> full_matrix = reader.read_corr()
        >>> # Read specific rows
        >>> subset = reader.read_corr([0, 1, 2])
        >>> # Read specific pairs
        >>> pairs = reader.read_corr([0, 1], [2, 3])
        """
        if snps1 is None:
            snps1 = []
        if snps2 is None:
            snps2 = []

        if len(snps1) == 0 and len(snps2) == 0:
            # Read full matrix
            return self._read_full_matrix()

        elif len(snps2) == 0:
            # Read specific rows
            corr = np.zeros([self._n_snps, len(snps1)], dtype=np.float32)
            for i, snp_x in enumerate(snps1):
                # Read all row values
                for snp_y in range(self._n_snps):
                    if snp_x == snp_y:
                        # Handle diagonal
                        if self._is_extended:
                            corr[snp_y, i] = self._diagonal_values[snp_x]
                        else:
                            corr[snp_y, i] = 1.0
                    else:
                        corr[snp_y, i] = self._read_corr_pair(snp_x, snp_y, seek=True)
            return corr

        else:
            # Read specific pairs
            corr = np.zeros([len(snps1), len(snps2)], dtype=np.float32)
            for i, snp_x in enumerate(snps1):
                for j, snp_y in enumerate(snps2):
                    if snp_x == snp_y:
                        # Handle diagonal
                        if self._is_extended:
                            corr[i, j] = self._diagonal_values[snp_x]
                        else:
                            corr[i, j] = 1.0
                    else:
                        corr[i, j] = self._read_corr_pair(snp_x, snp_y, seek=True)
            return corr

    def read_corr_by_rsid(
        self,
        rsids,
        rsids2=None,
        missing: str = "raise",
        range_merge_gap: int = 64 * 1024,
    ):
        if self._index is None:
            raise RuntimeError(
                "read_corr_by_rsid requires a .bcor.idx sidecar. Regenerate with index enabled "
                "(default) or supply one."
            )

        if missing not in ("raise", "warn", "skip"):
            raise ValueError(f"missing must be one of 'raise'|'warn'|'skip'; got {missing!r}")

        rsids = list(rsids)
        rsids2 = list(rsids2) if rsids2 is not None else None

        rows_a = self._index.rsids_to_rows(rsids)
        rows_b = self._index.rsids_to_rows(rsids2) if rsids2 is not None else rows_a

        def _filter(rsid_list, rows, label):
            missing_mask = rows < 0
            if np.any(missing_mask):
                missing_rsids = [r for r, m in zip(rsid_list, missing_mask) if m]
                if missing == "raise":
                    raise KeyError(f"{label} rsids not found in bcor: {missing_rsids[:10]}")
                if missing == "warn":
                    logger.warning(
                        f"{label} rsids not found in bcor (dropping): {missing_rsids[:10]}"
                    )
                keep = ~missing_mask
                return [r for r, k in zip(rsid_list, keep) if k], rows[keep]
            return rsid_list, rows

        rsids, rows_a = _filter(rsids, rows_a, "rsids")
        if rsids2 is not None:
            rsids2, rows_b = _filter(rsids2, rows_b, "rsids2")
        else:
            rows_b = rows_a

        if len(rows_a) == 0 or len(rows_b) == 0:
            shape = (len(rsids), len(rsids2) if rsids2 is not None else len(rsids))
            empty = np.zeros(shape, dtype=np.float32)
            return empty, self._subset_meta(rows_a, range_merge_gap=range_merge_gap)

        rows_a_arr = np.asarray(rows_a, dtype=np.int64)
        rows_b_arr = np.asarray(rows_b, dtype=np.int64)
        ia, jb = np.meshgrid(rows_a_arr, rows_b_arr, indexing="ij")
        ia = ia.ravel()
        jb = jb.ravel()

        diag_mask = ia == jb
        off_mask = ~diag_mask

        base = self._corr_data_offset if self._is_extended else self._corr_block_offset
        bpv = self._bytes_per_value

        off_ia = ia[off_mask]
        off_jb = jb[off_mask]
        lo = np.minimum(off_ia, off_jb)
        hi = np.maximum(off_ia, off_jb)
        n = self._n_snps
        tri_idx = lo * n - (lo * (lo + 1)) // 2 + (hi - lo - 1)
        byte_offsets = base + tri_idx.astype(np.int64) * bpv

        unique_offsets = np.unique(byte_offsets)
        merged = _merge_ranges(unique_offsets, bpv, range_merge_gap)

        raw_chunks = self._handle.read_ranges([(off, length) for off, length in merged])

        dtype = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[bpv]
        chunk_starts = []  # parallel to chunk_arrays; sorted ascending
        chunk_arrays = []
        for (chunk_off, chunk_len), chunk in zip(merged, raw_chunks):
            arr = np.frombuffer(chunk, dtype=dtype, count=chunk_len // bpv)
            chunk_starts.append(chunk_off)
            chunk_arrays.append(arr)

        def _value_at(off: int) -> int:
            # bisect into sorted chunk_starts: O(log M) instead of O(M).
            i = bisect.bisect_right(chunk_starts, off) - 1
            if i >= 0:
                chunk_off = chunk_starts[i]
                arr = chunk_arrays[i]
                idx = (off - chunk_off) // bpv
                if 0 <= idx < arr.size:
                    return int(arr[idx])
            raise AssertionError(
                f"byte offset {off} not found in any merged chunk; this is a bug in _merge_ranges"
            )

        result = np.empty((len(rows_a), len(rows_b)), dtype=np.float32)
        flat_result = result.ravel()

        if self._is_extended:
            flat_result[diag_mask] = self._diagonal_values[ia[diag_mask]]
        else:
            flat_result[diag_mask] = 1.0

        # Vectorized decode: build int_vals via per-pair bisect lookup, then dequantize
        # with np.ldexp in one shot. Matches the strategy used by _read_full_matrix.
        int_vals = np.fromiter(
            (_value_at(int(off)) for off in byte_offsets),
            dtype=np.int64,
            count=byte_offsets.shape[0],
        )
        na_mask = int_vals == self._na_value
        decoded = np.empty(int_vals.shape[0], dtype=np.float64)
        decoded[~na_mask] = np.ldexp(int_vals[~na_mask], self._shift_bits) - 1.0
        decoded[na_mask] = np.nan
        flat_result[off_mask] = decoded.astype(np.float32)

        subset_meta = self._subset_meta(rows_a_arr, range_merge_gap=range_merge_gap)
        return result, subset_meta

    def _subset_meta(self, rows, range_merge_gap: int = 64 * 1024) -> pd.DataFrame:
        """Return a metadata DataFrame for the given rows.

        If self._meta is already populated, slice it directly. Otherwise (lazy GCS path),
        fetch only the relevant meta records via range-merged reads.
        """
        rows = np.asarray(rows, dtype=np.int64)
        if self._meta is not None:
            return self._meta.iloc[rows].reset_index(drop=True)
        assert self._index is not None
        if len(rows) == 0:
            return pd.DataFrame(columns=["rsid", "position", "chromosome", "allele1", "allele2"])

        # Per-row (start, end) byte ranges in the parent .bcor.
        ranges_by_row = [self._index.meta_record_range(int(r)) for r in rows]
        starts = np.asarray([rs for rs, _ in ranges_by_row], dtype=np.int64)
        ends = np.asarray([re for _, re in ranges_by_row], dtype=np.int64)

        # Sort by start, merge adjacent runs, fetch, then slice per record.
        order = np.argsort(starts, kind="stable")
        starts_s = starts[order]
        ends_s = ends[order]

        merged = []  # list of (start, end)
        cur_start, cur_end = int(starts_s[0]), int(ends_s[0])
        for s, e in zip(starts_s[1:].tolist(), ends_s[1:].tolist()):
            if s - cur_end <= range_merge_gap:
                cur_end = max(cur_end, e)
            else:
                merged.append((cur_start, cur_end))
                cur_start, cur_end = s, e
        merged.append((cur_start, cur_end))

        chunks = self._handle.read_ranges([(s, e - s) for s, e in merged])
        # Parallel arrays for bisect lookup: chunk_starts is sorted ascending (merge
        # processes starts in sorted order).
        chunk_starts = [s for s, _ in merged]
        chunk_ends = [e for _, e in merged]
        chunk_buffers = [bytes(c) for c in chunks]

        def _slice(rec_start: int, rec_end: int) -> bytes:
            i = bisect.bisect_right(chunk_starts, rec_start) - 1
            if i >= 0 and chunk_starts[i] <= rec_start and rec_end <= chunk_ends[i]:
                buf = chunk_buffers[i]
                base = chunk_starts[i]
                return buf[rec_start - base : rec_end - base]
            raise AssertionError(
                f"meta record range ({rec_start}, {rec_end}) not in any merged chunk; "
                "this is a bug in the merge loop"
            )

        rsid, chromosome, allele1, allele2 = [], [], [], []
        position = np.empty(len(rows), dtype=np.int32)
        for i, (rs, re) in enumerate(zip(starts.tolist(), ends.tolist())):
            r, p, c, a1, a2 = _decode_meta_record(_slice(rs, re))
            rsid.append(r)
            position[i] = p
            chromosome.append(c)
            allele1.append(a1)
            allele2.append(a2)
        return pd.DataFrame(
            {
                "rsid": rsid,
                "position": position,
                "chromosome": chromosome,
                "allele1": allele1,
                "allele2": allele2,
            }
        )
