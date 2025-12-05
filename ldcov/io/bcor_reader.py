"""
BCOR (Binary CORrelation) file reader.

This module provides functionality for reading bcor files, which store
correlation matrices in a binary format compatible with LDstore.

Extended format: Supports reading extended bcor files (magic: "bcor1.1x")
that include diagonal values.
"""

import numpy as np
import pandas as pd
import struct
from typing import List, Optional
import logging
import os
import mmap

logger = logging.getLogger(__name__)


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
        """
        Initialize bcor reader.

        Parameters:
        -----------
        filename : str
            Path to the bcor file to read
        use_mmap : bool, optional
            Whether to use memory mapping for large files.
            If None, automatically decided based on file size.
        """
        self._filename = filename
        self._file_size = os.path.getsize(filename)

        # Decide whether to use mmap based on file size (>100MB)
        if use_mmap is None:
            use_mmap = self._file_size > 100 * 1024 * 1024

        self._use_mmap = use_mmap
        self._fh = open(filename, "rb")

        if self._use_mmap:
            self._mmap = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
            self._data = self._mmap
        else:
            self._data = self._fh

        self._read_header()
        self._read_meta()

        # Cache frequently used values
        self._bytes_per_value = self._COMPRESSION_BYTES[self._compression]
        self._corr_format = self._CORR_FORMATS[self._bytes_per_value]
        self._na_value = self._NA_VALUES[self._bytes_per_value]

        # Precompute shift value for int-to-float conversion
        self._shift_bits = -1 * (8 * self._bytes_per_value - 2)

        # Load diagonal values if extended format
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
        if hasattr(self, "_mmap") and self._mmap:
            self._mmap.close()
        if hasattr(self, "_fh") and self._fh:
            self._fh.close()

    def _read_header(self):
        """Read bcor file header."""
        # Read magic number (always 7 bytes)
        start_pos = self._data.tell() if hasattr(self._data, "tell") else 0
        magic = self._data.read(7)

        if magic == self.MAGIC_STANDARD:
            self._is_extended = False
        elif magic == self.MAGIC_EXTENDED:
            self._is_extended = True
        else:
            raise ValueError(f"File '{self._filename}' is not a valid bcor file")

        self._magic_size = 7

        logger.debug(f"Detected {'extended' if self._is_extended else 'standard'} bcor format")

        # Read header fields using precompiled structs
        self._file_size = self._HEADER_FORMATS["file_size"].unpack(self._data.read(8))[0]
        self._n_samples = self._HEADER_FORMATS["n_samples"].unpack(self._data.read(4))[0]
        self._n_snps = self._HEADER_FORMATS["n_snps"].unpack(self._data.read(4))[0]
        self._compression = self._HEADER_FORMATS["compression"].unpack(self._data.read(1))[0]
        self._corr_block_offset = self._HEADER_FORMATS["corr_offset"].unpack(self._data.read(8))[0]

        logger.debug(
            f"BCOR header: n_samples={self._n_samples}, n_snps={self._n_snps}, "
            f"compression={self._compression}, corr_offset={self._corr_block_offset}"
        )

    def _read_meta(self):
        """Read metadata block efficiently."""
        # Pre-allocate arrays
        rsid = []
        position = np.empty(self._n_snps, dtype=np.int32)
        chromosome = []
        allele1 = []
        allele2 = []

        # Calculate metadata size and read all at once
        if self._use_mmap:
            # For mmap, calculate position after header
            # Header size = magic(7) + file_size(8) + n_samples(4) + n_snps(4) + compression(1) + corr_offset(8) = 32
            header_size = 7 + 8 + 4 + 4 + 1 + 8
            meta_size = self._corr_block_offset - header_size
            meta_data = self._mmap[header_size : self._corr_block_offset]
        else:
            # For file handle, read all metadata at once
            meta_size = self._corr_block_offset - self._data.tell()
            meta_data = self._data.read(meta_size)

        # Process metadata in memory
        offset = 0
        for snp in range(self._n_snps):
            # Read buffer length
            L_buffer = struct.unpack_from("<I", meta_data, offset)[0]
            offset += 4

            # Read index
            index = struct.unpack_from("<I", meta_data, offset)[0]
            offset += 4

            # Read RSID
            L_rsid = struct.unpack_from("<H", meta_data, offset)[0]
            offset += 2
            rsid.append(meta_data[offset : offset + L_rsid].decode("utf-8"))
            offset += L_rsid

            # Read position
            position[snp] = struct.unpack_from("<I", meta_data, offset)[0]
            offset += 4

            # Read chromosome
            L_chromosome = struct.unpack_from("<H", meta_data, offset)[0]
            offset += 2
            chromosome.append(meta_data[offset : offset + L_chromosome].decode("utf-8"))
            offset += L_chromosome

            # Read alleles
            L_allele1 = struct.unpack_from("<I", meta_data, offset)[0]
            offset += 4
            allele1.append(meta_data[offset : offset + L_allele1].decode("utf-8"))
            offset += L_allele1

            L_allele2 = struct.unpack_from("<I", meta_data, offset)[0]
            offset += 4
            allele2.append(meta_data[offset : offset + L_allele2].decode("utf-8"))
            offset += L_allele2

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

        if self._use_mmap:
            # Read diagonal values from mmap
            offset = self._corr_block_offset
            if self._bytes_per_value == 1:
                diag_int = np.frombuffer(self._mmap, dtype=np.uint8, count=n, offset=offset)
            elif self._bytes_per_value == 2:
                diag_int = np.frombuffer(self._mmap, dtype=np.uint16, count=n, offset=offset)
            elif self._bytes_per_value == 4:
                diag_int = np.frombuffer(self._mmap, dtype=np.uint32, count=n, offset=offset)
            else:  # 8 bytes
                diag_int = np.frombuffer(self._mmap, dtype=np.uint64, count=n, offset=offset)
        else:
            # Seek to diagonal values
            self._data.seek(self._corr_block_offset)

            # Read diagonal data
            diag_bytes = n * self._bytes_per_value
            diag_data = self._data.read(diag_bytes)

            # Unpack diagonal values
            if self._bytes_per_value == 1:
                diag_int = np.frombuffer(diag_data, dtype=np.uint8)
            elif self._bytes_per_value == 2:
                diag_int = np.frombuffer(diag_data, dtype=np.uint16)
            elif self._bytes_per_value == 4:
                diag_int = np.frombuffer(diag_data, dtype=np.uint32)
            else:  # 8 bytes
                diag_int = np.frombuffer(diag_data, dtype=np.uint64)

        # Vectorized conversion to float
        mask = diag_int != self._na_value
        self._diagonal_values = np.empty(n, dtype=np.float32)
        self._diagonal_values[mask] = (
            np.ldexp(diag_int[mask].astype(np.float64), self._shift_bits) - 1.0
        )
        self._diagonal_values[~mask] = np.nan

        # Update correlation data offset
        self._corr_data_offset = self._corr_block_offset + n * self._bytes_per_value

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
        """
        Get metadata dataframe.

        Returns:
        --------
        pd.DataFrame
            DataFrame with columns: rsid, position, chromosome, allele1, allele2
        """
        return self._meta.copy()

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

        if seek:
            offset = base_offset + self._get_triangular_index(snp_x, snp_y) * self._bytes_per_value
            if self._use_mmap:
                # Direct access with mmap
                int_val = self._corr_format.unpack_from(self._mmap, offset)[0]
                if int_val == self._na_value:
                    return np.nan
                return np.ldexp(int_val, self._shift_bits) - 1.0
            else:
                self._data.seek(offset)

        # Read the value
        int_val = self._corr_format.unpack(self._data.read(self._bytes_per_value))[0]
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

        if self._use_mmap:
            # Read all values at once from mmap
            if self._bytes_per_value == 1:
                values = np.frombuffer(
                    self._mmap, dtype=np.uint8, count=n_values, offset=base_offset
                )
            elif self._bytes_per_value == 2:
                values = np.frombuffer(
                    self._mmap, dtype=np.uint16, count=n_values, offset=base_offset
                )
            elif self._bytes_per_value == 4:
                values = np.frombuffer(
                    self._mmap, dtype=np.uint32, count=n_values, offset=base_offset
                )
            else:  # 8 bytes
                values = np.frombuffer(
                    self._mmap, dtype=np.uint64, count=n_values, offset=base_offset
                )
        else:
            # Seek to correlation block
            self._data.seek(base_offset)

            # Read all correlation data at once
            corr_data = self._data.read(total_bytes)

            # Unpack all values at once
            if self._bytes_per_value == 1:
                values = np.frombuffer(corr_data, dtype=np.uint8)
            elif self._bytes_per_value == 2:
                values = np.frombuffer(corr_data, dtype=np.uint16)
            elif self._bytes_per_value == 4:
                values = np.frombuffer(corr_data, dtype=np.uint32)
            else:  # 8 bytes
                values = np.frombuffer(corr_data, dtype=np.uint64)

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
