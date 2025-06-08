"""
BCOR (Binary CORrelation) file writer.

This module provides functionality for writing correlation matrices
to bcor files in a format compatible with LDstore.

Extended format: When diagonal values are not all 1.0, the writer creates
an extended bcor file with magic string "bcor1.1x" that stores diagonal values.
"""

import numpy as np
import pandas as pd
import struct
from typing import Optional
import logging
import io

logger = logging.getLogger(__name__)


class BcorWriter:
    """
    Writer for bcor (binary correlation) files.

    The bcor format is a binary format for storing correlation matrices,
    originally developed for LDstore. This writer provides a pure Python
    implementation for creating bcor files.

    Extended format: When diagonal values are not all 1.0, creates an extended
    bcor file (magic: "bcor1.1x") that includes diagonal values.

    Attributes:
        output_file (str): Path to the output bcor file
        n_samples (int): Number of samples
        compression (int): Compression level (0-3)

    Example:
        >>> writer = BcorWriter('output.bcor', n_samples=1000, compression=1)
        >>> writer.write(correlation_matrix, variant_info)
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

    def __init__(self, output_file: str, n_samples: int = 1000, compression: int = 1):
        """
        Initialize bcor writer.

        Parameters:
        -----------
        output_file : str
            Path to the output bcor file
        n_samples : int, optional
            Number of samples (for metadata)
        compression : int, optional
            Compression level:
            - 0: 2 bytes per value (uint16)
            - 1: 4 bytes per value (uint32) [default]
            - 2: 8 bytes per value (uint64)
            - 3: 1 byte per value (uint8)
        """
        self.output_file = output_file
        self.n_samples = n_samples
        self.compression = compression

        if compression not in [0, 1, 2, 3]:
            raise ValueError(f"Invalid compression level: {compression}. Must be 0, 1, 2, or 3.")

        # Cache frequently used values
        self._bytes_per_value = self._COMPRESSION_BYTES[compression]
        self._corr_format = self._CORR_FORMATS[self._bytes_per_value]
        self._na_value = self._NA_VALUES[self._bytes_per_value]

        # Precompute shift value for float-to-int conversion
        self._shift_factor = 1 << (8 * self._bytes_per_value - 2)
        self._max_val = (1 << (8 * self._bytes_per_value)) - 1

    def write(self, corr_matrix: np.ndarray, variant_info: Optional[pd.DataFrame] = None):
        """
        Write correlation matrix to bcor file.

        Automatically detects if extended format is needed based on diagonal values.

        Parameters:
        -----------
        corr_matrix : numpy.ndarray
            Correlation matrix (variants × variants). Must be square and symmetric.
        variant_info : pandas.DataFrame, optional
            Variant information with columns: id, chrom, pos, ref, alt
            If not provided, default metadata will be generated.
        """
        n_snps = corr_matrix.shape[0]

        if corr_matrix.shape[0] != corr_matrix.shape[1]:
            raise ValueError("Correlation matrix must be square")

        # Create default variant info if not provided
        if variant_info is None:
            variant_info = pd.DataFrame(
                {
                    "id": [f"variant_{i}" for i in range(n_snps)],
                    "pos": np.arange(1, n_snps + 1, dtype=np.int32),
                    "chrom": ["1"] * n_snps,
                    "ref": ["A"] * n_snps,
                    "alt": ["T"] * n_snps,
                }
            )

        # Check if we need extended format (diagonal values not all 1.0)
        diagonal = np.diag(corr_matrix)
        use_extended = not np.allclose(diagonal, 1.0, rtol=1e-9, atol=1e-9)

        if use_extended:
            logger.info("Detected non-unit diagonal values, using extended bcor format")

        # Prepare metadata efficiently
        meta_buffer = self._prepare_metadata_buffer(variant_info)
        meta_size = len(meta_buffer)

        with open(self.output_file, "wb") as fh:
            # Write header
            corr_block_offset = self._write_header(fh, n_snps, meta_size, use_extended)

            # Write pre-built metadata buffer
            fh.write(meta_buffer)

            # Write correlation data
            self._write_correlations(fh, corr_matrix, use_extended)

            # Update file size in header
            actual_file_size = fh.tell()
            fh.seek(7)  # Position after magic number (always 7 bytes)
            fh.write(self._HEADER_FORMATS["file_size"].pack(actual_file_size))

        format_type = "extended bcor" if use_extended else "bcor"
        logger.info(f"Saved {format_type} format correlation matrix to {self.output_file}")
        logger.info(
            f"Matrix size: {n_snps}x{n_snps}, compression: {self.compression} "
            f"({self._bytes_per_value} bytes per value)"
        )

    def _prepare_metadata_buffer(self, variant_info: pd.DataFrame) -> bytes:
        """Prepare entire metadata buffer efficiently."""
        buffer = io.BytesIO()

        # Pre-encode strings
        encoded_data = []
        for idx, row in variant_info.iterrows():
            rsid = str(row.get("id", f"variant_{idx}")).encode("utf-8")
            chromosome = str(row.get("chrom", "1")).encode("utf-8")
            allele1 = str(row.get("ref", "A")).encode("utf-8")
            allele2 = str(row.get("alt", "T")).encode("utf-8")
            position = int(row.get("pos", 0))

            encoded_data.append(
                {
                    "rsid": rsid,
                    "chromosome": chromosome,
                    "allele1": allele1,
                    "allele2": allele2,
                    "position": position,
                    "index": idx,
                }
            )

        # Write all records
        for data in encoded_data:
            # Calculate buffer length
            L_buffer = (
                20
                + len(data["rsid"])
                + len(data["chromosome"])
                + len(data["allele1"])
                + len(data["allele2"])
            )

            # Pack all fixed-size fields at once
            buffer.write(struct.pack("<II", L_buffer, data["index"]))

            # Write RSID
            buffer.write(struct.pack("<H", len(data["rsid"])))
            buffer.write(data["rsid"])

            # Write position
            buffer.write(struct.pack("<I", data["position"]))

            # Write chromosome
            buffer.write(struct.pack("<H", len(data["chromosome"])))
            buffer.write(data["chromosome"])

            # Write alleles
            buffer.write(struct.pack("<I", len(data["allele1"])))
            buffer.write(data["allele1"])
            buffer.write(struct.pack("<I", len(data["allele2"])))
            buffer.write(data["allele2"])

        return buffer.getvalue()

    def _write_header(self, fh, n_snps: int, meta_size: int, use_extended: bool) -> int:
        """Write bcor file header and return correlation block offset."""
        # Magic number (always 7 bytes)
        if use_extended:
            fh.write(self.MAGIC_EXTENDED)
        else:
            fh.write(self.MAGIC_STANDARD)
        magic_size = 7

        # Calculate file size (approximation - will be updated later)
        corr_block_size = (n_snps * (n_snps - 1) // 2) * self._bytes_per_value
        if use_extended:
            # Add space for diagonal values
            corr_block_size += n_snps * self._bytes_per_value

        file_size = magic_size + 8 + 4 + 4 + 1 + 8 + meta_size + corr_block_size

        # Pack all header fields at once
        header_data = struct.pack(
            "<QIIBQ",
            file_size,  # File size (uint64)
            self.n_samples,  # Number of samples (uint32)
            n_snps,  # Number of SNPs (uint32)
            self.compression,  # Compression level (uint8)
            magic_size + 8 + 4 + 4 + 1 + 8 + meta_size,  # Correlation block offset (uint64)
        )
        fh.write(header_data)

        return magic_size + 8 + 4 + 4 + 1 + 8 + meta_size

    def _write_correlations(self, fh, corr_matrix: np.ndarray, use_extended: bool):
        """Write correlation data in lower triangular format."""
        n_snps = corr_matrix.shape[0]

        if use_extended:
            # Extended format: first write diagonal values
            diagonal = np.diag(corr_matrix)

            # Vectorized conversion of diagonal values
            nan_mask = np.isnan(diagonal)
            diag_float = np.empty_like(diagonal)
            diag_float[~nan_mask] = (diagonal[~nan_mask] + 1.0) * self._shift_factor
            diag_float[nan_mask] = self._na_value

            # Convert to appropriate integer type and clamp
            if self._bytes_per_value == 1:
                diag_values = np.clip(diag_float, 0, min(self._max_val, self._na_value - 1)).astype(
                    np.uint8
                )
            elif self._bytes_per_value == 2:
                diag_values = np.clip(diag_float, 0, min(self._max_val, self._na_value - 1)).astype(
                    np.uint16
                )
            elif self._bytes_per_value == 4:
                diag_values = np.clip(diag_float, 0, min(self._max_val, self._na_value - 1)).astype(
                    np.uint32
                )
            else:  # 8 bytes
                diag_values = np.clip(diag_float, 0, min(self._max_val, self._na_value - 1)).astype(
                    np.uint64
                )

            # Set NaN values to the NA marker
            diag_values[nan_mask] = self._na_value

            # Write diagonal values
            fh.write(diag_values.tobytes())

        # Write lower triangular values (same for both formats)
        n_values = n_snps * (n_snps - 1) // 2

        # Extract off-diagonal values using upper triangle indices
        # Since correlation matrices are symmetric, we can extract from upper triangle
        # Get upper triangle indices (excluding diagonal)
        row_indices, col_indices = np.triu_indices(n_snps, k=1)
        off_diagonal_values = corr_matrix[row_indices, col_indices]

        # Vectorized conversion from float to int
        # Handle NaN values
        nan_mask = np.isnan(off_diagonal_values)

        # Convert non-NaN values
        int_values_float = np.empty_like(off_diagonal_values)
        int_values_float[~nan_mask] = (off_diagonal_values[~nan_mask] + 1.0) * self._shift_factor
        int_values_float[nan_mask] = self._na_value

        # Convert to appropriate integer type and clamp
        if self._bytes_per_value == 1:
            int_values = np.clip(
                int_values_float, 0, min(self._max_val, self._na_value - 1)
            ).astype(np.uint8)
        elif self._bytes_per_value == 2:
            int_values = np.clip(
                int_values_float, 0, min(self._max_val, self._na_value - 1)
            ).astype(np.uint16)
        elif self._bytes_per_value == 4:
            int_values = np.clip(
                int_values_float, 0, min(self._max_val, self._na_value - 1)
            ).astype(np.uint32)
        else:  # 8 bytes
            int_values = np.clip(
                int_values_float, 0, min(self._max_val, self._na_value - 1)
            ).astype(np.uint64)

        # Set NaN values to the NA marker
        int_values[nan_mask] = self._na_value

        # Write all values at once
        fh.write(int_values.tobytes())


def save_bcor(
    corr_matrix: np.ndarray,
    output_file: str,
    variant_info: Optional[pd.DataFrame] = None,
    n_samples: int = 1000,
    compression: int = 1,
) -> None:
    """
    Save correlation matrix in bcor format.

    Automatically uses extended format if diagonal values are not all 1.0.

    This is a convenience function that creates a BcorWriter and writes the data.

    Parameters:
    -----------
    corr_matrix : numpy.ndarray
        Correlation matrix (variants × variants)
    output_file : str
        Path to output bcor file
    variant_info : pandas.DataFrame, optional
        Variant information with columns: id, chrom, pos, ref, alt
    n_samples : int, optional
        Number of samples (for metadata)
    compression : int, optional
        Compression level (0=2bytes, 1=4bytes, 2=8bytes, 3=1byte)

    Example:
        >>> save_bcor(corr_matrix, 'output.bcor', variant_info, compression=1)
    """
    writer = BcorWriter(output_file, n_samples=n_samples, compression=compression)
    writer.write(corr_matrix, variant_info)
