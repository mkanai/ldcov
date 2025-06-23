"""
I/O utilities for saving and loading correlation matrices.

This module provides functions for saving and loading correlation matrices
in various formats.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
import logging
import os
import gzip

# Import bcor functionality from dedicated modules
from .bcor_reader import BcorReader
from .bcor_writer import BcorWriter, save_bcor

logger = logging.getLogger(__name__)


def save_correlation_matrix(
    corr_matrix: np.ndarray,
    output_file: str,
    variant_info: Optional[pd.DataFrame] = None,
    output_format: str = "matrix",
    n_samples: int = 1000,
    compression: int = 1,
) -> None:
    """
    Save correlation matrix to file.

    Parameters:
    -----------
    corr_matrix : numpy.ndarray
        Correlation matrix (variants × variants)
    output_file : str
        Path to output file
    variant_info : pandas.DataFrame, optional
        Variant information
    output_format : str
        Output format ("matrix", "long", "bcor")
    n_samples : int, optional
        Number of samples (for bcor format metadata)
    compression : int, optional
        Compression level for bcor format (0=1byte, 1=2bytes, 2=4bytes, 3=8bytes)
    """
    # Check output format
    if output_format not in ["matrix", "long", "bcor"]:
        raise ValueError(f"Unsupported output format: {output_format}")

    # Save as numpy compressed file if specified
    if output_file.endswith(".npz"):
        np.savez_compressed(output_file, correlation=corr_matrix, variant_info=variant_info)
        return

    # Save as bcor file if specified
    if output_format == "bcor" or output_file.endswith(".bcor"):
        save_bcor(corr_matrix, output_file, variant_info, n_samples, compression)
        return

    # Determine if output should be compressed
    is_compressed = output_file.endswith((".gz", ".bgz"))
    open_func = gzip.open if is_compressed else open
    mode = "wt" if is_compressed else "w"

    # Extract variant IDs if available (for index file)
    variant_ids = None
    if variant_info is not None and "rsid" in variant_info.columns:
        variant_ids = variant_info["rsid"].tolist()

    # Save matrix in appropriate format
    if output_format == "matrix":
        # Use numpy's savetxt for efficient matrix writing
        if is_compressed:
            # For compressed files, we need to use the file handle
            with open_func(output_file, mode) as f:
                np.savetxt(f, corr_matrix, delimiter="\t", fmt="%.6f")
        else:
            # For uncompressed files, can use filename directly
            np.savetxt(output_file, corr_matrix, delimiter="\t", fmt="%.6f")

        # Optionally, save variant IDs in a separate index file if variant_info is provided
        if variant_ids:
            index_file = f"{os.path.splitext(output_file)[0]}.index.txt"
            with open(index_file, "w") as f:
                for var_id in variant_ids:
                    f.write(f"{var_id}\n")
            logger.info(f"Variant index file written to {index_file}")

    else:  # Long format
        with open_func(output_file, mode) as f:
            # Write header
            if variant_info is not None:
                f.write("#CHROM\tPOS\tID\tREF\tALT\tCHROM_B\tPOS_B\tID_B\tREF_B\tALT_B\tR\n")
            else:
                f.write("#VAR1\tVAR2\tR\n")

            # Write correlation values with buffering
            n_variants = corr_matrix.shape[0]
            buffer = []
            buffer_size = 10000  # Write in chunks of 10K lines

            if variant_info is not None:
                # Pre-extract variant info arrays for faster access
                chroms = variant_info["chrom"].values
                positions = variant_info["pos"].values
                ids = variant_info["rsid"].values
                refs = variant_info["ref"].values
                alts = variant_info["alt"].values

                for i in range(n_variants):
                    for j in range(i + 1, n_variants):  # Upper triangle only
                        r = corr_matrix[i, j]

                        line = (
                            f"{chroms[i]}\t{positions[i]}\t{ids[i]}\t{refs[i]}\t{alts[i]}\t"
                            f"{chroms[j]}\t{positions[j]}\t{ids[j]}\t{refs[j]}\t{alts[j]}\t"
                            f"{r:.6f}\n"
                        )
                        buffer.append(line)

                        if len(buffer) >= buffer_size:
                            f.writelines(buffer)
                            buffer.clear()
            else:
                # Use numpy to get upper triangle indices
                row_indices, col_indices = np.triu_indices(n_variants, k=1)

                for idx in range(len(row_indices)):
                    i, j = row_indices[idx], col_indices[idx]
                    r = corr_matrix[i, j]
                    buffer.append(f"{i}\t{j}\t{r:.6f}\n")

                    if len(buffer) >= buffer_size:
                        f.writelines(buffer)
                        buffer.clear()

            # Write remaining buffer
            if buffer:
                f.writelines(buffer)


def load_correlation_matrix(
    file_path: str,
) -> Tuple[np.ndarray, Optional[pd.DataFrame]]:
    """
    Load correlation matrix from file.

    Parameters:
    -----------
    file_path : str
        Path to correlation matrix file

    Returns:
    --------
    tuple
        (correlation_matrix, variant_info)
    """
    # Check file format based on extension
    if file_path.endswith(".npz"):
        # Load numpy compressed file
        data = np.load(file_path)
        corr_matrix = data["correlation"]
        variant_info = data.get("variant_info", None)
        if variant_info is not None and not isinstance(variant_info, pd.DataFrame):
            variant_info = pd.DataFrame(variant_info)
        return corr_matrix, variant_info

    # Handle bcor format
    if file_path.endswith(".bcor"):
        reader = BcorReader(file_path)
        corr_matrix = reader.read_corr()
        meta = reader.get_meta()

        # Convert metadata to standard format
        variant_info = pd.DataFrame(
            {
                "rsid": meta["rsid"],
                "chrom": meta["chromosome"],
                "pos": meta["position"],
                "ref": meta["allele1"],
                "alt": meta["allele2"],
            }
        )

        return corr_matrix, variant_info

    # Handle txt format (matrix or long)
    is_compressed = file_path.endswith((".gz", ".bgz"))
    open_func = gzip.open if is_compressed else open
    mode = "rt" if is_compressed else "r"

    # Check for index file
    index_file = f"{os.path.splitext(file_path)[0]}.index.txt"
    variant_ids = None
    if os.path.exists(index_file):
        with open(index_file, "r") as f:
            variant_ids = [line.strip() for line in f]

    # Determine format by checking first line
    with open_func(file_path, mode) as f:
        first_line = f.readline().strip()

    if first_line.startswith("#"):
        # Long format
        df = pd.read_csv(file_path, sep="\t", comment="#")

        # Extract variant info
        if all(
            col in df.columns
            for col in [
                "CHROM",
                "POS",
                "ID",
                "REF",
                "ALT",
                "CHROM_B",
                "POS_B",
                "ID_B",
                "REF_B",
                "ALT_B",
            ]
        ):
            variant_cols = ["CHROM", "POS", "ID", "REF", "ALT"]
            variant_a = df[variant_cols].drop_duplicates().reset_index(drop=True)
            variant_b = (
                df.rename(columns={f"{col}_B": col for col in variant_cols})[variant_cols]
                .drop_duplicates()
                .reset_index(drop=True)
            )
            variant_info = (
                pd.concat([variant_a, variant_b]).drop_duplicates().reset_index(drop=True)
            )
            variant_info.columns = ["chrom", "pos", "rsid", "ref", "alt"]
        else:
            variant_info = None

        # Create correlation matrix
        if variant_info is not None:
            n_variants = len(variant_info)
            corr_matrix = np.identity(n_variants)
            for _, row in df.iterrows():
                idx1 = variant_info[
                    (variant_info["chrom"] == row["CHROM"])
                    & (variant_info["pos"] == row["POS"])
                    & (variant_info["rsid"] == row["ID"])
                ].index[0]
                idx2 = variant_info[
                    (variant_info["chrom"] == row["CHROM_B"])
                    & (variant_info["pos"] == row["POS_B"])
                    & (variant_info["rsid"] == row["ID_B"])
                ].index[0]
                corr_matrix[idx1, idx2] = row["R"]
                corr_matrix[idx2, idx1] = row["R"]  # Symmetrical
        else:
            # Simple numeric indices
            max_var = max(df["VAR1"].max(), df["VAR2"].max())
            corr_matrix = np.identity(max_var + 1)
            for _, row in df.iterrows():
                idx1, idx2 = int(row["VAR1"]), int(row["VAR2"])
                corr_matrix[idx1, idx2] = row["R"]
                corr_matrix[idx2, idx1] = row["R"]  # Symmetrical
    else:
        # Plain matrix format (tab-separated values)
        with open_func(file_path, mode) as f:
            lines = f.readlines()

        # Parse the matrix
        corr_values = []
        for line in lines:
            try:
                # Try tab-separated first
                values = [float(val) for val in line.strip().split("\t")]
            except ValueError:
                # Fall back to space-separated
                values = [float(val) for val in line.strip().split()]
            corr_values.append(values)

        corr_matrix = np.array(corr_values)

        # Create minimal variant info if we have variant IDs
        if variant_ids:
            if len(variant_ids) == corr_matrix.shape[0]:
                variant_info = pd.DataFrame(
                    {
                        "rsid": variant_ids,
                        "chrom": [""] * len(variant_ids),  # Placeholder
                        "pos": [0] * len(variant_ids),  # Placeholder
                        "ref": [""] * len(variant_ids),  # Placeholder
                        "alt": [""] * len(variant_ids),  # Placeholder
                    }
                )
            else:
                logger.warning(
                    f"Number of variant IDs ({len(variant_ids)}) does not match matrix dimensions ({corr_matrix.shape[0]})"
                )
                variant_info = None
        else:
            variant_info = None

    return corr_matrix, variant_info
