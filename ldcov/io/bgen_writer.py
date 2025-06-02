"""
BGEN file writer for saving genotypes to BGEN format.

This module provides functions for writing genotype data to BGEN format,
with support for converting between standardized and allelic scales.
"""

import numpy as np
import pandas as pd
from typing import List
import logging
import os

logger = logging.getLogger(__name__)

try:
    from bgen import BgenWriter

    BGEN_WRITER_AVAILABLE = True
except ImportError:
    logger.warning("bgen module not available; install with 'pip install bgen'")
    BGEN_WRITER_AVAILABLE = False


def _dosages_to_probabilities_vectorized(dosages: np.ndarray) -> np.ndarray:
    """
    Convert allelic dosages to probabilities using vectorized operations.

    For diploid biallelic SNPs:
    - If dosage is 0, P(AA)=1, P(AB)=0, P(BB)=0
    - If dosage is 1, P(AA)=0, P(AB)=1, P(BB)=0
    - If dosage is 2, P(AA)=0, P(AB)=0, P(BB)=1
    - For intermediate values (e.g., 0.5), we linearly interpolate

    Parameters:
    -----------
    dosages : numpy.ndarray
        1D array of dosage values for a single variant

    Returns:
    --------
    numpy.ndarray
        2D array of shape (n_samples, 3) with probabilities [P(AA), P(AB), P(BB)]
    """
    n_samples = len(dosages)
    probs = np.zeros((n_samples, 3), dtype=np.float64)

    # Vectorized probability calculations
    # P(AA): 1 - dosage for dosage <= 1, 0 for dosage > 1
    probs[:, 0] = np.where(dosages <= 1, 1 - dosages, 0)

    # P(AB): dosage for dosage <= 1, 2 - dosage for dosage > 1
    probs[:, 1] = np.where(dosages <= 1, dosages, 2 - dosages)

    # P(BB): 0 for dosage <= 1, dosage - 1 for dosage > 1
    probs[:, 2] = np.where(dosages <= 1, 0, dosages - 1)

    # Normalize probabilities to ensure they sum to 1 (handle numerical precision)
    row_sums = probs.sum(axis=1, keepdims=True)
    # Avoid division by zero
    probs = np.divide(probs, row_sums, out=probs, where=row_sums != 0)

    return probs


def write_bgen(
    genotypes: np.ndarray,
    variant_info: pd.DataFrame,
    sample_ids: List[str],
    output_file: str,
    compression: str = "zstd",
    bit_depth: int = 8,
) -> None:
    """
    Write genotype data to BGEN format.

    Parameters:
    -----------
    genotypes : numpy.ndarray
        Genotype matrix in allelic scale (samples x variants)
    variant_info : pandas.DataFrame
        Variant information (must include chrom, pos, id, ref, alt columns)
    sample_ids : list of str
        Sample identifiers
    output_file : str
        Path to output BGEN file
    compression : str
        Compression type: None, 'zstd', or 'zlib' (default='zstd')
    bit_depth : int
        Number of bits for encoding probabilities (1-32, default=8)
    """
    if not BGEN_WRITER_AVAILABLE:
        raise ImportError("bgen module not available. Install with 'pip install bgen'")

    # Prepare output directory
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    n_samples = len(sample_ids)
    n_variants = len(variant_info)

    # Create BgenWriter
    with BgenWriter(
        path=output_file, n_samples=n_samples, samples=sample_ids, compression=compression, layout=2
    ) as writer:

        # Pre-extract variant info for efficient access
        chroms = variant_info["chrom"].values
        positions = variant_info["pos"].values
        varids = variant_info["id"].values
        rsids = variant_info.get("rsid", variant_info["id"]).values
        refs = variant_info["ref"].values
        alts = variant_info["alt"].values

        # Loop through variants and write to BGEN
        for i in range(n_variants):
            # Convert allelic dosages to probabilities - vectorized approach
            dosages = genotypes[:, i]
            probs = _dosages_to_probabilities_vectorized(dosages)

            # Add the variant to the file
            writer.add_variant(
                varid=varids[i],
                rsid=rsids[i],
                chrom=chroms[i],
                pos=positions[i],
                alleles=[refs[i], alts[i]],
                genotypes=probs,
                ploidy=2,
                phased=False,
                bit_depth=bit_depth,
            )

    # Always write sample file - optimized with pandas
    sample_file = f"{os.path.splitext(output_file)[0]}.sample"

    # Create sample DataFrame for efficient writing
    sample_df = pd.DataFrame(
        {"ID_1": sample_ids, "ID_2": sample_ids, "missing": [0] * len(sample_ids)}
    )

    with open(sample_file, "w") as f:
        # Write header
        f.write("ID_1 ID_2 missing\n")
        f.write("0 0 0\n")  # Format line

    # Append sample data efficiently using pandas
    sample_df.to_csv(sample_file, mode="a", sep=" ", header=False, index=False)

    logger.info(f"Sample file written to {sample_file}")

    logger.info(f"BGEN file written to {output_file}")


def save_metadata(variant_info: pd.DataFrame, output_file: str) -> None:
    """
    Save variant metadata to a file.

    Parameters:
    -----------
    variant_info : pandas.DataFrame
        Variant information. Can optionally include 'mean' and 'norm' columns for
        standardization parameters.
    output_file : str
        Path to output metadata file

    Raises:
    -------
    ValueError
        If variant_info is empty
    """
    # Check if variant_info is empty
    if variant_info.empty:
        raise ValueError("Cannot save metadata: variant_info is empty")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a DataFrame with the metadata
    # Include required columns and any additional columns present in variant_info
    metadata = variant_info.copy()

    # Ensure required columns are included
    required_cols = ["chrom", "pos", "id", "ref", "alt"]
    for col in required_cols:
        if col not in metadata.columns:
            raise ValueError(f"Required column '{col}' not found in variant_info")

    # Check if standardization parameters are included
    has_params = "mean" in metadata.columns and "norm" in metadata.columns
    if has_params:
        logger.info("Metadata includes standardization parameters (means and norms)")

    # Save as CSV
    is_compressed = output_file.endswith((".gz", ".bgz"))
    if is_compressed:
        metadata.to_csv(output_file, index=False, compression="gzip")
    else:
        metadata.to_csv(output_file, index=False)

    logger.info(f"Metadata file written to {output_file}")


def correlation_preserving_transform(
    standardized_genotypes: np.ndarray, impute_missing: bool = True
) -> np.ndarray:
    """
    Convert standardized genotypes to allelic scale using a correlation-preserving transformation.

    This transformation ensures that when the output is re-standardized (centered and scaled
    by L2 norm), it will produce the same correlation matrix as the input standardized genotypes.

    Parameters:
    -----------
    standardized_genotypes : numpy.ndarray
        Standardized genotype matrix (samples x variants) with mean 0 and L2 norm 1 per column
    impute_missing : bool
        Whether to impute missing values with 1.0

    Returns:
    --------
    numpy.ndarray
        Genotype matrix in allelic scale (samples x variants) that preserves correlations

    Notes:
    ------
    This implementation uses a per-variant transformation that maps the standardized values
    to the [0, 2] range while preserving the relative distances between samples. This ensures
    that the correlation structure is maintained when the genotypes are re-standardized.

    The transformation for each variant:
    1. Find the min and max of the standardized values
    2. Linearly map the range [min, max] to a subset of [0, 2] that avoids extreme values
    3. This preserves relative positions and hence correlations
    """
    logger.info("Converting to allelic scale with correlation-preserving transformation")

    # Create output array
    allelic_genotypes = np.zeros_like(standardized_genotypes)

    # Process each variant independently
    for j in range(standardized_genotypes.shape[1]):
        col = standardized_genotypes[:, j]

        # Skip if all values are NaN
        if np.all(np.isnan(col)):
            allelic_genotypes[:, j] = 1.0
            continue

        # Get non-NaN values
        non_nan_mask = ~np.isnan(col)
        col_values = col[non_nan_mask]

        if len(col_values) == 0 or np.std(col_values) == 0:
            # If no variation, set to middle value
            allelic_genotypes[:, j] = 1.0
        else:
            # Find range of standardized values
            min_std = np.min(col_values)
            max_std = np.max(col_values)
            range_std = max_std - min_std

            # Map to a range that avoids extreme values to prevent clipping
            # Use [0.1, 1.9] as the target range to avoid boundary effects
            target_min = 0.1
            target_max = 1.9
            target_range = target_max - target_min

            # Linear transformation that preserves relative positions
            if range_std > 0:
                scale = target_range / range_std
                shift = target_min - min_std * scale
                allelic_genotypes[:, j] = col * scale + shift
            else:
                allelic_genotypes[:, j] = 1.0

        # Handle missing values
        if impute_missing and np.any(~non_nan_mask):
            allelic_genotypes[~non_nan_mask, j] = 1.0

    # Final safety check - should not be needed with the conservative range
    allelic_genotypes = np.clip(allelic_genotypes, 0, 2)

    return allelic_genotypes
