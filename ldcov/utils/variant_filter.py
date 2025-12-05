"""
Utilities for reading and validating variant filter files (.z format).

This module provides functions to read .z files that specify which variants
to extract from BGEN files and in what order. Z-files are tab-delimited files
containing variant information with exact chromosome/position/allele matching.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Required columns for z-files
REQUIRED_COLUMNS = ["rsid", "chromosome", "position", "allele1", "allele2"]


def load_variant_filter(z_file_path: str) -> Dict[str, Any]:
    """
    Load and validate a .z file, returning a variant filter dictionary.

    The .z file format is expected to have columns:
    - rsid: variant ID
    - chromosome: chromosome (should be consistent across all variants)
    - position: genomic position (should be sorted)
    - allele1: reference allele
    - allele2: alternative allele

    Parameters:
    -----------
    z_file_path : str
        Path to the .z file

    Returns:
    --------
    Dict[str, Any]
        Filter dictionary with keys:
        - chromosome: str, chromosome name (exact format from z file)
        - positions: list of int, genomic positions to extract
        - rsids: list of str, variant IDs in same order as positions
        - allele1: list of str, first alleles (reference)
        - allele2: list of str, second alleles (alternative)
        - z_file_order: list of int, original indices for order preservation

    Raises:
    -------
    FileNotFoundError
        If the z file doesn't exist
    ValueError
        If file format is invalid, contains multiple chromosomes, or positions are not sorted
    """
    logger.info(f"Reading variant filter file: {z_file_path}")

    # Read and validate the z-file
    z_df = _read_and_validate_z_file(z_file_path)

    # Log summary
    chromosome = z_df["chromosome"].iloc[0]
    positions = z_df["position"].values
    logger.info(f"Loaded {len(z_df)} variants from chromosome {chromosome}")
    logger.info(f"Position range: {positions.min():,} - {positions.max():,}")

    # Create filter dictionary
    return {
        "chromosome": chromosome,
        "positions": z_df["position"].tolist(),
        "rsids": z_df["rsid"].tolist(),
        "allele1": z_df["allele1"].tolist(),
        "allele2": z_df["allele2"].tolist(),
        "z_file_order": list(range(len(z_df))),
    }


def get_filter_summary(variant_filter: Dict[str, Any]) -> str:
    """
    Get a human-readable summary of a variant filter.

    Parameters:
    -----------
    variant_filter : Dict[str, Any]
        Filter dictionary from load_variant_filter()

    Returns:
    --------
    str
        Summary string describing the filter
    """
    n_variants = len(variant_filter["positions"])

    if n_variants == 0:
        return "Empty variant filter"

    chromosome = variant_filter["chromosome"]
    positions = variant_filter["positions"]
    min_pos = min(positions)
    max_pos = max(positions)

    return (
        f"Variant filter: {n_variants} variants on chromosome {chromosome} "
        f"(positions {min_pos:,} - {max_pos:,})"
    )


def _read_and_validate_z_file(z_file_path: str) -> pd.DataFrame:
    """Read and validate a z-file, returning a cleaned DataFrame."""
    z_file = Path(z_file_path)
    if not z_file.exists():
        raise FileNotFoundError(f"Z file not found: {z_file_path}")

    # Read the file
    try:
        z_df = pd.read_csv(z_file, sep=r"\s+", dtype=str)
    except Exception as e:
        raise ValueError(f"Error reading .z file {z_file_path}: {e}")

    # Validate structure
    if len(z_df) == 0:
        raise ValueError("Z file contains no variants")

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in z_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in .z file: {missing_cols}")

    # Validate content
    _validate_z_file_content(z_df)

    return z_df


def _validate_z_file_content(z_df: pd.DataFrame) -> None:
    """Validate the content of a z-file DataFrame."""
    # Check chromosome consistency
    chromosomes = z_df["chromosome"].unique()
    if len(chromosomes) > 1:
        raise ValueError(
            f"Z file contains multiple chromosomes: {chromosomes}. "
            "Only single chromosome files are supported."
        )

    # Convert and validate positions
    try:
        z_df["position"] = z_df["position"].astype(int)
    except ValueError as e:
        raise ValueError(f"Invalid position values in .z file (must be integers): {e}")

    # Check if positions are sorted
    positions = z_df["position"].values
    if not np.all(positions[:-1] <= positions[1:]):
        raise ValueError("Positions in .z file are not sorted. Please sort by position.")

    # Check for and report duplicates
    _check_duplicates(z_df, positions)


def _check_duplicates(z_df: pd.DataFrame, positions: np.ndarray) -> None:
    """Check for and log duplicate variants."""
    # Check for exact duplicate variants (same position AND same alleles)
    exact_duplicates = z_df.duplicated(subset=["position", "allele1", "allele2"], keep=False)
    n_exact_duplicates = exact_duplicates.sum()

    if n_exact_duplicates > 0:
        logger.warning(
            f"Found {n_exact_duplicates} variants with duplicate position+allele combinations"
        )

    # Log info about duplicate positions with different alleles (allowed)
    position_duplicates = z_df.duplicated(subset=["position"], keep=False)
    n_position_duplicates = position_duplicates.sum()

    if n_position_duplicates > 0:
        n_unique_positions = len(np.unique(positions))
        n_duplicate_positions = len(positions) - n_unique_positions
        logger.info(
            f"Found {n_position_duplicates} variants at {n_duplicate_positions} "
            "duplicate positions (allowed if alleles differ)"
        )
