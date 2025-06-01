"""
Utilities for reading and validating variant filter files (.z format).

This module provides functions to read .z files that specify which variants
to extract from BGEN files and in what order.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def read_z_file(z_file_path: str) -> pd.DataFrame:
    """
    Read and validate a .z file containing variant information.

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
    pandas.DataFrame
        DataFrame with variant information, validated and sorted by position

    Raises:
    -------
    ValueError
        If file format is invalid, contains multiple chromosomes, or positions are not sorted
    """
    logger.info(f"Reading variant filter file: {z_file_path}")

    if not Path(z_file_path).exists():
        raise FileNotFoundError(f"Z file not found: {z_file_path}")

    # Read the .z file
    try:
        z_df = pd.read_csv(z_file_path, sep=r"\s+", dtype=str)
    except Exception as e:
        raise ValueError(f"Error reading .z file {z_file_path}: {e}")

    # Validate required columns
    required_cols = ["rsid", "chromosome", "position", "allele1", "allele2"]
    missing_cols = [col for col in required_cols if col not in z_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in .z file: {missing_cols}")

    # Check that we have at least one variant
    if len(z_df) == 0:
        raise ValueError("Z file contains no variants")

    # Validate chromosome consistency
    chromosomes = z_df["chromosome"].unique()
    if len(chromosomes) > 1:
        raise ValueError(
            f"Z file contains multiple chromosomes: {chromosomes}. Only single chromosome files are supported."
        )

    # Convert position to integer for sorting validation
    try:
        z_df["position"] = z_df["position"].astype(int)
    except ValueError as e:
        raise ValueError(f"Invalid position values in .z file (must be integers): {e}")

    # Check if positions are sorted
    positions = z_df["position"].values
    if not np.all(positions[:-1] <= positions[1:]):
        raise ValueError("Positions in .z file are not sorted. Please sort by position.")

    # Check for duplicate variants (same position AND same alleles)
    duplicate_variants = z_df[
        z_df.duplicated(subset=["position", "allele1", "allele2"], keep=False)
    ]
    if len(duplicate_variants) > 0:
        logger.warning(
            f"Found {len(duplicate_variants)} variants with duplicate position+allele combinations in .z file"
        )

    # Log info about duplicate positions with different alleles (which is allowed)
    duplicate_positions = z_df[z_df.duplicated(subset=["position"], keep=False)]
    if len(duplicate_positions) > 0:
        unique_pos_count = len(np.unique(positions))
        logger.info(
            f"Found {len(duplicate_positions)} variants at {len(positions) - unique_pos_count} duplicate positions (allowed if alleles differ)"
        )

    # Normalize chromosome format (remove leading zeros, add 'chr' prefix if needed)
    chromosome = _normalize_chromosome(chromosomes[0])
    z_df["chromosome_normalized"] = chromosome

    logger.info(f"Loaded {len(z_df)} variants from chromosome {chromosome}")
    logger.info(f"Position range: {positions.min()} - {positions.max()}")

    return z_df


def _normalize_chromosome(chrom: str) -> str:
    """
    Normalize chromosome name to standard format.

    Parameters:
    -----------
    chrom : str
        Chromosome name (e.g., "01", "1", "chr1")

    Returns:
    --------
    str
        Normalized chromosome name (e.g., "1")
    """
    # Remove 'chr' prefix if present
    if chrom.startswith("chr"):
        chrom = chrom[3:]

    # Remove leading zeros
    try:
        chrom_int = int(chrom)
        return str(chrom_int)
    except ValueError:
        # For non-numeric chromosomes (X, Y, MT), return as-is
        return chrom.upper()


def create_variant_filter_from_z(z_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create a variant filter dictionary from a .z file DataFrame.

    Parameters:
    -----------
    z_df : pandas.DataFrame
        DataFrame from read_z_file()

    Returns:
    --------
    dict
        Filter dictionary with keys:
        - chromosome: str, normalized chromosome name
        - positions: list, list of positions to extract
        - rsids: list, list of rsids in the same order as positions
        - expected_order: list, indices for sorting variants to match .z file order
    """
    return {
        "chromosome": z_df["chromosome_normalized"].iloc[0],
        "positions": z_df["position"].tolist(),
        "rsids": z_df["rsid"].tolist(),
        "allele1": z_df["allele1"].tolist(),
        "allele2": z_df["allele2"].tolist(),
        "z_file_order": list(range(len(z_df))),  # Original order from .z file
    }


def validate_variants_match_z_file(
    variant_info: pd.DataFrame, z_filter: Dict[str, Any]
) -> Tuple[List[int], List[int]]:
    """
    Validate that loaded variants match the .z file and return mapping indices.

    Parameters:
    -----------
    variant_info : pandas.DataFrame
        Variant information from BGEN file
    z_filter : dict
        Filter dictionary from create_variant_filter_from_z()

    Returns:
    --------
    tuple
        (bgen_indices, z_indices) - mapping between BGEN variants and .z file order

    Raises:
    -------
    ValueError
        If variants don't match between BGEN and .z file
    """
    # Find variants in BGEN that match .z file
    bgen_indices = []
    z_indices = []

    # Create lookup dictionaries for efficient matching by position+alleles
    z_variant_lookup = {}
    for idx, (pos, a1, a2, rsid) in enumerate(
        zip(z_filter["positions"], z_filter["allele1"], z_filter["allele2"], z_filter["rsids"])
    ):
        # Create key with position and sorted alleles (to handle ref/alt order differences)
        alleles_sorted = tuple(sorted([a1, a2]))
        key = (pos, alleles_sorted)
        if key not in z_variant_lookup:
            z_variant_lookup[key] = []
        z_variant_lookup[key].append(idx)

    # Also create rsid lookup as fallback
    z_rsid_to_idx = {}
    for idx, rsid in enumerate(z_filter["rsids"]):
        if rsid not in z_rsid_to_idx:
            z_rsid_to_idx[rsid] = []
        z_rsid_to_idx[rsid].append(idx)

    matched_z_indices = set()

    for bgen_idx, row in variant_info.iterrows():
        # Try to match by position + alleles first (most reliable)
        bgen_alleles_sorted = tuple(sorted([row["ref"], row["alt"]]))
        key = (row["pos"], bgen_alleles_sorted)

        if key in z_variant_lookup:
            # Find the first unmatched z_index for this variant
            for z_idx in z_variant_lookup[key]:
                if z_idx not in matched_z_indices:
                    bgen_indices.append(bgen_idx)
                    z_indices.append(z_idx)
                    matched_z_indices.add(z_idx)
                    break
        # Fallback: match by rsid
        elif row["id"] in z_rsid_to_idx:
            for z_idx in z_rsid_to_idx[row["id"]]:
                if z_idx not in matched_z_indices:
                    bgen_indices.append(bgen_idx)
                    z_indices.append(z_idx)
                    matched_z_indices.add(z_idx)
                    break

    # Check that we found variants from .z file
    total_z_variants = len(z_filter["positions"])
    if len(matched_z_indices) < total_z_variants:
        missing_count = total_z_variants - len(matched_z_indices)
        logger.warning(f"Could not find {missing_count} variants from .z file in BGEN")

    if len(bgen_indices) == 0:
        raise ValueError("No variants from .z file found in BGEN file")

    logger.info(f"Matched {len(bgen_indices)} out of {total_z_variants} variants from .z file")

    return bgen_indices, z_indices
