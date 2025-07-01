"""
NaN handling strategies for BGEN genotype data.

This module provides different strategies for handling NaN values in genotype matrices:
- error: Raise an error with detailed information about NaN locations
- mean: Impute NaN values with variant-wise mean
- omit: Remove samples containing any NaN values
- warn: Issue a warning but preserve NaN values in the data
"""

import numpy as np
import pandas as pd
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


def handle_nan_values(
    dosages: np.ndarray, variant_info: pd.DataFrame, sample_ids: List[str], action: str = "error"
) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    """
    Handle NaN values in genotype matrix based on specified action.

    Parameters
    ----------
    dosages : np.ndarray
        Genotype dosage matrix (samples x variants)
    variant_info : pd.DataFrame
        Variant information
    sample_ids : List[str]
        Sample IDs
    action : str
        Action to take: "error", "mean", "omit", or "warn"

    Returns
    -------
    Tuple[np.ndarray, pd.DataFrame, List[str]]
        (processed_dosages, variant_info, sample_ids)
    """
    if not np.any(np.isnan(dosages)):
        return dosages, variant_info, sample_ids

    if action == "error":
        _report_nan_error(dosages, variant_info, sample_ids)
    elif action == "mean":
        return _impute_nan_with_mean(dosages, variant_info, sample_ids)
    elif action == "omit":
        return _omit_nan_samples(dosages, variant_info, sample_ids)
    elif action == "warn":
        return _warn_about_nan(dosages, variant_info, sample_ids)
    else:
        raise ValueError(f"Unknown nan_action: {action}")


def _impute_nan_with_mean(
    dosages: np.ndarray, variant_info: pd.DataFrame, sample_ids: List[str]
) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    """
    Impute NaN values with variant-wise mean.

    Parameters:
    -----------
    dosages : np.ndarray
        Genotype dosage matrix (samples x variants)
    variant_info : pd.DataFrame
        Variant information
    sample_ids : List[str]
        Sample IDs

    Returns:
    --------
    tuple
        (imputed_dosages, variant_info, sample_ids)
    """
    # Count NaN values for logging
    nan_mask = np.isnan(dosages)
    n_nan_total = np.sum(nan_mask)
    variants_with_nan = np.any(nan_mask, axis=0)
    n_variants_with_nan = np.sum(variants_with_nan)

    logger.warning(
        f"Found {n_nan_total} NaN values across {n_variants_with_nan} variants. "
        f"Imputing with variant-wise mean."
    )

    # Create a copy to avoid modifying original
    imputed_dosages = dosages.copy()

    # Impute variant by variant
    for j in range(dosages.shape[1]):
        if np.any(nan_mask[:, j]):
            # Calculate mean excluding NaN values
            variant_mean = np.nanmean(dosages[:, j])

            # If all values are NaN for this variant, use 0
            if np.isnan(variant_mean):
                variant_rsid = variant_info.iloc[j]["rsid"]
                variant_pos = variant_info.iloc[j]["pos"]
                logger.warning(
                    f"Variant {variant_rsid} at position {variant_pos} has all NaN values. "
                    f"Imputing with 0."
                )
                variant_mean = 0.0

            # Impute NaN values with mean
            imputed_dosages[nan_mask[:, j], j] = variant_mean

    return imputed_dosages, variant_info, sample_ids


def _omit_nan_samples(
    dosages: np.ndarray, variant_info: pd.DataFrame, sample_ids: List[str]
) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    """
    Omit samples with any NaN values.

    Parameters:
    -----------
    dosages : np.ndarray
        Genotype dosage matrix (samples x variants)
    variant_info : pd.DataFrame
        Variant information
    sample_ids : list of str
        Sample IDs

    Returns:
    --------
    tuple
        (filtered_dosages, variant_info, filtered_sample_ids)
    """
    # Find samples with any NaN values
    nan_mask = np.isnan(dosages)
    samples_with_nan = np.any(nan_mask, axis=1)
    n_samples_with_nan = np.sum(samples_with_nan)

    if n_samples_with_nan == 0:
        return dosages, variant_info, sample_ids

    # Log warning about samples being removed
    logger.warning(
        f"Removing {n_samples_with_nan} samples with NaN values out of {len(sample_ids)} total samples."
    )

    # Show first few sample IDs being removed
    removed_sample_ids = [sample_ids[i] for i in np.where(samples_with_nan)[0][:5]]
    if n_samples_with_nan <= 5:
        logger.warning(f"Removed samples: {', '.join(removed_sample_ids)}")
    else:
        logger.warning(
            f"First 5 removed samples: {', '.join(removed_sample_ids)} "
            f"(and {n_samples_with_nan - 5} more)"
        )

    # Keep only samples without NaN
    keep_mask = ~samples_with_nan
    filtered_dosages = dosages[keep_mask, :]
    filtered_sample_ids = [sid for i, sid in enumerate(sample_ids) if keep_mask[i]]

    # Also check if any variants now have all missing values
    all_nan_variants = np.all(np.isnan(filtered_dosages), axis=0)
    if np.any(all_nan_variants):
        n_all_nan = np.sum(all_nan_variants)
        logger.warning(
            f"After removing samples, {n_all_nan} variants have no valid data. "
            f"Consider using 'mean' imputation instead."
        )

    return filtered_dosages, variant_info, filtered_sample_ids


def _report_nan_error(
    dosages: np.ndarray, variant_info: pd.DataFrame, sample_ids: List[str]
) -> None:
    """
    Report detailed error message for NaN values in genotype matrix.

    Parameters:
    -----------
    dosages : np.ndarray
        Genotype dosage matrix
    variant_info : pd.DataFrame
        Variant information
    sample_ids : list of str
        Sample IDs

    Raises:
    -------
    ValueError
        Always raises with detailed NaN information
    """
    nan_mask = np.isnan(dosages)

    # Count samples and variants with NaN
    samples_with_nan = np.any(nan_mask, axis=1)
    variants_with_nan = np.any(nan_mask, axis=0)
    n_samples_with_nan = np.sum(samples_with_nan)
    n_variants_with_nan = np.sum(variants_with_nan)

    # Find first 5 sample/variant pairs with NaN
    nan_locations = np.argwhere(nan_mask)[:5]

    # Build detailed error message
    error_msg = (
        f"Genotype matrix contains NaN values:\n"
        f"  - {n_samples_with_nan} out of {dosages.shape[0]} samples have NaN values\n"
        f"  - {n_variants_with_nan} out of {dosages.shape[1]} variants have NaN values\n"
    )

    if len(nan_locations) > 0:
        error_msg += "\nFirst (up to 5) sample/variant pairs with NaN:\n"
        for i, (sample_idx, variant_idx) in enumerate(nan_locations):
            sample_id = sample_ids[sample_idx]
            variant_rsid = variant_info.iloc[variant_idx]["rsid"]
            variant_pos = variant_info.iloc[variant_idx]["pos"]
            error_msg += (
                f"  {i+1}. Sample '{sample_id}' (index {sample_idx}), "
                f"Variant '{variant_rsid}' at position {variant_pos} (index {variant_idx})\n"
            )

    error_msg += "\nThis may indicate issues with the input BGEN file or variant filtering."

    raise ValueError(error_msg)


def _warn_about_nan(
    dosages: np.ndarray, variant_info: pd.DataFrame, sample_ids: List[str]
) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    """
    Warn about NaN values but return data unchanged.

    Parameters
    ----------
    dosages : np.ndarray
        Genotype dosage matrix
    variant_info : pd.DataFrame
        Variant information
    sample_ids : List[str]
        Sample IDs

    Returns
    -------
    Tuple[np.ndarray, pd.DataFrame, List[str]]
        Original data unchanged
    """
    nan_mask = np.isnan(dosages)

    # Count samples and variants with NaN
    samples_with_nan = np.any(nan_mask, axis=1)
    variants_with_nan = np.any(nan_mask, axis=0)
    n_samples_with_nan = np.sum(samples_with_nan)
    n_variants_with_nan = np.sum(variants_with_nan)

    # Find first 5 sample/variant pairs with NaN
    nan_locations = np.argwhere(nan_mask)[:5]

    # Build warning message
    warning_msg = (
        f"Genotype matrix contains NaN values:\n"
        f"  - {n_samples_with_nan} out of {dosages.shape[0]} samples have NaN values\n"
        f"  - {n_variants_with_nan} out of {dosages.shape[1]} variants have NaN values"
    )

    if len(nan_locations) > 0:
        warning_msg += "\nFirst (up to 5) sample/variant pairs with NaN:\n"
        for i, (sample_idx, variant_idx) in enumerate(nan_locations):
            sample_id = sample_ids[sample_idx]
            variant_rsid = variant_info.iloc[variant_idx]["rsid"]
            variant_pos = variant_info.iloc[variant_idx]["pos"]
            warning_msg += (
                f"  {i+1}. Sample '{sample_id}' (index {sample_idx}), "
                f"Variant '{variant_rsid}' at position {variant_pos} (index {variant_idx})\n"
            )

    warning_msg += (
        "\nNaN values preserved for analysis. Use 'mean' or 'omit' to handle missing data."
    )

    logger.warning(warning_msg)

    # Return data unchanged
    return dosages, variant_info, sample_ids
