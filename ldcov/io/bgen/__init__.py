"""
BGEN file format reader implementation.

This module provides a high-performance Cython implementation of BGEN file reading,
optimized for ldcov's specific use cases.
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict, Any
import logging
from tqdm import tqdm

# Import the high-performance BGEN reader
from .reader import BgenReader
from .bgi import BGIReader
from .nan_handler import handle_nan_values

logger = logging.getLogger(__name__)

__all__ = ["BgenReader", "BGIReader", "load_bgen"]


def _create_progress_callback(show_progress: bool, total: int, desc: str = "Loading variants"):
    """Create a progress callback function using tqdm if requested."""
    if not show_progress:
        return None

    pbar = tqdm(total=total, desc=desc, unit="variants")

    def callback(current):
        pbar.n = current
        pbar.refresh()
        if current >= total:
            pbar.close()

    return callback


def _validate_dosages(dosages: np.ndarray) -> None:
    """Validate dosage values are within expected range."""
    # Check for out-of-range dosage values
    min_dosage = np.nanmin(dosages)
    max_dosage = np.nanmax(dosages)

    if min_dosage < 0.0 or max_dosage > 2.0:
        raise ValueError(
            f"Dosage values out of valid range [0, 2] detected "
            f"(min: {min_dosage:.6f}, max: {max_dosage:.6f}). "
            f"This may indicate: "
            f"1) Corrupted BGEN file data, "
            f"2) Memory initialization issue in the reader, "
            f"3) Invalid genotype probabilities that don't sum to 1.0"
        )


def load_bgen(
    file_path: str,
    index_path: Optional[str] = None,
    sample_path: Optional[str] = None,
    region: Optional[str] = None,
    variant_filter: Optional[Dict[str, Any]] = None,
    sample_ids: Optional[List[str]] = None,
    dtype: np.dtype = np.float64,
    show_progress: bool = True,
    nan_action: str = "error",
) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    """
    Load genotype data from BGEN file.

    Parameters
    ----------
    file_path : str
        Path to BGEN file
    index_path : str, optional
        Path to BGI index file. If None, will look for file_path + '.bgi'
    sample_path : str, optional
        Path to sample file
    region : str, optional
        Genomic region in format "chr:start-end"
    variant_filter : dict, optional
        Variant filter from .z file (from load_variant_filter)
    sample_ids : list of str, optional
        Sample IDs to keep. If None, all samples are loaded.
    dtype : numpy.dtype, optional
        Data type for the dosage array (default: np.float64)
    show_progress : bool, optional
        Whether to show progress bars during loading (default: True)
    nan_action : str, optional
        Action for handling NaN values: 'error' (default), 'mean', or 'omit'

    Returns
    -------
    tuple
        (genotypes, variant_info, sample_ids)
        Note: genotypes are returned as floating point values of the specified dtype
        If variant_filter is provided, variants are ordered according to the .z file order
        If sample_ids is provided, only those samples are returned
    """
    # Check BGEN file exists (skip for GCS paths)
    if not file_path.startswith("gs://") and not os.path.exists(file_path):
        raise FileNotFoundError(f"BGEN file not found: {file_path}")

    # Determine BGI path
    if index_path is not None:
        bgi_path = index_path
    else:
        bgi_path = file_path + ".bgi"

    # BGI is mandatory (skip check for GCS paths as they'll be handled by reader)
    if not bgi_path.startswith("gs://") and not os.path.exists(bgi_path):
        raise FileNotFoundError(
            f"BGI index required but not found: {bgi_path}\n"
            f"Please create index using: bgenix -g {file_path}"
        )

    logger.info(f"Opening BGEN file: {file_path}")

    # Create reader with explicit BGI path
    reader = BgenReader(
        file_path, sample_path=sample_path if sample_path else None, bgi_path=bgi_path
    )

    try:
        # Process sample filtering if requested
        sample_indices = None
        filtered_sample_ids = reader.samples

        if sample_ids is not None:
            logger.info(f"Filtering BGEN to {len(sample_ids)} requested samples")
            sample_indices, filtered_sample_ids = reader.get_sample_indices(sample_ids)

            if not sample_indices:
                raise ValueError(
                    "No requested samples found in BGEN file. "
                    "Please check that sample IDs match between files."
                )

            # Only log if there's a difference between requested and found
            if len(sample_indices) < len(sample_ids):
                missing = len(sample_ids) - len(sample_indices)
                logger.warning(
                    f"Found {len(sample_indices)} out of {len(sample_ids)} requested samples. "
                    f"Missing {missing} samples."
                )

        # Parse region if provided
        region_chrom = None
        region_start = None
        region_end = None
        if region:
            from ...utils.region_utils import parse_region

            region_chrom, (region_start, region_end) = parse_region(region)

        # Get total variant count for progress bar
        if variant_filter is not None:
            total_variants = len(variant_filter["positions"])
        elif region:
            # We don't know exact count for region, so estimate
            total_variants = None
        else:
            # Get variant count from reader
            total_variants = reader.nvariants

        # Create progress callback
        progress_callback = None
        if show_progress and total_variants:
            progress_callback = _create_progress_callback(show_progress, total_variants)

        # Convert sample indices to numpy array if needed
        if sample_indices is not None:
            sample_indices = np.array(sample_indices, dtype=np.int32)

        # Load variants using unified method
        dosages, variant_info = reader.load_variants(
            region_chrom=region_chrom,
            region_start=region_start,
            region_end=region_end,
            variant_filter=variant_filter,
            sample_indices=sample_indices,
            dtype=dtype,
            progress_callback=progress_callback,
        )

        # Check if we loaded any variants
        if dosages.size == 0 or dosages.shape[1] == 0:
            raise ValueError(
                "No variants were loaded from the BGEN file. "
                "This may be due to: "
                "1) An empty genomic region, "
                "2) No variants passing the filter criteria, "
                "3) Issues with the BGEN file format"
            )

        # Validate genotypes
        assert np.issubdtype(dosages.dtype, np.floating), "Genotypes must be floating point"
        _validate_dosages(dosages)

        # Handle NaN values if present
        if np.any(np.isnan(dosages)):
            dosages, variant_info, filtered_sample_ids = handle_nan_values(
                dosages, variant_info, filtered_sample_ids, nan_action
            )

        return dosages, variant_info, filtered_sample_ids

    except Exception as e:
        logger.error(f"Error loading BGEN file: {e}")
        raise
    finally:
        reader.close()
