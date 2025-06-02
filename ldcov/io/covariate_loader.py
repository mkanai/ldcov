"""
Functions for loading covariate data.

This module provides utilities for loading covariate data from various
file formats, including text files and Excel files.
"""

import logging
import os
from typing import Optional, List
import pandas as pd

from ..utils.categorical_utils import one_hot_encode_categorical

logger = logging.getLogger(__name__)


def load_covariates(
    file_path: str,
    sample_ids: Optional[List[str]] = None,
    id_col: str = "IID",
    cols_to_use: Optional[List[str]] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Load covariate data from file and convert categorical covariates to one-hot encoding.

    Parameters:
    -----------
    file_path : str
        Path to covariate file (supports gs:// paths via pandas)
    sample_ids : list, optional
        Sample IDs to subset
    id_col : str, optional
        Column name for sample IDs (default: "IID")
    cols_to_use : list of str, optional
        Specific columns to use as covariates. If None, all columns except ID are used
    **kwargs : dict
        Additional keyword arguments for reading the file

    Returns:
    --------
    pandas.DataFrame
        Covariate data with categorical variables automatically one-hot encoded

    Notes:
    ------
    - Expects whitespace-delimited file by default
    - Pandas automatically handles Google Cloud Storage paths (gs://) if gcsfs is installed.
    """
    logger.info(f"Loading covariate data from {file_path}")

    # Determine file format from extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    # Handle compressed files
    if ext in [".gz", ".bgz"]:
        _, base_ext = os.path.splitext(os.path.splitext(file_path)[0])
        ext = base_ext.lower()

    # Read file based on format
    try:
        if ext in [".xlsx", ".xls"]:
            covariates = pd.read_excel(file_path, **kwargs)
        elif ext in [".csv"]:
            covariates = pd.read_csv(file_path, **kwargs)
        else:
            # Default to whitespace-delimited
            if "sep" not in kwargs:
                kwargs["sep"] = None  # pandas interprets None as any whitespace
            if "engine" not in kwargs:
                kwargs["engine"] = "python"  # needed for sep=None
            covariates = pd.read_csv(file_path, **kwargs)
    except Exception as e:
        logger.error(f"Error reading covariate file: {e}")
        raise ValueError(f"Failed to read covariate file '{file_path}': {e}")

    # Check if covariate file has sample IDs
    if id_col in covariates.columns:
        logger.info(f"Setting index to column '{id_col}'")
        # Convert ID column to string to ensure proper matching
        covariates[id_col] = covariates[id_col].astype(str)
        covariates = covariates.set_index(id_col)
    else:
        raise ValueError(
            f"ID column '{id_col}' not found in covariate file. "
            f"Available columns: {list(covariates.columns)}"
        )

    # Filter to specific columns if requested
    if cols_to_use is not None:
        logger.info(f"Filtering to specific covariate columns: {cols_to_use}")
        # Check that all requested columns exist
        missing_cols = [col for col in cols_to_use if col not in covariates.columns]
        if missing_cols:
            raise ValueError(
                f"Requested covariate columns not found: {missing_cols}. "
                f"Available columns: {list(covariates.columns)}"
            )
        # Keep only the requested columns
        covariates = covariates[cols_to_use]

    # Subset to specified sample IDs if provided
    if sample_ids is not None:
        logger.info(f"Subsetting to {len(sample_ids)} samples")
        # Filter by index (which should now be the sample IDs)
        missing_samples = [sid for sid in sample_ids if sid not in covariates.index]
        if missing_samples:
            logger.warning(f"Warning: {len(missing_samples)} samples not found in covariate file")
        # Keep only samples that exist in both
        common_samples = [sid for sid in sample_ids if sid in covariates.index]
        if not common_samples:
            raise ValueError("No common samples found between genotype and covariate files")
        covariates = covariates.loc[common_samples]

    # Check if any covariates are categorical and convert to one-hot encoding
    covariates = one_hot_encode_categorical(covariates)

    return covariates
