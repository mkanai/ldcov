"""
Core correlation computation module for calculating linkage disequilibrium.

This module provides functions for computing correlation between genetic variants,
which can be used as a measure of linkage disequilibrium.
"""

from typing import Optional, List, Tuple
import logging
import os

# Defer numpy/pandas imports until actually needed

logger = logging.getLogger(__name__)


def load_and_adjust_genotypes(
    genotype_file: str,
    covariate_file: Optional[str] = None,
    projection_matrix_file: Optional[str] = None,
    region: Optional[str] = None,
    index_file: Optional[str] = None,
    sample_file: Optional[str] = None,
    z_file: Optional[str] = None,
    covariate_id_col: str = "IID",
    covariate_cols: Optional[List[str]] = None,
    show_progress: bool = False,
    nan_action: str = "error",
):
    """
    Load genotypes, standardize them, and optionally adjust for covariates.

    Parameters:
    -----------
    genotype_file : str
        Path to BGEN genotype file
    covariate_file : str, optional
        Path to covariate file for adjustment. Ignored if projection_matrix_file is provided.
    projection_matrix_file : str, optional
        Path to pre-computed projection matrix file (.proj.npz). If provided, uses this
        instead of computing from covariates.
    region : str, optional
        Genomic region in format "chr:start-end"
    index_file : str, optional
        Path to BGEN index file (.bgi)
    sample_file : str, optional
        Path to sample file
    z_file : str, optional
        Path to .z file specifying which variants to load and their order
    covariate_id_col : str, optional
        Column name for sample IDs in covariate file (default: "IID")
    covariate_cols : list of str, optional
        Specific columns to use as covariates. If None, all columns except ID are used
    show_progress : bool, optional
        Whether to show progress bars during loading (default: False)
    nan_action : str, optional
        Action for handling NaN values: 'error' (default), 'mean', or 'omit'

    Returns:
    --------
    tuple
        (standardized_genotypes, variant_info, sample_ids, means, norms)
        - standardized_genotypes: Standardized (and possibly adjusted) genotypes
        - variant_info: DataFrame with variant information (ordered by .z file if provided)
        - sample_ids: List of sample IDs
        - means: Original means before standardization
        - norms: Original norms before standardization
    """
    # Lazy import numpy and pandas
    import numpy as np
    import pandas as pd
    
    # Load genotype data
    logger.info(f"Loading genotype data from BGEN file: {genotype_file}")

    # Process .z file if provided
    variant_filter = None
    if z_file is not None:
        from ..utils.variant_filter import load_variant_filter

        logger.info(f"Reading variant filter from .z file: {z_file}")
        variant_filter = load_variant_filter(z_file)

    # Determine which samples to load based on covariate/projection data
    samples_to_load = None

    if projection_matrix_file:
        # If using projection matrix, we need to load only samples in the projection
        logger.info(f"Pre-loading projection matrix to determine samples")
        from .projection import load_projection_matrix

        projection_data = load_projection_matrix(projection_matrix_file)
        samples_to_load = projection_data.sample_ids
        logger.info(f"Will load {len(samples_to_load)} samples from projection matrix")

    elif covariate_file:
        # If using covariates, we can pre-load the sample IDs to filter early
        logger.info(f"Pre-loading covariate sample IDs for early filtering")
        # Quick load just to get sample IDs
        import pandas as pd

        # Try to read with automatic delimiter detection
        try:
            cov_df = pd.read_csv(
                covariate_file,
                sep=None,
                engine="python",
                usecols=[covariate_id_col],
                dtype={covariate_id_col: str},
            )
        except Exception:
            # Fallback to reading all columns if usecols fails
            cov_df = pd.read_csv(covariate_file, sep=None, engine="python", dtype=str)
            if covariate_id_col not in cov_df.columns:
                raise ValueError(f"Column '{covariate_id_col}' not found in covariate file")
            cov_df = cov_df[[covariate_id_col]]

        samples_to_load = cov_df[covariate_id_col].astype(str).tolist()
        logger.info(f"Will load {len(samples_to_load)} samples from covariate file")

    # Load genotypes with sample filtering if applicable
    # Lazy import load_bgen
    from ..io import load_bgen
    
    genotypes, variant_info, sample_ids = load_bgen(
        file_path=genotype_file,
        index_path=index_file,
        sample_path=sample_file,
        region=region,
        variant_filter=variant_filter,
        sample_ids=samples_to_load,
        show_progress=show_progress,
        nan_action=nan_action,
    )

    # Standardize genotypes
    logger.info("Standardizing genotypes")
    # Lazy import standardize_genotypes
    from .covariate import standardize_genotypes
    
    standardized_genotypes, means, norms = standardize_genotypes(
        genotypes, center=True, scale=True, inplace=True
    )

    # Apply covariate adjustment if provided
    if projection_matrix_file:
        # Use pre-computed projection matrix
        # Note: samples were already filtered during loading
        logger.info("Using pre-computed projection matrix for adjustment")

        # Projection data was already loaded above, but we load it again
        # to avoid variable scope issues
        if "projection_data" not in locals():
            from .projection import load_projection_matrix

            projection_data = load_projection_matrix(projection_matrix_file)

        logger.info("Adjusting genotypes using pre-computed projection matrix")
        # Lazy import regress_out_covariates
        from .covariate import regress_out_covariates
        
        standardized_genotypes = regress_out_covariates(
            standardized_genotypes, projection_matrix_Q=projection_data.Q, inplace=True
        )

    elif covariate_file:
        # Original workflow: load covariates and compute projection
        # Note: samples were already filtered during loading if covariate file was provided
        logger.info(f"Loading full covariates from {covariate_file}")
        # Lazy import load_covariates
        from ..io.covariate_loader import load_covariates
        
        covariates = load_covariates(
            covariate_file, sample_ids, id_col=covariate_id_col, cols_to_use=covariate_cols
        )

        # Since we already filtered during loading, all samples should match
        if len(covariates) != len(sample_ids):
            logger.warning(
                f"Unexpected sample mismatch after pre-filtering: "
                f"{len(covariates)} covariate samples vs {len(sample_ids)} genotype samples"
            )

        logger.info("Adjusting genotypes for covariates")
        # Import regress_out_covariates if not already imported
        if 'regress_out_covariates' not in locals():
            from .covariate import regress_out_covariates
        
        standardized_genotypes = regress_out_covariates(
            standardized_genotypes, covariates=covariates, inplace=True
        )

    return standardized_genotypes, variant_info, sample_ids, means, norms


def compute_ld_from_standardized(
    standardized_genotypes,
    variant_info,
    output_file: str,
    output_format: str = "matrix",
) -> None:
    """
    Compute LD matrix from standardized genotypes and save to file.

    Parameters:
    -----------
    standardized_genotypes : numpy.ndarray
        Standardized genotype matrix (samples x variants)
    variant_info : pandas.DataFrame
        Variant information
    output_file : str
        Path to output LD file
    output_format : str
        Output format ("matrix", "long", "bcor")
    """
    # Lazy imports
    import numpy as np
    import pandas as pd
    
    logger.info("Computing LD matrix")
    corr_matrix = compute_correlation_matrix(standardized_genotypes)

    # Save to file
    # Lazy import save_correlation_matrix
    from ..io.correlation_io import save_correlation_matrix
    
    save_correlation_matrix(
        corr_matrix, output_file, variant_info=variant_info, output_format=output_format
    )

    logger.info(f"LD matrix saved to {output_file}")


def compute_correlation_matrix(standardized_genotypes):
    """
    Compute correlation matrix from standardized genotype data.

    Parameters:
    -----------
    standardized_genotypes : numpy.ndarray
        Standardized genotype matrix (samples x variants). Genotypes should already be
        standardized (centered and scaled) before calling this function.

    Returns:
    --------
    numpy.ndarray
        Correlation matrix (variants x variants).
    """
    # Lazy import numpy
    import numpy as np
    
    # Calculate correlations using dot product of standardized genotypes: t(X_scaled) %*% X_scaled
    corr_matrix = np.dot(standardized_genotypes.T, standardized_genotypes)

    return corr_matrix
