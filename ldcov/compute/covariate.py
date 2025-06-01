"""
Covariate adjustment module for regressing out covariates from genotypes.

This module implements Frisch-Waugh-Lovell (FWL) projection for adjusting genotypes
for covariates, following the implementation in the UK Biobank Pan Ancestry project.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Union, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


def standardize_genotypes(
    genotypes: np.ndarray, center: bool = True, scale: bool = True, inplace: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize genotypes by centering and scaling using L2 normalization.

    Parameters:
    -----------
    genotypes : numpy.ndarray
        Genotype matrix (samples x variants) in allelic scale, must be floating point
    center : bool, optional
        Whether to center the genotypes (default: True)
    scale : bool, optional
        Whether to scale the genotypes (default: True)
    inplace : bool, optional
        Whether to modify the input genotypes in place (default: True)

    Returns:
    --------
    Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Tuple containing:
        - Standardized genotypes
        - Array of means used for centering
        - Array of norms used for scaling
    """
    # Genotypes should be float32 or float64 from the loader.
    # Assert to catch any issues with program flow.
    assert np.issubdtype(
        genotypes.dtype, np.floating
    ), "Genotypes must be floating point for standardization"

    # Calculate means for centering
    means = np.mean(genotypes, axis=0) if center else np.zeros(genotypes.shape[1])

    # Use input array or create a new one
    if inplace:
        standardized_genotypes = genotypes
    else:
        standardized_genotypes = genotypes.copy()

    # Center each column (variant)
    if center:
        standardized_genotypes -= means[np.newaxis, :]

    # Calculate L2 norms (sqrt of sum of squares) for scaling
    if scale:
        # Calculate norm (sqrt of sum of squares for each column)
        norms = np.sqrt(np.sum(standardized_genotypes**2, axis=0))

        # Avoid division by zero for variants with no variance
        norms[norms == 0] = 1.0

        # Scale by L2 norm
        standardized_genotypes /= norms[np.newaxis, :]

    return standardized_genotypes, means, norms


def regress_out_covariates(
    standardized_genotypes: np.ndarray,
    covariates: Union[np.ndarray, pd.DataFrame],
    inplace: bool = True,
) -> np.ndarray:
    """
    Regress out covariates from standardized genotypes using FWL (Frisch-Waugh-Lovell) projection.

    Parameters:
    -----------
    standardized_genotypes : numpy.ndarray
        Standardized genotype matrix (samples x variants). Genotypes should already be
        standardized (centered and scaled) before calling this function.
    covariates : numpy.ndarray or pandas.DataFrame
        Covariate matrix (samples x covariates). Categorical covariates should already be
        one-hot encoded by the load_covariates function.
    inplace : bool, optional
        Whether to modify the input genotypes in place (default: True). This saves memory but
        modifies the original array.

    Returns:
    --------
    numpy.ndarray
        Standardized genotypes with covariates regressed out

    Raises:
    -------
    ValueError
        If pandas DataFrame contains categorical columns.
    """
    # Input genotypes should already be standardized
    # Create a copy if not inplace
    if not inplace:
        standardized_genotypes = standardized_genotypes.copy()

    # Convert covariates to DataFrame if numpy array
    if isinstance(covariates, np.ndarray):
        covariates = pd.DataFrame(covariates)

    # Check if any covariates are still categorical and raise error
    for col in covariates.columns:
        if covariates[col].dtype == "object" or isinstance(
            covariates[col].dtype, pd.CategoricalDtype
        ):
            raise ValueError(
                f"Column '{col}' is categorical. All categorical covariates should be "
                f"one-hot encoded in the load_covariates function before calling regress_out_covariates."
            )

    # Add intercept
    covariates["intercept"] = 1.0

    # Convert to numpy array
    X = covariates.to_numpy()

    logger.info(f"Performing FWL projection with {X.shape[1]} covariates (including intercept)")

    # Perform FWL projection on standardized genotypes
    # This modifies standardized_genotypes if inplace=True
    adjusted_standardized_genotypes = _apply_fwl_projection(standardized_genotypes, X)

    return adjusted_standardized_genotypes


def _apply_fwl_projection(G: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Apply FWL projection to regress out covariates.

    Parameters:
    -----------
    G : numpy.ndarray
        Genotype matrix (samples x variants), already standardized
    X : numpy.ndarray
        Covariate matrix with intercept (samples x (covariates+1))

    Returns:
    --------
    numpy.ndarray
        Residualized genotypes (still in standardized scale)

    Raises:
    -------
    ValueError
        If G is empty (G.size == 0)
    """
    # Check if G is empty - raise error if so
    if G.size == 0:
        raise ValueError("Cannot apply FWL projection to empty genotype matrix")

    # Calculate projection matrix
    # Following the FWL approach:
    # P_X = X(X'X)^(-1)X'
    # Residual: (I - P_X)G

    # Using a more numerically stable approach with QR decomposition
    Q, R = np.linalg.qr(X)
    # Compute the projection onto the orthogonal complement of X
    # This modifies G in-place
    G -= Q @ (Q.T @ G)

    return G
