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

    # Check for NaN values in genotypes
    if np.any(np.isnan(genotypes)):
        raise ValueError(
            "Genotype matrix contains NaN values. "
            "This may indicate issues with the input BGEN file or variant filtering."
        )

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

    # Convert to numpy array, ensuring float type
    X = covariates.to_numpy(dtype=np.float64)

    logger.info(f"Performing FWL projection with {X.shape[1]} covariates (including intercept)")

    # Check if we have too few samples relative to covariates
    n_samples, n_covariates = X.shape
    if n_samples <= n_covariates:
        raise ValueError(
            f"Number of samples ({n_samples}) is less than or equal to number of covariates ({n_covariates}). "
            "This will result in a rank-deficient system. Please use fewer covariates."
        )

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
        If QR decomposition fails
    """
    try:
        # Check for rank deficiency
        rank = np.linalg.matrix_rank(X)
        if rank < X.shape[1]:
            logger.warning(
                f"Covariate matrix is rank deficient (rank {rank} < {X.shape[1]} columns). "
                "Attempting to handle multicollinearity."
            )

            # Try to identify and remove linearly dependent columns
            _, R = np.linalg.qr(X)
            tol = 1e-10
            independent_cols = np.abs(np.diag(R)) > tol

            if np.sum(independent_cols) == 0:
                raise ValueError("All covariate columns appear to be linearly dependent")

            # Use only independent columns
            X_reduced = X[:, independent_cols]
            logger.info(f"Using {np.sum(independent_cols)} independent columns out of {X.shape[1]}")

            # Perform QR decomposition on reduced matrix
            Q, R = np.linalg.qr(X_reduced)
        else:
            # Perform standard QR decomposition
            Q, R = np.linalg.qr(X)

        # Check for numerical issues in R
        if np.any(np.isnan(R)) or np.any(np.isinf(R)):
            raise ValueError("QR decomposition produced NaN or Inf values")

        # Compute the projection onto the orthogonal complement of X
        projection = Q @ (Q.T @ G)

        # Check for NaN in projection before modifying G
        if np.any(np.isnan(projection)):
            raise ValueError("Projection calculation produced NaN values")

        # This modifies G in-place
        G -= projection

        # Check if all values became zero (or near zero)
        if np.allclose(G, 0, atol=1e-10):
            logger.warning(
                "FWL projection resulted in all near-zero values. "
                "This may indicate perfect collinearity between genotypes and covariates."
            )

        return G

    except Exception as e:
        logger.error(f"FWL projection failed: {str(e)}")
        logger.warning("Falling back to pseudoinverse method")

        try:
            # Fallback: Use pseudoinverse (Moore-Penrose) which is more robust
            # P_X = X @ pinv(X'X) @ X'
            # Residual: (I - P_X)G

            # Compute pseudoinverse
            XtX = X.T @ X
            XtX_pinv = np.linalg.pinv(XtX, rcond=1e-10)

            # Check for numerical issues
            if np.any(np.isnan(XtX_pinv)) or np.any(np.isinf(XtX_pinv)):
                raise ValueError("Pseudoinverse calculation failed")

            # Compute projection matrix
            P_X = X @ XtX_pinv @ X.T

            # Apply projection
            G -= P_X @ G

            # Final check
            if np.any(np.isnan(G)):
                raise ValueError("Pseudoinverse method also resulted in NaN values")

            logger.info("Successfully applied FWL projection using pseudoinverse method")
            return G

        except Exception as fallback_error:
            logger.error(f"Fallback method also failed: {str(fallback_error)}")
            logger.error(
                "Unable to regress out covariates. Consider checking for: "
                "1) Perfect collinearity between covariates, "
                "2) Covariates that perfectly explain genotype variation, "
                "3) Numerical precision issues with input data"
            )
            raise ValueError(
                "Both QR decomposition and pseudoinverse methods failed. "
                "Cannot regress out covariates from genotypes."
            )
