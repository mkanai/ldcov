"""
Projection matrix computation and I/O for pre-computed covariate adjustment.

This module provides functions to pre-compute, save, and load the orthogonal
projection matrix Q from the QR decomposition of covariates, enabling efficient
reuse across multiple genomic analyses.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, NamedTuple
import logging
from datetime import datetime
import os

from ..io.covariate_loader import load_covariates

logger = logging.getLogger(__name__)


def _ldcov_version() -> str:
    """Return the installed ldcov version for provenance metadata."""
    from ldcov import __version__

    return __version__


# Version for projection matrix format
PROJECTION_FORMAT_VERSION = "1.0"


class ProjectionData(NamedTuple):
    """Container for projection matrix data."""

    Q: np.ndarray  # Orthogonal matrix from QR decomposition
    sample_ids: List[str]  # Ordered sample IDs
    n_covariates: int  # Number of covariates (including intercept)
    covariate_names: List[str]  # Names of covariate columns
    metadata: Dict[str, str]  # Additional metadata


def compute_projection_matrix(
    covariate_file: str,
    sample_ids: Optional[List[str]] = None,
    sample_file: Optional[str] = None,
    covariate_id_col: str = "IID",
    covariate_cols: Optional[List[str]] = None,
) -> ProjectionData:
    """
    Compute orthogonal projection matrix Q from covariates.

    Parameters:
    -----------
    covariate_file : str
        Path to covariate file
    sample_ids : list of str, optional
        Sample IDs to include. If None, all samples in covariate file are used
    sample_file : str, optional
        Path to sample file (.sample format) to determine sample ordering
    covariate_id_col : str, optional
        Column name for sample IDs in covariate file (default: "IID")
    covariate_cols : list of str, optional
        Specific columns to use as covariates. If None, all columns except ID are used

    Returns:
    --------
    ProjectionData
        Named tuple containing Q matrix and associated metadata
    """
    # Load sample IDs from sample file if provided
    if sample_file and sample_ids is None:
        logger.info(f"Loading sample IDs from {sample_file}")
        # Simple sample file reading - assuming standard format
        with open(sample_file, "r") as f:
            lines = f.readlines()
            # Skip header lines
            sample_ids = []
            for line in lines[2:]:  # Skip two header lines
                fields = line.strip().split()
                if fields:
                    sample_ids.append(fields[0])

    # If no sample IDs provided, we'll use all from covariate file
    if sample_ids is None:
        logger.info("No sample IDs specified, will use all samples from covariate file")
        # Load covariates without filtering
        temp_covs = pd.read_csv(covariate_file, nrows=0)  # Just get columns
        if covariate_id_col not in temp_covs.columns:
            raise ValueError(f"ID column '{covariate_id_col}' not found in covariate file")

        # Read all IDs
        id_df = pd.read_csv(covariate_file, usecols=[covariate_id_col])
        sample_ids = id_df[covariate_id_col].tolist()

    logger.info(f"Loading covariates for {len(sample_ids)} samples")
    covariates = load_covariates(
        covariate_file, sample_ids, id_col=covariate_id_col, cols_to_use=covariate_cols
    )

    # Track covariate names before adding intercept
    covariate_names = list(covariates.columns)

    # Add intercept
    covariates["intercept"] = 1.0
    covariate_names.append("intercept")

    # Convert to numpy array
    X = covariates.to_numpy(dtype=np.float64)

    logger.info(f"Computing QR decomposition for {X.shape[1]} covariates (including intercept)")

    # Check for rank deficiency
    n_samples, n_covariates = X.shape
    if n_samples <= n_covariates:
        raise ValueError(
            f"Number of samples ({n_samples}) must be greater than number of covariates ({n_covariates})"
        )

    # Perform QR decomposition
    try:
        rank = np.linalg.matrix_rank(X)
        if rank < X.shape[1]:
            logger.warning(
                f"Covariate matrix is rank deficient (rank {rank} < {X.shape[1]} columns). "
                "Using reduced QR decomposition."
            )

            # Identify independent columns
            _, R = np.linalg.qr(X)
            tol = 1e-10
            independent_cols = np.abs(np.diag(R)) > tol

            if np.sum(independent_cols) == 0:
                raise ValueError("All covariate columns appear to be linearly dependent")

            # Use only independent columns
            X_reduced = X[:, independent_cols]
            covariate_names_reduced = [
                name for i, name in enumerate(covariate_names) if independent_cols[i]
            ]
            logger.info(f"Using {np.sum(independent_cols)} independent columns out of {X.shape[1]}")

            # Perform QR on reduced matrix
            Q, R = np.linalg.qr(X_reduced)

            # Update metadata
            n_covariates = X_reduced.shape[1]
            covariate_names = covariate_names_reduced
        else:
            # Standard QR decomposition
            Q, R = np.linalg.qr(X)
            n_covariates = X.shape[1]

        # Verify Q is orthogonal
        QtQ = Q.T @ Q
        if not np.allclose(QtQ, np.eye(Q.shape[1]), atol=1e-10):
            logger.warning("Q matrix may not be perfectly orthogonal due to numerical precision")

    except Exception as e:
        logger.error(f"QR decomposition failed: {str(e)}")
        raise

    # Get ordered sample IDs from covariates DataFrame
    ordered_sample_ids = covariates.index.tolist()

    # Create metadata
    metadata = {
        "format_version": PROJECTION_FORMAT_VERSION,
        "creation_date": datetime.now().isoformat(),
        "ldcov_version": _ldcov_version(),
        "covariate_file": os.path.abspath(covariate_file),
        "n_samples": str(len(ordered_sample_ids)),
        "n_covariates_original": str(X.shape[1]),
        "n_covariates_used": str(n_covariates),
    }

    return ProjectionData(
        Q=Q,
        sample_ids=ordered_sample_ids,
        n_covariates=n_covariates,
        covariate_names=covariate_names,
        metadata=metadata,
    )


def save_projection_matrix(projection_data: ProjectionData, output_file: str) -> None:
    """
    Save projection matrix data to NPZ file.

    Parameters:
    -----------
    projection_data : ProjectionData
        Projection data to save
    output_file : str
        Output file path (should end with .proj.npz)
    """
    if not output_file.endswith(".proj.npz"):
        output_file += ".proj.npz"

    logger.info(f"Saving projection matrix to {output_file}")

    # Prepare data for saving
    save_dict = {
        "Q": projection_data.Q,
        "sample_ids": np.array(projection_data.sample_ids, dtype="U"),  # Unicode strings
        "n_covariates": projection_data.n_covariates,
        "covariate_names": np.array(projection_data.covariate_names, dtype="U"),
    }

    # Add metadata as separate arrays (NPZ doesn't handle dicts well)
    for key, value in projection_data.metadata.items():
        save_dict[f"metadata_{key}"] = value

    # Save as compressed NPZ
    np.savez_compressed(output_file, **save_dict)

    logger.info(
        f"Projection matrix saved: {projection_data.Q.shape} for {len(projection_data.sample_ids)} samples"
    )


def load_projection_matrix(projection_file: str) -> ProjectionData:
    """
    Load projection matrix data from NPZ file.

    Parameters:
    -----------
    projection_file : str
        Path to projection matrix file (.proj.npz)

    Returns:
    --------
    ProjectionData
        Loaded projection data
    """
    logger.info(f"Loading projection matrix from {projection_file}")

    if not os.path.exists(projection_file):
        raise FileNotFoundError(f"Projection matrix file not found: {projection_file}")

    # Load NPZ file
    with np.load(projection_file, allow_pickle=False) as data:
        # Extract main data
        Q = data["Q"]
        sample_ids = data["sample_ids"].tolist()
        n_covariates = int(data["n_covariates"])
        covariate_names = data["covariate_names"].tolist()

        # Extract metadata
        metadata = {}
        for key in data.files:
            if key.startswith("metadata_"):
                metadata_key = key.replace("metadata_", "")
                # Handle different types
                value = data[key]
                if value.ndim == 0:  # Scalar
                    metadata[metadata_key] = str(value)
                else:
                    metadata[metadata_key] = value

        # Validate format version
        format_version = metadata.get("format_version", "unknown")
        if format_version != PROJECTION_FORMAT_VERSION:
            logger.warning(
                f"Projection matrix format version mismatch: "
                f"expected {PROJECTION_FORMAT_VERSION}, got {format_version}"
            )

    logger.info(
        f"Loaded projection matrix: {Q.shape} for {len(sample_ids)} samples, "
        f"{n_covariates} covariates"
    )

    return ProjectionData(
        Q=Q,
        sample_ids=sample_ids,
        n_covariates=n_covariates,
        covariate_names=covariate_names,
        metadata=metadata,
    )


def validate_projection_compatibility(
    projection_data: ProjectionData, genotype_sample_ids: List[str]
) -> Tuple[np.ndarray, List[int]]:
    """
    Validate and align projection matrix with genotype samples.

    Parameters:
    -----------
    projection_data : ProjectionData
        Pre-computed projection data
    genotype_sample_ids : list of str
        Sample IDs from genotype file

    Returns:
    --------
    tuple
        (Q_subset, sample_indices) where:
        - Q_subset: Q matrix rows corresponding to genotype samples
        - sample_indices: Indices mapping genotype samples to Q rows

    Raises:
    -------
    ValueError
        If samples are incompatible
    """
    # Create mapping from sample ID to index in projection matrix
    proj_sample_map = {sid: i for i, sid in enumerate(projection_data.sample_ids)}

    # Check if all genotype samples are in projection matrix
    sample_indices = []
    missing_samples = []

    for sid in genotype_sample_ids:
        if sid in proj_sample_map:
            sample_indices.append(proj_sample_map[sid])
        else:
            missing_samples.append(sid)

    if missing_samples:
        raise ValueError(
            f"Found {len(missing_samples)} samples in genotype file not present in projection matrix. "
            f"First 10 missing: {missing_samples[:10]}"
        )

    # Extract subset of Q matrix
    Q_subset = projection_data.Q[sample_indices, :]

    logger.info(
        f"Validated projection matrix compatibility: "
        f"{len(genotype_sample_ids)} genotype samples matched in projection matrix"
    )

    return Q_subset, sample_indices
