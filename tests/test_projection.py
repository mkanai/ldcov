"""
Tests for projection matrix computation and I/O functionality.
"""

import numpy as np
import pandas as pd
import pytest
import os
from pathlib import Path

from ldcov.compute.projection import (
    ProjectionData,
    compute_projection_matrix,
    save_projection_matrix,
    load_projection_matrix,
    validate_projection_compatibility,
)
from ldcov.compute.covariate import regress_out_covariates


@pytest.fixture
def test_data(tmp_path):
    """Set up test data."""
    # Create test samples and covariates
    np.random.seed(42)
    n_samples = 100
    sample_ids = [f"sample_{i}" for i in range(n_samples)]

    # Create test covariate data
    covariates_df = pd.DataFrame(
        {
            "IID": sample_ids,
            "PC1": np.random.normal(0, 1, n_samples),
            "PC2": np.random.normal(0, 1, n_samples),
            "PC3": np.random.normal(0, 1, n_samples),
            "sex": np.random.choice(["male", "female"], n_samples),
            "age": np.random.normal(50, 10, n_samples),
        }
    )

    # Save covariates to file
    cov_file = tmp_path / "test_covariates.csv"
    covariates_df.to_csv(cov_file, index=False)

    # Create test genotypes
    n_variants = 50
    genotypes = np.random.binomial(2, 0.3, size=(n_samples, n_variants)).astype(np.float64)

    return {
        "n_samples": n_samples,
        "sample_ids": sample_ids,
        "covariates_df": covariates_df,
        "cov_file": cov_file,
        "genotypes": genotypes,
        "n_variants": n_variants,
    }


def test_compute_projection_matrix_basic(test_data):
    """Test basic projection matrix computation."""
    projection_data = compute_projection_matrix(
        covariate_file=str(test_data["cov_file"]),
        sample_ids=test_data["sample_ids"],
    )

    # Check output structure
    assert isinstance(projection_data, ProjectionData)
    assert len(projection_data.sample_ids) == test_data["n_samples"]

    # Check Q matrix properties
    Q = projection_data.Q
    assert Q.shape[0] == test_data["n_samples"]  # n_samples rows
    # Q should have n_covariates columns (PC1, PC2, PC3, sex_male, age, intercept)
    expected_n_covs = 6  # 3 PCs + 1 sex (one-hot becomes 1 col) + age + intercept
    assert Q.shape[1] == expected_n_covs

    # Check Q is orthogonal
    QtQ = Q.T @ Q
    np.testing.assert_allclose(QtQ, np.eye(Q.shape[1]), atol=1e-10)

    # Check metadata
    assert projection_data.n_covariates == expected_n_covs
    assert "intercept" in projection_data.covariate_names


def test_compute_projection_matrix_subset_samples(test_data):
    """Test projection matrix computation with sample subset."""
    subset_samples = test_data["sample_ids"][:50]

    projection_data = compute_projection_matrix(
        covariate_file=str(test_data["cov_file"]),
        sample_ids=subset_samples,
    )

    assert len(projection_data.sample_ids) == 50
    assert projection_data.Q.shape[0] == 50


def test_compute_projection_matrix_specific_covariates(test_data):
    """Test projection matrix computation with specific covariates."""
    projection_data = compute_projection_matrix(
        covariate_file=str(test_data["cov_file"]),
        sample_ids=test_data["sample_ids"],
        covariate_cols=["PC1", "PC2"],  # Only use PC1 and PC2
    )

    # Should have 3 columns: PC1, PC2, intercept
    assert projection_data.Q.shape[1] == 3
    assert projection_data.n_covariates == 3
    assert "PC1" in projection_data.covariate_names
    assert "PC2" in projection_data.covariate_names
    assert "PC3" not in projection_data.covariate_names


def test_save_load_projection_matrix(test_data, tmp_path):
    """Test saving and loading projection matrix."""
    # Compute projection
    projection_data = compute_projection_matrix(
        covariate_file=str(test_data["cov_file"]),
        sample_ids=test_data["sample_ids"],
    )

    # Save to file
    output_file = tmp_path / "test_projection.proj.npz"
    save_projection_matrix(projection_data, str(output_file))

    # Check file exists
    assert os.path.exists(output_file)

    # Load from file
    loaded_data = load_projection_matrix(str(output_file))

    # Check loaded data matches original
    np.testing.assert_array_equal(loaded_data.Q, projection_data.Q)
    assert loaded_data.sample_ids == projection_data.sample_ids
    assert loaded_data.n_covariates == projection_data.n_covariates
    assert loaded_data.covariate_names == projection_data.covariate_names

    # Check metadata
    assert "format_version" in loaded_data.metadata
    assert "creation_date" in loaded_data.metadata


def test_validate_projection_compatibility_exact_match(test_data):
    """Test projection compatibility validation with exact sample match."""
    projection_data = compute_projection_matrix(
        covariate_file=str(test_data["cov_file"]),
        sample_ids=test_data["sample_ids"],
    )

    # Same samples
    Q_subset, indices = validate_projection_compatibility(projection_data, test_data["sample_ids"])

    # Should return full Q matrix
    np.testing.assert_array_equal(Q_subset, projection_data.Q)
    assert indices == list(range(test_data["n_samples"]))


def test_validate_projection_compatibility_subset(test_data):
    """Test projection compatibility validation with genotype sample subset."""
    projection_data = compute_projection_matrix(
        covariate_file=str(test_data["cov_file"]),
        sample_ids=test_data["sample_ids"],
    )

    # Subset of samples in different order
    geno_samples = test_data["sample_ids"][25:75][::-1]  # Reverse order

    Q_subset, indices = validate_projection_compatibility(projection_data, geno_samples)

    # Check dimensions
    assert Q_subset.shape[0] == 50
    assert Q_subset.shape[1] == projection_data.Q.shape[1]

    # Check indices are correct
    assert len(indices) == 50
    # First genotype sample (sample_74) should map to index 74 in projection
    assert indices[0] == 74


def test_validate_projection_compatibility_missing_samples(test_data):
    """Test projection compatibility validation with missing samples."""
    projection_data = compute_projection_matrix(
        covariate_file=str(test_data["cov_file"]),
        sample_ids=test_data["sample_ids"][:50],  # Only first 50 samples
    )

    # Try to use with all samples (including ones not in projection)
    with pytest.raises(ValueError, match="not present in projection matrix"):
        validate_projection_compatibility(projection_data, test_data["sample_ids"])


def test_regression_with_precomputed_projection(test_data):
    """Test that pre-computed projection gives same results as direct computation."""
    from ldcov.compute.covariate import standardize_genotypes

    # Standardize genotypes
    std_geno1 = test_data["genotypes"].copy()
    std_geno1, _, _ = standardize_genotypes(std_geno1, inplace=True)

    std_geno2 = test_data["genotypes"].copy()
    std_geno2, _, _ = standardize_genotypes(std_geno2, inplace=True)

    # Method 1: Direct covariate regression
    covariates = test_data["covariates_df"].set_index("IID")
    adjusted1 = regress_out_covariates(
        std_geno1, covariates=covariates[["PC1", "PC2", "PC3"]], inplace=True
    )

    # Method 2: Pre-computed projection
    projection_data = compute_projection_matrix(
        covariate_file=str(test_data["cov_file"]),
        sample_ids=test_data["sample_ids"],
        covariate_cols=["PC1", "PC2", "PC3"],
    )

    adjusted2 = regress_out_covariates(
        std_geno2, projection_matrix_Q=projection_data.Q, inplace=True
    )

    # Results should be identical
    np.testing.assert_allclose(adjusted1, adjusted2, atol=1e-10)


def test_projection_with_rank_deficient_covariates(test_data, tmp_path):
    """Test projection matrix computation with rank deficient covariates."""
    # Create rank deficient covariates (PC3 = PC1 + PC2)
    df = test_data["covariates_df"].copy()
    df["PC3"] = df["PC1"] + df["PC2"]
    df["PC4"] = df["PC1"] * 2  # Another linear dependency

    # Save to file
    rank_def_file = tmp_path / "rank_def_covariates.csv"
    df.to_csv(rank_def_file, index=False)

    # Should handle rank deficiency gracefully
    projection_data = compute_projection_matrix(
        covariate_file=str(rank_def_file),
        sample_ids=test_data["sample_ids"],
        covariate_cols=["PC1", "PC2", "PC3", "PC4"],
    )

    # Should have fewer columns than requested due to rank deficiency
    assert projection_data.Q.shape[1] < 5  # Less than 4 PCs + intercept

    # Q should still be orthogonal
    QtQ = projection_data.Q.T @ projection_data.Q
    np.testing.assert_allclose(QtQ, np.eye(projection_data.Q.shape[1]), atol=1e-10)
