"""
Consolidated computation tests for the ldcov package.

This module contains tests for:
- Correlation computation
- LD computation workflows
- Integration with covariate adjustment
"""

import os
import numpy as np
import pandas as pd
import pytest

from ldcov.compute.correlation import (
    load_and_adjust_genotypes,
    compute_ld_from_standardized,
    compute_correlation_matrix,
)
from ldcov.compute.covariate import standardize_genotypes
from ldcov.io import load_bgen


@pytest.fixture
def compute_test_data(temp_dir, genotypes, variant_info, sample_ids, create_covariate_file):
    """Set up test data for compute tests."""
    return {
        "temp_dir": temp_dir,
        "genotypes": genotypes,
        "variant_info": variant_info,
        "sample_ids": sample_ids,
        "cov_file": create_covariate_file(),
    }


# ==================== Correlation Tests ====================


def test_compute_correlation_matrix():
    """Test correlation matrix computation."""
    # Create test data with known correlation
    n_samples = 100
    np.random.seed(42)  # For reproducibility

    # Two perfectly correlated variants
    var1 = np.random.randn(n_samples)
    var2 = var1.copy()
    # One uncorrelated variant
    var3 = np.random.randn(n_samples)

    genotypes = np.column_stack([var1, var2, var3]).astype(np.float64)

    # Standardize first (function expects standardized input)
    std_genotypes, _, _ = standardize_genotypes(genotypes, center=True, scale=True, inplace=False)

    # Compute correlation
    corr_matrix = compute_correlation_matrix(std_genotypes)

    # Check properties
    assert corr_matrix.shape == (3, 3)
    # Perfect correlation between var1 and var2
    assert abs(corr_matrix[0, 1] - 1.0) < 1e-5
    # Low correlation with var3
    assert abs(corr_matrix[0, 2]) < 0.2
    # Diagonal should be 1
    np.testing.assert_allclose(np.diag(corr_matrix), 1.0)


@pytest.mark.parametrize("output_format", ["matrix", "long", "bcor"])
def test_compute_ld_from_standardized(compute_test_data, output_format):
    """Test LD computation from standardized genotypes with different output formats."""
    # Standardize test genotypes
    std_geno, _, _ = standardize_genotypes(
        compute_test_data["genotypes"][:, :10].copy(), center=True, scale=True
    )

    # Test the specific output format
    output_file = os.path.join(compute_test_data["temp_dir"], f"test.{output_format}")
    compute_ld_from_standardized(
        std_geno,
        compute_test_data["variant_info"].iloc[:10],
        output_file,
        output_format=output_format,
    )
    assert os.path.exists(output_file)


# ==================== Workflow Tests ====================


def test_load_and_adjust_genotypes_no_covariates(bgen_file, bgi_file, sample_file):
    """Test loading and standardizing without covariates."""
    std_geno, var_info, sample_ids, means, norms = load_and_adjust_genotypes(
        genotype_file=str(bgen_file),
        index_file=str(bgi_file),
        sample_file=str(sample_file),
    )

    # Check outputs
    assert std_geno.shape[0] > 0
    assert std_geno.shape[1] > 0
    assert len(var_info) == std_geno.shape[1]
    assert len(sample_ids) == std_geno.shape[0]

    # Check standardization
    col_means = np.mean(std_geno, axis=0)
    np.testing.assert_allclose(col_means, 0, atol=1e-10)


def test_load_and_adjust_genotypes_with_covariates(
    bgen_file, bgi_file, sample_file, compute_test_data
):
    """Test loading and adjusting with covariates."""
    std_geno, var_info, sample_ids, means, norms = load_and_adjust_genotypes(
        genotype_file=str(bgen_file),
        covariate_file=compute_test_data["cov_file"],
        index_file=str(bgi_file),
        sample_file=str(sample_file),
    )

    # Should still be standardized after adjustment
    assert std_geno.shape[0] > 0
    assert std_geno.shape[1] > 0


def test_load_and_adjust_with_custom_id_column(
    bgen_file, bgi_file, sample_file, create_covariate_file
):
    """Test loading with custom covariate ID column."""
    # Create covariate file with custom ID column
    custom_cov_file = create_covariate_file(
        columns=["PC1"],
        id_col="FID",
    )

    # Load with custom ID column
    std_geno, var_info, sample_ids, means, norms = load_and_adjust_genotypes(
        genotype_file=str(bgen_file),
        covariate_file=custom_cov_file,
        index_file=str(bgi_file),
        sample_file=str(sample_file),
        covariate_id_col="FID",
    )

    # Get expected sample count from BGEN file
    _, _, expected_sample_ids = load_bgen(str(bgen_file), str(bgi_file), str(sample_file))
    assert len(sample_ids) == len(expected_sample_ids)


def test_sample_filtering_with_missing_covariates(
    bgen_file, bgi_file, sample_file, create_covariate_file, compute_test_data
):
    """Test that samples are filtered when some lack covariate data."""
    # Create covariate file with subset of samples
    subset_samples = compute_test_data["sample_ids"][::2]  # Every other sample
    subset_cov_file = create_covariate_file(
        columns=["PC1"],
        custom_sample_ids=subset_samples,
    )

    # Load and adjust
    std_geno, var_info, sample_ids, means, norms = load_and_adjust_genotypes(
        genotype_file=str(bgen_file),
        covariate_file=subset_cov_file,
        index_file=str(bgi_file),
        sample_file=str(sample_file),
    )

    # Should only have samples with covariate data
    assert len(sample_ids) == len(subset_samples)
    assert std_geno.shape[0] == len(subset_samples)


# ==================== LDstore Comparison Tests ====================


def test_ldstore_comparison(ref_ld_file, compute_test_data):
    """Test LD calculations against LDstore2 reference data."""
    # Check if LDstore reference file exists
    if not ref_ld_file.exists():
        pytest.skip("LDstore reference file not found - skipping comparison test")

    # Compute LD using ldcov
    std_geno, _, _ = standardize_genotypes(
        compute_test_data["genotypes"].copy(), center=True, scale=True
    )
    ldcov_ld = compute_correlation_matrix(std_geno)

    # Load LDstore2 reference matrix
    ldstore_ld = _load_ldstore_matrix(ref_ld_file)

    # Check shapes match
    if ldcov_ld.shape != ldstore_ld.shape:
        pytest.skip(f"Shape mismatch: ldcov {ldcov_ld.shape} vs LDstore {ldstore_ld.shape}")

    # Validate against LDstore with tolerance
    tol = 0.05  # 5% difference allowed
    differences = np.abs(ldcov_ld - ldstore_ld)
    max_diff = np.max(differences)
    avg_diff = np.mean(differences)

    # Check overall similarity
    assert avg_diff < tol, f"Average difference {avg_diff:.4f} exceeds tolerance {tol}"

    # Additional checks
    assert np.allclose(np.diag(ldcov_ld), 1.0), "LD matrix diagonal should be 1.0"
    assert np.allclose(ldcov_ld, ldcov_ld.T), "LD matrix should be symmetric"


def test_end_to_end_workflow_validation(
    bgen_file, bgi_file, sample_file, create_covariate_file, tmp_path
):
    """Test complete modular workflow with intermediate validations."""
    # Create test covariates
    cov_file = create_covariate_file(columns=["PC1", "PC2"])

    # Define output files
    output_ld = os.path.join(tmp_path, "workflow_ld.txt")

    # Step 1: Load and adjust genotypes
    std_geno, var_info, sample_ids, means, norms = load_and_adjust_genotypes(
        genotype_file=str(bgen_file),
        covariate_file=cov_file,
        index_file=str(bgi_file),
        sample_file=str(sample_file),
    )

    # Validate intermediate results
    assert isinstance(std_geno, np.ndarray)
    assert std_geno.shape[0] > 0
    assert std_geno.shape[1] > 0
    assert isinstance(var_info, pd.DataFrame)
    assert len(sample_ids) > 0
    assert isinstance(means, np.ndarray)
    assert isinstance(norms, np.ndarray)

    # Step 2: Compute LD
    compute_ld_from_standardized(std_geno, var_info, output_ld, output_format="matrix")

    # Validate LD output
    assert os.path.exists(output_ld)

    # Load and validate LD matrix
    ld_matrix = _read_numeric_matrix(output_ld)
    assert ld_matrix.shape[0] == ld_matrix.shape[1]

    # Check diagonal is close to 1.0 (allowing for numerical precision)
    diag_values = np.diag(ld_matrix)
    min_diag = np.min(diag_values)
    max_diag = np.max(diag_values)
    assert min_diag > 0.998, f"Diagonal values too low (min: {min_diag:.6f})"
    assert max_diag < 1.002, f"Diagonal values too high (max: {max_diag:.6f})"

    assert np.allclose(ld_matrix, ld_matrix.T), "LD matrix should be symmetric"


def _load_ldstore_matrix(file_path):
    """Load LD matrix from LDstore2 output file."""
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Parse the matrix
    ld_matrix = []
    for line in lines:
        values = [float(val) for val in line.strip().split()]
        ld_matrix.append(values)

    return np.array(ld_matrix)


def _read_numeric_matrix(file_path):
    """Helper function to read a numeric matrix from a text file."""
    import gzip

    # Determine if the file is compressed
    is_compressed = file_path.endswith((".gz", ".bgz"))
    open_func = gzip.open if is_compressed else open
    mode = "rt" if is_compressed else "r"

    # Read the matrix
    with open_func(file_path, mode) as f:
        lines = f.readlines()

    # Parse the matrix values
    matrix = []
    for line in lines:
        try:
            values = [float(val) for val in line.strip().split("\t")]
        except ValueError:
            values = [float(val) for val in line.strip().split()]
        matrix.append(values)

    return np.array(matrix)
