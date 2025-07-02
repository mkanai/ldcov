"""
Test for comparing ldcov LD calculations with LDstore2 output.

This test compares the LD calculations from ldcov with those from LDstore2,
using example data files provided with the package.
"""

import numpy as np
import pandas as pd
import pytest
import os
from pathlib import Path

import ldcov


@pytest.fixture(scope="module")
def test_data():
    """Set up test data once for all test methods."""
    # Get paths to example data files
    examples_dir = Path(__file__).parents[1] / "examples"
    bgen_file = examples_dir / "data" / "data.bgen"
    bgi_file = examples_dir / "data" / "data.bgen.bgi"
    ldstore_file = examples_dir / "data" / "data.ld"
    
    # Load genotype data
    genotypes, variant_info, sample_ids = ldcov.load_bgen(
        file_path=str(bgen_file), index_path=str(bgi_file)
    )
    
    # First standardize the genotypes
    from ldcov.compute.covariate import standardize_genotypes
    
    standardized_genotypes, _, _ = standardize_genotypes(genotypes, center=True, scale=True)
    
    # Compute LD using ldcov
    ldcov_ld = ldcov.compute_correlation_matrix(standardized_genotypes)
    
    # Load LDstore2 output
    ldstore_ld = _load_ldstore_matrix(ldstore_file)
    
    return {
        "genotypes": genotypes,
        "variant_info": variant_info,
        "sample_ids": sample_ids,
        "ldcov_ld": ldcov_ld,
        "ldstore_ld": ldstore_ld
    }


def _load_ldstore_matrix(file_path):
    """
    Load LD matrix from LDstore2 output file.
    
    Parameters:
    -----------
    file_path : str or Path
        Path to LDstore2 output file
    
    Returns:
    --------
    numpy.ndarray
        LD matrix from LDstore2
    """
    # Read LDstore2 output, which is a plain text matrix of correlation values
    with open(file_path, "r") as f:
        lines = f.readlines()
    
    # Parse the matrix
    ld_matrix = []
    for line in lines:
        # Split line by whitespace and convert values to float
        values = [float(val) for val in line.strip().split()]
        ld_matrix.append(values)
    
    # Convert to numpy array
    return np.array(ld_matrix)


def test_ld_matrix_shape(test_data):
    """Test that the LD matrix has the expected shape."""
    expected_shape = (len(test_data["variant_info"]), len(test_data["variant_info"]))
    assert test_data["ldcov_ld"].shape == expected_shape


def test_ld_matrix_values(test_data):
    """Test that the LD values are similar to LDstore2 output."""
    # Check if the shapes match
    if test_data["ldcov_ld"].shape == test_data["ldstore_ld"].shape:
        # Use a tolerance for floating-point comparison
        tol = 0.05  # 5% difference allowed
        differences = np.abs(test_data["ldcov_ld"] - test_data["ldstore_ld"])
        max_diff = np.max(differences)
        avg_diff = np.mean(differences)
        
        # Check overall similarity
        assert avg_diff < tol, f"Average difference {avg_diff:.4f} exceeds tolerance {tol}"
        
        # Print max difference for information
        print(f"Maximum difference between ldcov and LDstore2: {max_diff:.4f}")
        print(f"Average difference between ldcov and LDstore2: {avg_diff:.4f}")
    else:
        # If shapes don't match, we need to compare a subset
        # This is more complex and would need to match variants by position
        # For simplicity, we just check the shapes in this case
        assert test_data["ldcov_ld"].shape == test_data["ldstore_ld"].shape, \
            "LD matrices have different shapes. Cannot directly compare values."


def test_ld_matrix_diagonal(test_data):
    """Test that the diagonal of the LD matrix is 1."""
    diag = np.diag(test_data["ldcov_ld"])
    assert np.allclose(diag, 1.0), "Diagonal values should be 1.0"


def test_ld_matrix_symmetry(test_data):
    """Test that the LD matrix is symmetric."""
    assert np.allclose(test_data["ldcov_ld"], test_data["ldcov_ld"].T), \
        "LD matrix should be symmetric"