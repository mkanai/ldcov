"""
Test for comparing ldcov LD calculations with LDstore2 output.

This test compares the LD calculations from ldcov with those from LDstore2,
using example data files provided with the package.
"""

import unittest
import numpy as np
import pandas as pd
import os
from pathlib import Path

import ldcov


class TestLDstoreComparison(unittest.TestCase):
    """
    Test case for comparing ldcov LD calculations with LDstore2 output.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up test data once for all test methods.
        """
        # Get paths to example data files
        cls.examples_dir = Path(__file__).parents[1] / "examples"
        cls.bgen_file = cls.examples_dir / "data" / "data.bgen"
        cls.bgi_file = cls.examples_dir / "data" / "data.bgen.bgi"
        cls.ldstore_file = cls.examples_dir / "data" / "data.ld"

        # Load genotype data
        cls.genotypes, cls.variant_info, cls.sample_ids = ldcov.load_bgen(
            file_path=str(cls.bgen_file), index_path=str(cls.bgi_file)
        )

        # First standardize the genotypes
        from ldcov.compute.covariate import standardize_genotypes

        standardized_genotypes, _, _ = standardize_genotypes(cls.genotypes, center=True, scale=True)

        # Compute LD using ldcov
        cls.ldcov_ld = ldcov.compute_correlation_matrix(standardized_genotypes)

        # Load LDstore2 output
        cls.ldstore_ld = cls._load_ldstore_matrix(cls.ldstore_file)

    @staticmethod
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

    def test_ld_matrix_shape(self):
        """
        Test that the LD matrix has the expected shape.
        """
        expected_shape = (len(self.variant_info), len(self.variant_info))
        self.assertEqual(self.ldcov_ld.shape, expected_shape)

    def test_ld_matrix_values(self):
        """
        Test that the LD values are similar to LDstore2 output.
        """
        # Check if the shapes match
        if self.ldcov_ld.shape == self.ldstore_ld.shape:
            # Use a tolerance for floating-point comparison
            tol = 0.05  # 5% difference allowed
            differences = np.abs(self.ldcov_ld - self.ldstore_ld)
            max_diff = np.max(differences)
            avg_diff = np.mean(differences)

            # Check overall similarity
            self.assertLess(
                avg_diff, tol, f"Average difference {avg_diff:.4f} exceeds tolerance {tol}"
            )

            # Print max difference for information
            print(f"Maximum difference between ldcov and LDstore2: {max_diff:.4f}")
            print(f"Average difference between ldcov and LDstore2: {avg_diff:.4f}")
        else:
            # If shapes don't match, we need to compare a subset
            # This is more complex and would need to match variants by position
            # For simplicity, we just check the shapes in this case
            self.assertEqual(
                self.ldcov_ld.shape,
                self.ldstore_ld.shape,
                "LD matrices have different shapes. Cannot directly compare values.",
            )

    def test_ld_matrix_diagonal(self):
        """
        Test that the diagonal of the LD matrix is 1.
        """
        diag = np.diag(self.ldcov_ld)
        self.assertTrue(np.allclose(diag, 1.0), "Diagonal values should be 1.0")

    def test_ld_matrix_symmetry(self):
        """
        Test that the LD matrix is symmetric.
        """
        self.assertTrue(
            np.allclose(self.ldcov_ld, self.ldcov_ld.T), "LD matrix should be symmetric"
        )


if __name__ == "__main__":
    unittest.main()
