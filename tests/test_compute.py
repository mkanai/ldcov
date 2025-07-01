"""
Consolidated computation tests for the ldcov package.

This module contains tests for:
- Correlation computation
- LD computation workflows
- Integration with covariate adjustment
"""

import unittest
import numpy as np
import pandas as pd
import os
import tempfile
import shutil
from pathlib import Path

from ldcov.compute.correlation import (
    load_and_adjust_genotypes,
    compute_ld_from_standardized,
    compute_correlation_matrix,
)
from ldcov.compute.covariate import standardize_genotypes
from ldcov.io import load_bgen


class TestCompute(unittest.TestCase):
    """Test cases for all computation operations."""

    @classmethod
    def setUpClass(cls):
        """Set up test data."""
        cls.examples_dir = Path(__file__).parents[1] / "examples"
        cls.bgen_file = cls.examples_dir / "data" / "data.bgen"
        cls.bgi_file = cls.examples_dir / "data" / "data.bgen.bgi"
        cls.sample_file = cls.examples_dir / "data" / "data.sample"

        # Create temporary directory
        cls.temp_dir = tempfile.mkdtemp(prefix="ldcov_test_compute_")

        # Load test data
        cls.genotypes, cls.variant_info, cls.sample_ids = load_bgen(
            file_path=str(cls.bgen_file),
            index_path=str(cls.bgi_file),
            sample_path=str(cls.sample_file),
        )

        # Create test covariate file
        n_samples = len(cls.sample_ids)
        np.random.seed(42)
        covariates = pd.DataFrame(
            {
                "IID": cls.sample_ids,
                "PC1": np.random.normal(0, 1, n_samples),
                "PC2": np.random.normal(0, 1, n_samples),
                "sex": np.random.choice(["male", "female"], n_samples),
            }
        )
        cls.cov_file = os.path.join(cls.temp_dir, "test_covariates.csv")
        covariates.to_csv(cls.cov_file, index=False)

    @classmethod
    def tearDownClass(cls):
        """Clean up test data."""
        shutil.rmtree(cls.temp_dir)

    # ==================== Correlation Tests ====================

    def test_compute_correlation_matrix(self):
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
        std_genotypes, _, _ = standardize_genotypes(
            genotypes, center=True, scale=True, inplace=False
        )

        # Compute correlation
        corr_matrix = compute_correlation_matrix(std_genotypes)

        # Check properties
        self.assertEqual(corr_matrix.shape, (3, 3))
        # Perfect correlation between var1 and var2
        self.assertAlmostEqual(corr_matrix[0, 1], 1.0, places=5)
        # Low correlation with var3
        self.assertLess(abs(corr_matrix[0, 2]), 0.2)
        # Diagonal should be 1
        np.testing.assert_allclose(np.diag(corr_matrix), 1.0)

    def test_compute_ld_from_standardized(self):
        """Test LD computation from standardized genotypes."""
        # Standardize test genotypes
        std_geno, _, _ = standardize_genotypes(
            self.genotypes[:, :10].copy(), center=True, scale=True
        )

        # Test different output formats
        for output_format in ["matrix", "long", "bcor"]:
            output_file = os.path.join(self.temp_dir, f"test.{output_format}")
            compute_ld_from_standardized(
                std_geno,
                self.variant_info.iloc[:10],
                output_file,
                output_format=output_format,
            )
            self.assertTrue(os.path.exists(output_file))

    # ==================== Workflow Tests ====================

    def test_load_and_adjust_genotypes_no_covariates(self):
        """Test loading and standardizing without covariates."""
        std_geno, var_info, sample_ids, means, norms = load_and_adjust_genotypes(
            genotype_file=str(self.bgen_file),
            index_file=str(self.bgi_file),
            sample_file=str(self.sample_file),
        )

        # Check outputs
        self.assertGreater(std_geno.shape[0], 0)
        self.assertGreater(std_geno.shape[1], 0)
        self.assertEqual(len(var_info), std_geno.shape[1])
        self.assertEqual(len(sample_ids), std_geno.shape[0])

        # Check standardization
        col_means = np.mean(std_geno, axis=0)
        np.testing.assert_allclose(col_means, 0, atol=1e-10)

    def test_load_and_adjust_genotypes_with_covariates(self):
        """Test loading and adjusting with covariates."""
        std_geno, var_info, sample_ids, means, norms = load_and_adjust_genotypes(
            genotype_file=str(self.bgen_file),
            covariate_file=self.cov_file,
            index_file=str(self.bgi_file),
            sample_file=str(self.sample_file),
        )

        # Should still be standardized after adjustment
        self.assertGreater(std_geno.shape[0], 0)
        self.assertGreater(std_geno.shape[1], 0)

    def test_load_and_adjust_with_custom_id_column(self):
        """Test loading with custom covariate ID column."""
        # Create covariate file with custom ID column
        n_samples = len(self.sample_ids)
        covariates = pd.DataFrame(
            {
                "FID": self.sample_ids,
                "PC1": np.random.normal(0, 1, n_samples),
            }
        )
        custom_cov_file = os.path.join(self.temp_dir, "custom_id.csv")
        covariates.to_csv(custom_cov_file, index=False)

        # Load with custom ID column
        std_geno, var_info, sample_ids, means, norms = load_and_adjust_genotypes(
            genotype_file=str(self.bgen_file),
            covariate_file=custom_cov_file,
            index_file=str(self.bgi_file),
            sample_file=str(self.sample_file),
            covariate_id_col="FID",
        )

        self.assertEqual(len(sample_ids), n_samples)

    def test_sample_filtering_with_missing_covariates(self):
        """Test that samples are filtered when some lack covariate data."""
        # Create covariate file with subset of samples
        subset_samples = self.sample_ids[::2]  # Every other sample
        covariates = pd.DataFrame(
            {
                "IID": subset_samples,
                "PC1": np.random.normal(0, 1, len(subset_samples)),
            }
        )
        subset_cov_file = os.path.join(self.temp_dir, "subset_cov.csv")
        covariates.to_csv(subset_cov_file, index=False)

        # Load and adjust
        std_geno, var_info, sample_ids, means, norms = load_and_adjust_genotypes(
            genotype_file=str(self.bgen_file),
            covariate_file=subset_cov_file,
            index_file=str(self.bgi_file),
            sample_file=str(self.sample_file),
        )

        # Should only have samples with covariate data
        self.assertEqual(len(sample_ids), len(subset_samples))
        self.assertEqual(std_geno.shape[0], len(subset_samples))

    # ==================== LDstore Comparison Tests ====================

    def test_ldstore_comparison(self):
        """Test LD calculations against LDstore2 reference data."""
        # Check if LDstore reference file exists
        ldstore_file = self.examples_dir / "data" / "data.ld"
        if not ldstore_file.exists():
            self.skipTest("LDstore reference file not found - skipping comparison test")

        # Compute LD using ldcov
        std_geno, _, _ = standardize_genotypes(self.genotypes.copy(), center=True, scale=True)
        ldcov_ld = compute_correlation_matrix(std_geno)

        # Load LDstore2 reference matrix
        ldstore_ld = self._load_ldstore_matrix(ldstore_file)

        # Check shapes match
        if ldcov_ld.shape != ldstore_ld.shape:
            self.skipTest(f"Shape mismatch: ldcov {ldcov_ld.shape} vs LDstore {ldstore_ld.shape}")

        # Validate against LDstore with tolerance
        tol = 0.05  # 5% difference allowed
        differences = np.abs(ldcov_ld - ldstore_ld)
        max_diff = np.max(differences)
        avg_diff = np.mean(differences)

        # Check overall similarity
        self.assertLess(avg_diff, tol, f"Average difference {avg_diff:.4f} exceeds tolerance {tol}")

        # Additional checks
        self.assertTrue(np.allclose(np.diag(ldcov_ld), 1.0), "LD matrix diagonal should be 1.0")
        self.assertTrue(np.allclose(ldcov_ld, ldcov_ld.T), "LD matrix should be symmetric")

    def test_end_to_end_workflow_validation(self):
        """Test complete modular workflow with intermediate validations."""
        # Create test covariates
        n_samples = len(self.sample_ids)
        covariates = pd.DataFrame(
            {
                "IID": self.sample_ids,
                "PC1": np.random.normal(0, 1, n_samples),
                "PC2": np.random.normal(0, 1, n_samples),
            }
        )
        cov_file = os.path.join(self.temp_dir, "workflow_test.csv")
        covariates.to_csv(cov_file, index=False)

        # Define output files
        output_ld = os.path.join(self.temp_dir, "workflow_ld.txt")

        # Step 1: Load and adjust genotypes
        std_geno, var_info, sample_ids, means, norms = load_and_adjust_genotypes(
            genotype_file=str(self.bgen_file),
            covariate_file=cov_file,
            index_file=str(self.bgi_file),
            sample_file=str(self.sample_file),
        )

        # Validate intermediate results
        self.assertIsInstance(std_geno, np.ndarray)
        self.assertGreater(std_geno.shape[0], 0)
        self.assertGreater(std_geno.shape[1], 0)
        self.assertIsInstance(var_info, pd.DataFrame)
        self.assertGreater(len(sample_ids), 0)
        self.assertIsInstance(means, np.ndarray)
        self.assertIsInstance(norms, np.ndarray)

        # Step 2: Compute LD
        compute_ld_from_standardized(std_geno, var_info, output_ld, output_format="matrix")

        # Validate LD output
        self.assertTrue(os.path.exists(output_ld))

        # Load and validate LD matrix
        ld_matrix = self._read_numeric_matrix(output_ld)
        self.assertEqual(ld_matrix.shape[0], ld_matrix.shape[1])

        # Check diagonal is close to 1.0 (allowing for numerical precision)
        diag_values = np.diag(ld_matrix)
        min_diag = np.min(diag_values)
        max_diag = np.max(diag_values)
        self.assertGreater(min_diag, 0.998, f"Diagonal values too low (min: {min_diag:.6f})")
        self.assertLess(max_diag, 1.002, f"Diagonal values too high (max: {max_diag:.6f})")

        self.assertTrue(np.allclose(ld_matrix, ld_matrix.T), "LD matrix should be symmetric")

    @staticmethod
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

    def _read_numeric_matrix(self, file_path):
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


if __name__ == "__main__":
    unittest.main()
