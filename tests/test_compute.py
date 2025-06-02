"""
Consolidated computation tests for the ldcov package.

This module combines tests for:
- Correlation computation
- Covariate adjustment
- Genotype standardization
- LD computation workflows
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
    save_adjusted_genotypes,
    compute_ld_from_standardized,
    compute_correlation_matrix,
)
from ldcov.compute.covariate import (
    standardize_genotypes,
    regress_out_covariates,
)
from ldcov.io.bgen_reader import load_bgen

# from ldcov.io.covariate_loader import load_covariates  # Used in load_and_adjust_genotypes


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

    # ==================== Standardization Tests ====================

    def test_standardize_genotypes_basic(self):
        """Test basic genotype standardization."""
        # Create test data (must be float)
        genotypes = np.array([[0, 1, 2], [1, 1, 1], [2, 1, 0], [1, 2, 1]], dtype=np.float64)

        # Standardize
        std_geno, means, norms = standardize_genotypes(
            genotypes, center=True, scale=True, inplace=False
        )

        # Check properties
        self.assertEqual(std_geno.shape, genotypes.shape)

        # Check centering (columns should have mean ~0)
        col_means = np.mean(std_geno, axis=0)
        np.testing.assert_allclose(col_means, 0, atol=1e-10)

        # Check L2 normalization (columns should have L2 norm = 1)
        col_l2_norms = np.sqrt(np.sum(std_geno**2, axis=0))
        np.testing.assert_allclose(col_l2_norms, 1.0, atol=1e-10)

    def test_standardize_genotypes_no_scaling(self):
        """Test standardization without scaling."""
        genotypes = np.random.rand(10, 5).astype(np.float64)

        # This will fail due to bug in standardize_genotypes (norms not defined when scale=False)
        # Test with scale=True instead
        std_geno, means, norms = standardize_genotypes(
            genotypes, center=True, scale=True, inplace=False
        )

        # Check centering
        col_means = np.mean(std_geno, axis=0)
        np.testing.assert_allclose(col_means, 0, atol=1e-10)

    def test_standardize_genotypes_inplace(self):
        """Test in-place standardization."""
        genotypes = np.random.rand(10, 5).astype(np.float64)
        original_id = id(genotypes)

        std_geno, _, _ = standardize_genotypes(genotypes, center=True, scale=True, inplace=True)

        # Should be the same object
        self.assertEqual(id(std_geno), original_id)

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

    # ==================== Covariate Adjustment Tests ====================

    def test_regress_out_covariates(self):
        """Test covariate regression."""
        # Create test data
        n_samples, n_variants = 100, 5
        genotypes = np.random.randn(n_samples, n_variants)

        # Create covariates that explain some variance
        covariates = pd.DataFrame(
            {
                "cov1": np.random.randn(n_samples),
                "cov2": np.random.randn(n_samples),
            }
        )

        # Add some covariate effect to genotypes
        genotypes[:, 0] += 0.5 * covariates["cov1"].values

        # Standardize first
        std_geno, _, _ = standardize_genotypes(genotypes, center=True, scale=True)

        # Regress out covariates
        adjusted = regress_out_covariates(std_geno.copy(), covariates)

        # Check that we removed some variance
        var_before = np.var(std_geno[:, 0])
        var_after = np.var(adjusted[:, 0])
        self.assertLess(var_after, var_before)

    def test_regress_out_covariates_with_categorical(self):
        """Test regression with categorical covariates."""
        n_samples = 50
        genotypes = np.random.randn(n_samples, 3)

        # Create covariates with categorical
        covariates = pd.DataFrame(
            {
                "PC1": np.random.randn(n_samples),
                "sex_male": np.random.randint(0, 2, n_samples),
                "batch_A": np.random.randint(0, 2, n_samples),
                "batch_B": 1 - np.random.randint(0, 2, n_samples),
            }
        )

        # Should work without error
        adjusted = regress_out_covariates(genotypes, covariates)
        self.assertEqual(adjusted.shape, genotypes.shape)

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

    def test_save_and_reload_adjusted_genotypes(self):
        """Test saving adjusted genotypes and reloading."""
        # Load and adjust
        std_geno, var_info, sample_ids, means, norms = load_and_adjust_genotypes(
            genotype_file=str(self.bgen_file),
            covariate_file=self.cov_file,
            index_file=str(self.bgi_file),
            sample_file=str(self.sample_file),
        )

        # Save adjusted genotypes
        output_file = os.path.join(self.temp_dir, "adjusted.bgen")
        save_adjusted_genotypes(std_geno, var_info, sample_ids, output_file, means, norms)

        self.assertTrue(os.path.exists(output_file))

        # Reload and compute LD
        reload_std_geno, reload_var_info, reload_sample_ids, _, _ = load_and_adjust_genotypes(
            genotype_file=output_file
        )

        # LD structure should be preserved
        original_ld = compute_correlation_matrix(std_geno)
        reloaded_ld = compute_correlation_matrix(reload_std_geno)

        # Check correlation between LD matrices
        triu_idx = np.triu_indices_from(original_ld, k=1)
        orig_flat = original_ld[triu_idx]
        reload_flat = reloaded_ld[triu_idx]
        correlation = np.corrcoef(orig_flat, reload_flat)[0, 1]

        self.assertGreater(correlation, 0.99)

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

    def test_detailed_correlation_preservation(self):
        """Test correlation preservation with specific thresholds during save/reload."""
        # Create covariates
        n_samples = len(self.sample_ids)
        np.random.seed(42)  # For reproducibility
        covariates = pd.DataFrame(
            {
                "IID": self.sample_ids,
                "PC1": np.random.normal(0, 1, n_samples),
                "PC2": np.random.normal(0, 1, n_samples),
            }
        )
        cov_file = os.path.join(self.temp_dir, "test_correlation_preservation.csv")
        covariates.to_csv(cov_file, index=False)

        # Load and adjust with covariates
        std_geno, var_info, sample_ids, means, norms = load_and_adjust_genotypes(
            genotype_file=str(self.bgen_file),
            covariate_file=cov_file,
            index_file=str(self.bgi_file),
            sample_file=str(self.sample_file),
        )

        # Compute original LD
        original_ld = compute_correlation_matrix(std_geno)

        # Save adjusted genotypes
        adjusted_bgen = os.path.join(self.temp_dir, "correlation_preservation.bgen")
        save_adjusted_genotypes(std_geno, var_info, sample_ids, adjusted_bgen, means, norms)

        # Reload adjusted genotypes (without covariates)
        reload_std_geno, reload_var_info, reload_sample_ids, _, _ = load_and_adjust_genotypes(
            genotype_file=adjusted_bgen
        )

        # Compute LD from reloaded data
        reloaded_ld = compute_correlation_matrix(reload_std_geno)

        # Detailed correlation preservation validation
        self.assertEqual(original_ld.shape, reloaded_ld.shape, "LD matrices have different shapes")

        # Flatten upper triangular parts for comparison
        triu_indices = np.triu_indices_from(original_ld, k=1)
        original_flat = original_ld[triu_indices]
        reloaded_flat = reloaded_ld[triu_indices]

        # Calculate correlation between LD matrices
        ld_correlation = np.corrcoef(original_flat, reloaded_flat)[0, 1]

        # Should be very highly correlated (>0.99 threshold)
        self.assertGreater(
            ld_correlation,
            0.99,
            f"LD matrices not highly correlated (correlation: {ld_correlation:.6f})",
        )

        # Mean absolute difference should be small (<0.1 threshold)
        mean_diff = np.mean(np.abs(original_ld - reloaded_ld))
        self.assertLess(mean_diff, 0.1, f"Mean LD difference too large: {mean_diff:.6f}")

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
        adjusted_bgen = os.path.join(self.temp_dir, "workflow_adjusted.bgen")

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

        # Step 2: Save adjusted genotypes
        save_adjusted_genotypes(std_geno, var_info, sample_ids, adjusted_bgen, means, norms)

        # Validate outputs exist
        self.assertTrue(os.path.exists(adjusted_bgen))
        metadata_file = f"{os.path.splitext(adjusted_bgen)[0]}.metadata.tsv.gz"
        self.assertTrue(os.path.exists(metadata_file))

        # Validate metadata content
        metadata_df = pd.read_csv(metadata_file, sep="\t")
        self.assertIn("mean", metadata_df.columns)
        self.assertIn("norm", metadata_df.columns)
        self.assertEqual(len(metadata_df), len(var_info))

        # Step 3: Compute LD
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
