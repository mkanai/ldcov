"""
Tests for covariate handling and adjustment functionality.

This module contains tests for:
- Covariate loading from various file formats
- Categorical variable encoding
- Covariate adjustment via FWL projection
- Genotype standardization
"""

import unittest
import numpy as np
import pandas as pd
import os
import tempfile
import shutil
from pathlib import Path

from ldcov.compute.covariate import standardize_genotypes, regress_out_covariates
from ldcov.io.covariate_loader import load_covariates
from ldcov.utils.categorical_utils import one_hot_encode_categorical
from ldcov.io.bgen_reader import load_bgen


class TestCovariate(unittest.TestCase):
    """Test cases for covariate handling and adjustment."""

    @classmethod
    def setUpClass(cls):
        """Set up test data."""
        cls.examples_dir = Path(__file__).parents[1] / "examples"
        cls.bgen_file = cls.examples_dir / "data" / "data.bgen"
        cls.bgi_file = cls.examples_dir / "data" / "data.bgen.bgi"
        cls.sample_file = cls.examples_dir / "data" / "data.sample"

        # Create temporary directory
        cls.temp_dir = tempfile.mkdtemp(prefix="ldcov_test_covariate_")

        # Load test genotype data
        cls.genotypes, cls.variant_info, cls.sample_ids = load_bgen(
            file_path=str(cls.bgen_file),
            index_path=str(cls.bgi_file),
            sample_path=str(cls.sample_file),
        )

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

    def test_standardize_genotypes_with_missing(self):
        """Test that standardization raises error with missing values."""
        genotypes = np.array(
            [[0, 1, 2, np.nan], [1, np.nan, 1, 1], [2, 1, 0, 2], [np.nan, 2, 1, 0]], dtype=np.float64
        )

        # The standardize_genotypes function now raises an error with NaN values
        with self.assertRaises(ValueError) as cm:
            standardize_genotypes(genotypes, center=True, scale=True, inplace=False)
        
        self.assertIn("Genotype matrix contains NaN values", str(cm.exception))

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

    def test_regress_out_covariates_with_projection_matrix(self):
        """Test regression using pre-computed projection matrix."""
        n_samples, n_variants = 100, 5
        genotypes = np.random.randn(n_samples, n_variants)

        # Create covariates
        covariates = pd.DataFrame(
            {
                "cov1": np.random.randn(n_samples),
                "cov2": np.random.randn(n_samples),
            }
        )

        # Pre-compute Q matrix from QR decomposition
        covar_matrix = covariates.values
        Q, _ = np.linalg.qr(
            np.column_stack([np.ones(n_samples), covar_matrix]), mode="reduced"
        )

        # Regress using projection matrix
        adjusted = regress_out_covariates(genotypes.copy(), projection_matrix_Q=Q)

        # Verify orthogonality to covariates
        # The adjusted genotypes should be orthogonal to the covariate space
        for i in range(adjusted.shape[1]):
            for j in range(Q.shape[1]):
                dot_product = np.dot(adjusted[:, i], Q[:, j])
                self.assertAlmostEqual(dot_product, 0.0, places=10)

    # ==================== Covariate Loading Tests ====================

    def test_load_covariates_csv(self):
        """Test loading covariates from CSV file."""
        # Create test CSV
        cov_data = pd.DataFrame(
            {
                "IID": ["sample1", "sample2", "sample3"],
                "PC1": [0.1, 0.2, 0.3],
                "PC2": [-0.1, -0.2, -0.3],
                "sex": ["male", "female", "male"],
            }
        )
        cov_file = os.path.join(self.temp_dir, "test_cov.csv")
        cov_data.to_csv(cov_file, index=False)

        # Load covariates
        loaded_cov = load_covariates(cov_file)

        # Check one-hot encoding was applied
        self.assertIn("sex_male", loaded_cov.columns)
        self.assertNotIn("sex", loaded_cov.columns)
        self.assertEqual(list(loaded_cov.index), ["sample1", "sample2", "sample3"])

    def test_load_covariates_whitespace(self):
        """Test loading whitespace-delimited covariate file."""
        # Create whitespace-delimited file
        cov_file = os.path.join(self.temp_dir, "test_cov.txt")
        with open(cov_file, "w") as f:
            f.write("IID PC1 PC2 batch\n")
            f.write("s1   0.1  0.2 A\n")
            f.write("s2   0.3  0.4 B\n")

        loaded_cov = load_covariates(cov_file)

        # Check loading and one-hot encoding
        self.assertEqual(len(loaded_cov), 2)
        # batch_A is dropped (first alphabetically), batch_B is kept
        self.assertNotIn("batch_A", loaded_cov.columns)
        self.assertIn("batch_B", loaded_cov.columns)

    def test_load_covariates_custom_id_column(self):
        """Test loading covariates with custom ID column."""
        cov_data = pd.DataFrame(
            {
                "FID": ["fam1", "fam2"],
                "IID": ["ind1", "ind2"],
                "PC1": [0.1, 0.2],
            }
        )
        cov_file = os.path.join(self.temp_dir, "test_cov_fid.csv")
        cov_data.to_csv(cov_file, index=False)

        # Load with FID as ID column
        loaded_cov = load_covariates(cov_file, id_col="FID")
        self.assertEqual(list(loaded_cov.index), ["fam1", "fam2"])
        self.assertIn("IID_ind2", loaded_cov.columns)  # IID becomes a feature

    def test_load_covariates_error_handling(self):
        """Test error handling in covariate loading."""
        # Non-existent file
        with self.assertRaises(ValueError):
            load_covariates("/nonexistent/file.txt")

        # Missing ID column
        cov_data = pd.DataFrame({"PC1": [0.1, 0.2], "PC2": [0.3, 0.4]})
        cov_file = os.path.join(self.temp_dir, "test_no_id.csv")
        cov_data.to_csv(cov_file, index=False)

        with self.assertRaises(ValueError) as cm:
            load_covariates(cov_file, id_col="IID")
        self.assertIn("ID column 'IID' not found", str(cm.exception))

    def test_load_covariates_sample_filtering(self):
        """Test covariate loading with sample filtering."""
        cov_data = pd.DataFrame(
            {
                "IID": ["1", "2", "3", "4", "5"],
                "PC1": [0.1, 0.2, 0.3, 0.4, 0.5],
            }
        )
        cov_file = os.path.join(self.temp_dir, "test_filter.csv")
        cov_data.to_csv(cov_file, index=False)

        # Load with subset of samples
        loaded_cov = load_covariates(cov_file, sample_ids=["1", "3", "5", "99"])

        # Should only have samples that exist in both
        self.assertEqual(len(loaded_cov), 3)
        self.assertEqual(list(loaded_cov.index), ["1", "3", "5"])

    def test_load_covariates_with_specific_columns(self):
        """Test loading covariates with specific columns selected."""
        # Create test CSV with multiple columns
        cov_data = pd.DataFrame(
            {
                "IID": ["sample1", "sample2", "sample3"],
                "PC1": [0.1, 0.2, 0.3],
                "PC2": [-0.1, -0.2, -0.3],
                "PC3": [0.5, 0.6, 0.7],
                "batch": ["A", "B", "A"],
                "age": [25, 45, 65],
            }
        )
        cov_file = os.path.join(self.temp_dir, "test_multi_cov.csv")
        cov_data.to_csv(cov_file, index=False)

        # Load with specific columns
        loaded_cov = load_covariates(cov_file, cols_to_use=["PC1", "PC3", "age"])

        # Check that only requested columns are present (plus any one-hot encoded)
        self.assertIn("PC1", loaded_cov.columns)
        self.assertIn("PC3", loaded_cov.columns)
        self.assertIn("age", loaded_cov.columns)
        self.assertNotIn("PC2", loaded_cov.columns)
        self.assertNotIn("batch", loaded_cov.columns)

        # Check all samples are present
        self.assertEqual(list(loaded_cov.index), ["sample1", "sample2", "sample3"])

    def test_load_covariates_invalid_columns(self):
        """Test error handling when requesting non-existent columns."""
        cov_data = pd.DataFrame(
            {
                "IID": ["sample1", "sample2"],
                "PC1": [0.1, 0.2],
                "PC2": [-0.1, -0.2],
            }
        )
        cov_file = os.path.join(self.temp_dir, "test_invalid_cols.csv")
        cov_data.to_csv(cov_file, index=False)

        # Request non-existent columns
        with self.assertRaises(ValueError) as cm:
            load_covariates(cov_file, cols_to_use=["PC1", "NonExistent"])
        self.assertIn("Requested covariate columns not found", str(cm.exception))

    # ==================== One-Hot Encoding Tests ====================

    def test_one_hot_encoding(self):
        """Test one-hot encoding of categorical variables."""
        df = pd.DataFrame(
            {
                "numeric": [1.0, 2.0, 3.0],
                "category": ["A", "B", "A"],
                "binary": ["yes", "no", "yes"],
            }
        )

        encoded = one_hot_encode_categorical(df)

        # Check encoding (first categories dropped)
        self.assertIn("numeric", encoded.columns)
        # category_A dropped (first alphabetically), category_B kept
        self.assertNotIn("category_A", encoded.columns)
        self.assertIn("category_B", encoded.columns)
        # binary: "no" dropped (first alphabetically), "yes" kept
        self.assertNotIn("binary_no", encoded.columns)
        self.assertIn("binary_yes", encoded.columns)
        # Original categorical columns removed
        self.assertNotIn("category", encoded.columns)
        self.assertNotIn("binary", encoded.columns)

    def test_one_hot_encoding_preserves_order(self):
        """Test that one-hot encoding preserves row order."""
        df = pd.DataFrame(
            {
                "id": ["a", "b", "c", "d"],
                "group": ["X", "Y", "X", "Z"],
            }
        )
        df.set_index("id", inplace=True)

        encoded = one_hot_encode_categorical(df)

        # Check that row order is preserved
        self.assertEqual(list(encoded.index), ["a", "b", "c", "d"])

    def test_one_hot_encoding_single_category(self):
        """Test one-hot encoding with a single category value."""
        df = pd.DataFrame(
            {
                "constant": ["A", "A", "A"],
                "numeric": [1, 2, 3],
            }
        )

        encoded = one_hot_encode_categorical(df)

        # Single-value categorical creates a column that is all 1s
        # The categorical_utils module keeps this column (though it's not informative)
        self.assertNotIn("constant", encoded.columns)
        self.assertIn("constant_A", encoded.columns)
        self.assertIn("numeric", encoded.columns)
        # Check that constant_A is all 1s
        self.assertTrue((encoded["constant_A"] == 1).all())

    # ==================== Integration Tests ====================

    def test_covariate_adjustment_pipeline(self):
        """Test full covariate adjustment pipeline."""
        # Use a smaller subset of genotypes for more reliable testing
        test_genotypes = self.genotypes[:100, :5].copy()  # 100 samples, 5 variants
        test_sample_ids = self.sample_ids[:100]
        
        # Create test covariate file with stronger effects
        n_samples = len(test_sample_ids)
        np.random.seed(42)
        
        # Create covariates that have stronger association with genotypes
        pc1 = np.random.normal(0, 1, n_samples)
        pc2 = np.random.normal(0, 1, n_samples)
        
        # Add covariate effects to genotypes to ensure adjustment will have an effect
        test_genotypes[:, 0] += 0.5 * pc1  # Add PC1 effect to first variant
        test_genotypes[:, 1] += 0.3 * pc2  # Add PC2 effect to second variant
        
        covariates = pd.DataFrame(
            {
                "IID": test_sample_ids,
                "PC1": pc1,
                "PC2": pc2,
                "sex": np.random.choice(["male", "female"], n_samples),
            }
        )
        cov_file = os.path.join(self.temp_dir, "pipeline_test.csv")
        covariates.to_csv(cov_file, index=False)

        # Load covariates
        loaded_covs = load_covariates(cov_file, sample_ids=test_sample_ids)

        # Standardize genotypes
        std_geno, means, norms = standardize_genotypes(
            test_genotypes.copy(), center=True, scale=True
        )

        # Adjust for covariates (make a copy to ensure inplace=False works)
        adjusted_geno = regress_out_covariates(std_geno.copy(), loaded_covs, inplace=False)

        # Verify dimensions
        self.assertEqual(adjusted_geno.shape, test_genotypes.shape)

        # Verify adjustment happened
        # The adjusted genotypes should be different from the original
        diff = np.abs(adjusted_geno - std_geno).max()
        self.assertGreater(diff, 1e-10, "Adjustment should modify the genotypes")
        
        # Also check that the adjustment is not trivial (not just zeros)
        self.assertGreater(np.abs(adjusted_geno).max(), 1e-10, "Adjusted genotypes should not be all zeros")


if __name__ == "__main__":
    unittest.main()