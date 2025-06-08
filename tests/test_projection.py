"""
Tests for projection matrix computation and I/O functionality.
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
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


class TestProjection(unittest.TestCase):
    """Test cases for projection matrix functionality."""

    def setUp(self):
        """Set up test data."""
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp(prefix="ldcov_test_projection_")

        # Create test samples and covariates
        np.random.seed(42)
        self.n_samples = 100
        self.sample_ids = [f"sample_{i}" for i in range(self.n_samples)]

        # Create test covariate data
        self.covariates_df = pd.DataFrame(
            {
                "IID": self.sample_ids,
                "PC1": np.random.normal(0, 1, self.n_samples),
                "PC2": np.random.normal(0, 1, self.n_samples),
                "PC3": np.random.normal(0, 1, self.n_samples),
                "sex": np.random.choice(["male", "female"], self.n_samples),
                "age": np.random.normal(50, 10, self.n_samples),
            }
        )

        # Save covariates to file
        self.cov_file = os.path.join(self.temp_dir, "test_covariates.csv")
        self.covariates_df.to_csv(self.cov_file, index=False)

        # Create test genotypes
        self.n_variants = 50
        self.genotypes = np.random.binomial(2, 0.3, size=(self.n_samples, self.n_variants)).astype(
            np.float64
        )

    def tearDown(self):
        """Clean up test data."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_compute_projection_matrix_basic(self):
        """Test basic projection matrix computation."""
        projection_data = compute_projection_matrix(
            covariate_file=self.cov_file,
            sample_ids=self.sample_ids,
        )

        # Check output structure
        self.assertIsInstance(projection_data, ProjectionData)
        self.assertEqual(len(projection_data.sample_ids), self.n_samples)

        # Check Q matrix properties
        Q = projection_data.Q
        self.assertEqual(Q.shape[0], self.n_samples)  # n_samples rows
        # Q should have n_covariates columns (PC1, PC2, PC3, sex_male, age, intercept)
        expected_n_covs = 6  # 3 PCs + 1 sex (one-hot becomes 1 col) + age + intercept
        self.assertEqual(Q.shape[1], expected_n_covs)

        # Check Q is orthogonal
        QtQ = Q.T @ Q
        np.testing.assert_allclose(QtQ, np.eye(Q.shape[1]), atol=1e-10)

        # Check metadata
        self.assertEqual(projection_data.n_covariates, expected_n_covs)
        self.assertIn("intercept", projection_data.covariate_names)

    def test_compute_projection_matrix_subset_samples(self):
        """Test projection matrix computation with sample subset."""
        subset_samples = self.sample_ids[:50]

        projection_data = compute_projection_matrix(
            covariate_file=self.cov_file,
            sample_ids=subset_samples,
        )

        self.assertEqual(len(projection_data.sample_ids), 50)
        self.assertEqual(projection_data.Q.shape[0], 50)

    def test_compute_projection_matrix_specific_covariates(self):
        """Test projection matrix computation with specific covariates."""
        projection_data = compute_projection_matrix(
            covariate_file=self.cov_file,
            sample_ids=self.sample_ids,
            covariate_cols=["PC1", "PC2"],  # Only use PC1 and PC2
        )

        # Should have 3 columns: PC1, PC2, intercept
        self.assertEqual(projection_data.Q.shape[1], 3)
        self.assertEqual(projection_data.n_covariates, 3)
        self.assertIn("PC1", projection_data.covariate_names)
        self.assertIn("PC2", projection_data.covariate_names)
        self.assertNotIn("PC3", projection_data.covariate_names)

    def test_save_load_projection_matrix(self):
        """Test saving and loading projection matrix."""
        # Compute projection
        projection_data = compute_projection_matrix(
            covariate_file=self.cov_file,
            sample_ids=self.sample_ids,
        )

        # Save to file
        output_file = os.path.join(self.temp_dir, "test_projection.proj.npz")
        save_projection_matrix(projection_data, output_file)

        # Check file exists
        self.assertTrue(os.path.exists(output_file))

        # Load from file
        loaded_data = load_projection_matrix(output_file)

        # Check loaded data matches original
        np.testing.assert_array_equal(loaded_data.Q, projection_data.Q)
        self.assertEqual(loaded_data.sample_ids, projection_data.sample_ids)
        self.assertEqual(loaded_data.n_covariates, projection_data.n_covariates)
        self.assertEqual(loaded_data.covariate_names, projection_data.covariate_names)

        # Check metadata
        self.assertIn("format_version", loaded_data.metadata)
        self.assertIn("creation_date", loaded_data.metadata)

    def test_validate_projection_compatibility_exact_match(self):
        """Test projection compatibility validation with exact sample match."""
        projection_data = compute_projection_matrix(
            covariate_file=self.cov_file,
            sample_ids=self.sample_ids,
        )

        # Same samples
        Q_subset, indices = validate_projection_compatibility(projection_data, self.sample_ids)

        # Should return full Q matrix
        np.testing.assert_array_equal(Q_subset, projection_data.Q)
        self.assertEqual(indices, list(range(self.n_samples)))

    def test_validate_projection_compatibility_subset(self):
        """Test projection compatibility validation with genotype sample subset."""
        projection_data = compute_projection_matrix(
            covariate_file=self.cov_file,
            sample_ids=self.sample_ids,
        )

        # Subset of samples in different order
        geno_samples = self.sample_ids[25:75][::-1]  # Reverse order

        Q_subset, indices = validate_projection_compatibility(projection_data, geno_samples)

        # Check dimensions
        self.assertEqual(Q_subset.shape[0], 50)
        self.assertEqual(Q_subset.shape[1], projection_data.Q.shape[1])

        # Check indices are correct
        self.assertEqual(len(indices), 50)
        # First genotype sample (sample_74) should map to index 74 in projection
        self.assertEqual(indices[0], 74)

    def test_validate_projection_compatibility_missing_samples(self):
        """Test projection compatibility validation with missing samples."""
        projection_data = compute_projection_matrix(
            covariate_file=self.cov_file,
            sample_ids=self.sample_ids[:50],  # Only first 50 samples
        )

        # Try to use with all samples (including ones not in projection)
        with self.assertRaises(ValueError) as cm:
            validate_projection_compatibility(projection_data, self.sample_ids)

        self.assertIn("not present in projection matrix", str(cm.exception))

    def test_regression_with_precomputed_projection(self):
        """Test that pre-computed projection gives same results as direct computation."""
        from ldcov.compute.covariate import standardize_genotypes

        # Standardize genotypes
        std_geno1 = self.genotypes.copy()
        std_geno1, _, _ = standardize_genotypes(std_geno1, inplace=True)

        std_geno2 = self.genotypes.copy()
        std_geno2, _, _ = standardize_genotypes(std_geno2, inplace=True)

        # Method 1: Direct covariate regression
        covariates = self.covariates_df.set_index("IID")
        adjusted1 = regress_out_covariates(
            std_geno1, covariates=covariates[["PC1", "PC2", "PC3"]], inplace=True
        )

        # Method 2: Pre-computed projection
        projection_data = compute_projection_matrix(
            covariate_file=self.cov_file,
            sample_ids=self.sample_ids,
            covariate_cols=["PC1", "PC2", "PC3"],
        )

        adjusted2 = regress_out_covariates(
            std_geno2, projection_matrix_Q=projection_data.Q, inplace=True
        )

        # Results should be identical
        np.testing.assert_allclose(adjusted1, adjusted2, atol=1e-10)

    def test_projection_with_rank_deficient_covariates(self):
        """Test projection matrix computation with rank deficient covariates."""
        # Create rank deficient covariates (PC3 = PC1 + PC2)
        df = self.covariates_df.copy()
        df["PC3"] = df["PC1"] + df["PC2"]
        df["PC4"] = df["PC1"] * 2  # Another linear dependency

        # Save to file
        rank_def_file = os.path.join(self.temp_dir, "rank_def_covariates.csv")
        df.to_csv(rank_def_file, index=False)

        # Should handle rank deficiency gracefully
        projection_data = compute_projection_matrix(
            covariate_file=rank_def_file,
            sample_ids=self.sample_ids,
            covariate_cols=["PC1", "PC2", "PC3", "PC4"],
        )

        # Should have fewer columns than requested due to rank deficiency
        self.assertLess(projection_data.Q.shape[1], 5)  # Less than 4 PCs + intercept

        # Q should still be orthogonal
        QtQ = projection_data.Q.T @ projection_data.Q
        np.testing.assert_allclose(QtQ, np.eye(projection_data.Q.shape[1]), atol=1e-10)


if __name__ == "__main__":
    unittest.main()
