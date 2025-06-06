"""
Test edge cases for covariate adjustment, including QR decomposition failures.
"""

import unittest
import numpy as np
import pandas as pd
import logging
from ldcov.compute.covariate import (
    regress_out_covariates,
    _apply_fwl_projection,
    standardize_genotypes,
)
from ldcov.compute.correlation import compute_correlation_matrix

# Set up logging to capture warnings and errors
logging.basicConfig(level=logging.INFO)


class TestCovariateEdgeCases(unittest.TestCase):
    """Test edge cases and numerical stability in covariate adjustment."""

    def test_rank_deficient_covariates(self):
        """Test handling of rank-deficient (multicollinear) covariates."""
        n_samples = 50
        n_variants = 5

        # Create genotypes
        np.random.seed(42)
        genotypes = np.random.randn(n_samples, n_variants)

        # Standardize genotypes
        std_geno, _, _ = standardize_genotypes(genotypes.copy())

        # Create rank-deficient covariates (PC2 = 2*PC1 + noise)
        covariates = pd.DataFrame(
            {
                "PC1": np.random.randn(n_samples),
            }
        )
        covariates["PC2"] = 2 * covariates["PC1"] + 0.001 * np.random.randn(n_samples)
        covariates["PC3"] = covariates["PC1"] + covariates["PC2"]  # Perfect linear combination

        # Should handle rank deficiency without error
        adjusted = regress_out_covariates(std_geno.copy(), covariates)

        # Check that adjustment was performed
        self.assertEqual(adjusted.shape, std_geno.shape)
        self.assertFalse(np.array_equal(adjusted, std_geno))

        # Check no NaN values
        self.assertFalse(np.any(np.isnan(adjusted)))

    def test_perfect_collinearity_with_genotypes(self):
        """Test when covariates perfectly explain some genotypes."""
        n_samples = 50
        n_variants = 5

        # Create genotypes
        np.random.seed(42)
        genotypes = np.random.randn(n_samples, n_variants)

        # Make first variant perfectly correlated with a covariate
        covariate_values = np.random.randn(n_samples)
        genotypes[:, 0] = 2.5 * covariate_values + 1.0  # Linear relationship

        # Standardize genotypes
        std_geno, _, _ = standardize_genotypes(genotypes.copy())

        # Create covariates
        covariates = pd.DataFrame({"cov1": covariate_values, "cov2": np.random.randn(n_samples)})

        # Perform adjustment
        adjusted = regress_out_covariates(std_geno.copy(), covariates)

        # First variant should be nearly zero after adjustment
        self.assertTrue(np.allclose(adjusted[:, 0], 0, atol=1e-10))

        # Other variants should still have variance
        for i in range(1, n_variants):
            self.assertGreater(np.var(adjusted[:, i]), 1e-5)

    def test_all_zero_genotypes(self):
        """Test handling of genotypes with no variance."""
        n_samples = 50
        n_variants = 5

        # Create genotypes with one zero-variance variant
        genotypes = np.random.randn(n_samples, n_variants)
        genotypes[:, 2] = 1.0  # Constant variant

        # Standardize - should handle zero variance
        std_geno, _, norms = standardize_genotypes(genotypes.copy())

        # Check that constant variant has norm of 1.0 (to avoid division by zero)
        self.assertEqual(norms[2], 1.0)

        # Create covariates
        covariates = pd.DataFrame({"PC1": np.random.randn(n_samples)})

        # Should handle without error
        adjusted = regress_out_covariates(std_geno.copy(), covariates)
        self.assertFalse(np.any(np.isnan(adjusted)))

    def test_extreme_covariate_values(self):
        """Test numerical stability with extreme covariate values."""
        n_samples = 50
        n_variants = 5

        # Create genotypes
        genotypes = np.random.randn(n_samples, n_variants)
        std_geno, _, _ = standardize_genotypes(genotypes.copy())

        # Create covariates with extreme values
        covariates = pd.DataFrame(
            {
                "extreme1": np.random.randn(n_samples) * 1e10,  # Very large scale
                "extreme2": np.random.randn(n_samples) * 1e-10,  # Very small scale
                "normal": np.random.randn(n_samples),
            }
        )

        # Should handle without numerical issues
        adjusted = regress_out_covariates(std_geno.copy(), covariates)

        # Check no NaN or Inf values
        self.assertFalse(np.any(np.isnan(adjusted)))
        self.assertFalse(np.any(np.isinf(adjusted)))

    def test_empty_genotype_matrix(self):
        """Test error handling for empty genotype matrix."""
        # Empty matrix
        empty_geno = np.array([]).reshape(0, 0)
        covariates = pd.DataFrame({"cov1": []})

        # _apply_fwl_projection no longer checks for empty matrix
        # (this is now checked in load_bgen)
        # Just verify it doesn't crash with empty input
        result = _apply_fwl_projection(empty_geno, covariates.to_numpy())
        self.assertEqual(result.shape, empty_geno.shape)

    def test_correlation_matrix_with_zero_variance_adjusted(self):
        """Test correlation computation when adjustment removes all variance."""
        n_samples = 50
        n_variants = 3

        # Create genotypes perfectly explained by covariates
        covariates_raw = np.random.randn(n_samples, 2)
        genotypes = np.zeros((n_samples, n_variants))

        # Make genotypes linear combinations of covariates
        genotypes[:, 0] = 2 * covariates_raw[:, 0] + 3 * covariates_raw[:, 1]
        genotypes[:, 1] = -1 * covariates_raw[:, 0] + 0.5 * covariates_raw[:, 1]
        genotypes[:, 2] = covariates_raw[:, 0]

        # Standardize
        std_geno, _, _ = standardize_genotypes(genotypes.copy())

        # Create covariate DataFrame
        covariates = pd.DataFrame(covariates_raw, columns=["cov1", "cov2"])

        # Adjust
        adjusted = regress_out_covariates(std_geno.copy(), covariates)

        # All adjusted genotypes should be near zero
        self.assertTrue(np.allclose(adjusted, 0, atol=1e-10))

        # Computing correlation on all-zero matrix should work but produce zero matrix
        corr_matrix = compute_correlation_matrix(adjusted)

        # Correlation matrix should be all zeros when genotypes are all zero
        self.assertTrue(np.allclose(corr_matrix, 0, atol=1e-10))

    def test_singular_covariate_matrix(self):
        """Test handling of singular covariate matrix."""
        n_samples = 50
        n_variants = 5

        # Create genotypes
        genotypes = np.random.randn(n_samples, n_variants)
        std_geno, _, _ = standardize_genotypes(genotypes.copy())

        # Create singular covariate matrix (duplicate columns)
        cov_values = np.random.randn(n_samples)
        covariates = pd.DataFrame(
            {
                "cov1": cov_values,
                "cov2": cov_values,  # Exact duplicate
                "cov3": cov_values,  # Another duplicate
            }
        )

        # Should handle singular matrix without crashing
        adjusted = regress_out_covariates(std_geno.copy(), covariates)

        # Check result is valid
        self.assertFalse(np.any(np.isnan(adjusted)))
        self.assertEqual(adjusted.shape, std_geno.shape)

    def test_fallback_to_pseudoinverse(self):
        """Test that fallback to pseudoinverse works correctly."""
        n_samples = 20
        n_variants = 3

        # Create a pathological case that might cause QR to fail
        # Small sample size with many covariates
        genotypes = np.random.randn(n_samples, n_variants)
        std_geno, _, _ = standardize_genotypes(genotypes.copy())

        # Create more covariates than reasonable
        n_covs = 15
        covariates = pd.DataFrame(
            np.random.randn(n_samples, n_covs), columns=[f"cov{i}" for i in range(n_covs)]
        )

        # Add some highly correlated covariates
        for i in range(5, 10):
            covariates[f"cov{i}"] = covariates["cov0"] + 0.01 * np.random.randn(n_samples)

        # Should complete without error (may use fallback)
        adjusted = regress_out_covariates(std_geno.copy(), covariates)

        # Verify output is valid
        self.assertFalse(np.any(np.isnan(adjusted)))
        self.assertEqual(adjusted.shape, std_geno.shape)

    def test_numerical_precision_edge_case(self):
        """Test numerical precision with values near machine epsilon."""
        n_samples = 50
        n_variants = 5

        # Create genotypes with very small variance
        genotypes = np.random.randn(n_samples, n_variants) * 1e-15
        genotypes += 1.0  # Add offset to avoid zero mean

        # This should work despite small values
        std_geno, _, _ = standardize_genotypes(genotypes.copy())

        # Create normal covariates
        covariates = pd.DataFrame({"PC1": np.random.randn(n_samples)})

        # Should handle without numerical issues
        adjusted = regress_out_covariates(std_geno.copy(), covariates)

        # Check output
        self.assertFalse(np.any(np.isnan(adjusted)))
        self.assertFalse(np.any(np.isinf(adjusted)))

    def test_too_few_samples_error(self):
        """Test that error is raised when n_samples <= n_covariates."""
        n_samples = 5
        n_variants = 10

        # Create genotypes
        genotypes = np.random.randn(n_samples, n_variants)
        std_geno, _, _ = standardize_genotypes(genotypes.copy())

        # Create more covariates than samples
        covariates = pd.DataFrame(
            {
                "PC1": np.random.randn(n_samples),
                "PC2": np.random.randn(n_samples),
                "PC3": np.random.randn(n_samples),
                "PC4": np.random.randn(n_samples),
                "PC5": np.random.randn(n_samples),
            }
        )

        # Should raise ValueError
        with self.assertRaises(ValueError) as context:
            regress_out_covariates(std_geno.copy(), covariates)

        self.assertIn("Number of samples", str(context.exception))
        self.assertIn("rank-deficient", str(context.exception))

    def test_nan_genotypes_error(self):
        """Test that NaN values in genotypes are caught during standardization."""
        n_samples = 50
        n_variants = 5

        # Create genotypes with some NaN values
        genotypes = np.random.randn(n_samples, n_variants).astype(np.float64)
        genotypes[10, 2] = np.nan
        genotypes[20, 3] = np.nan

        # Should raise ValueError during standardization
        with self.assertRaises(ValueError) as context:
            standardize_genotypes(genotypes.copy())

        self.assertIn("Genotype matrix contains NaN", str(context.exception))


if __name__ == "__main__":
    unittest.main()
