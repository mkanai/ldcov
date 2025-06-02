"""
Consolidated I/O tests for the ldcov package.

This module combines tests for:
- BGEN file reading and writing
- Correlation matrix I/O
- Covariate loading
- Metadata handling
"""

import unittest
import numpy as np
import pandas as pd
import os
import tempfile
import shutil
from pathlib import Path

# import gzip  # Used by pd.read_csv for compressed files

from ldcov.io.bgen_reader import load_bgen, BgenFileReader
from ldcov.io.bgen_writer import (
    correlation_preserving_transform,
    write_bgen,
    save_metadata,
)
from ldcov.io.correlation_io import save_correlation_matrix, load_correlation_matrix
from ldcov.io.covariate_loader import load_covariates
from ldcov.compute.covariate import standardize_genotypes
from ldcov.utils.categorical_utils import one_hot_encode_categorical


class TestIO(unittest.TestCase):
    """Test cases for all I/O operations."""

    @classmethod
    def setUpClass(cls):
        """Set up test data."""
        cls.examples_dir = Path(__file__).parents[1] / "examples"
        cls.bgen_file = cls.examples_dir / "data" / "data.bgen"
        cls.bgi_file = cls.examples_dir / "data" / "data.bgen.bgi"
        cls.sample_file = cls.examples_dir / "data" / "data.sample"

        # Create temporary directory for output files
        cls.temp_dir = tempfile.mkdtemp(prefix="ldcov_test_io_")

        # Load test data once
        cls.genotypes, cls.variant_info, cls.sample_ids = load_bgen(
            file_path=str(cls.bgen_file),
            index_path=str(cls.bgi_file),
            sample_path=str(cls.sample_file),
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up test data."""
        shutil.rmtree(cls.temp_dir)

    # ==================== BGEN Reader Tests ====================

    def test_bgen_reader_initialization(self):
        """Test BgenFileReader initialization."""
        reader = BgenFileReader(
            file_path=str(self.bgen_file),
            index_path=str(self.bgi_file),
            sample_path=str(self.sample_file),
        )
        self.assertIsNotNone(reader)
        self.assertIsNotNone(reader.bgen_file)
        self.assertGreater(len(reader.sample_ids), 0)

    def test_load_bgen_basic(self):
        """Test basic BGEN loading."""
        genotypes, variant_info, sample_ids = load_bgen(
            file_path=str(self.bgen_file),
            index_path=str(self.bgi_file),
            sample_path=str(self.sample_file),
        )

        # Check shapes
        self.assertGreater(genotypes.shape[0], 0)
        self.assertGreater(genotypes.shape[1], 0)
        self.assertEqual(len(variant_info), genotypes.shape[1])
        self.assertEqual(len(sample_ids), genotypes.shape[0])

    def test_load_bgen_with_region(self):
        """Test BGEN loading with region filter."""
        region = "01:1000000-2000000"
        genotypes, variant_info, sample_ids = load_bgen(
            file_path=str(self.bgen_file),
            index_path=str(self.bgi_file),
            sample_path=str(self.sample_file),
            region=region,
        )

        # Check that all variants are within the region
        for _, variant in variant_info.iterrows():
            self.assertEqual(variant["chrom"], "01")
            self.assertGreaterEqual(variant["pos"], 1000000)
            self.assertLessEqual(variant["pos"], 2000000)

    def test_load_bgen_without_index(self):
        """Test BGEN loading without index file."""
        genotypes, variant_info, sample_ids = load_bgen(
            file_path=str(self.bgen_file), sample_path=str(self.sample_file)
        )

        # Should still load successfully
        self.assertGreater(genotypes.shape[0], 0)
        self.assertGreater(genotypes.shape[1], 0)

    # ==================== BGEN Writer Tests ====================

    def test_correlation_preserving_transform(self):
        """Test correlation-preserving transformation."""
        # Standardize genotypes
        standardized_genotypes, means, norms = standardize_genotypes(
            self.genotypes.copy(), center=True, scale=True
        )

        # Apply transform
        allelic_genotypes = correlation_preserving_transform(standardized_genotypes)

        # Check properties
        self.assertEqual(allelic_genotypes.shape, self.genotypes.shape)
        self.assertTrue(np.all((allelic_genotypes >= 0) & (allelic_genotypes <= 2)))

    def test_write_bgen_and_read_back(self):
        """Test writing and reading back BGEN files."""
        output_bgen = os.path.join(self.temp_dir, "test_output.bgen")

        # Write genotypes
        write_bgen(
            genotypes=self.genotypes,
            variant_info=self.variant_info,
            sample_ids=self.sample_ids,
            output_file=output_bgen,
        )

        self.assertTrue(os.path.exists(output_bgen))

        # Read back
        loaded_genotypes, loaded_variant_info, loaded_sample_ids = load_bgen(file_path=output_bgen)

        # Check consistency
        self.assertEqual(loaded_genotypes.shape, self.genotypes.shape)
        self.assertEqual(len(loaded_variant_info), len(self.variant_info))
        self.assertEqual(loaded_sample_ids, self.sample_ids)

    def test_save_metadata(self):
        """Test metadata saving and loading."""
        metadata_file = os.path.join(self.temp_dir, "test.metadata.tsv.gz")

        # Create test metadata DataFrame with required columns
        variant_info = pd.DataFrame(
            {
                "id": ["var1", "var2", "var3"],
                "chrom": ["01", "01", "01"],
                "pos": [100, 200, 300],
                "ref": ["A", "C", "G"],
                "alt": ["G", "T", "A"],
                "mean": [0.1, 0.2, 0.3],
                "norm": [1.0, 1.1, 1.2],
            }
        )

        # Save metadata
        save_metadata(variant_info, metadata_file)

        self.assertTrue(os.path.exists(metadata_file))

        # Load and verify
        loaded_df = pd.read_csv(metadata_file, sep="\t")
        self.assertIn("mean", loaded_df.columns)
        self.assertIn("norm", loaded_df.columns)
        np.testing.assert_array_almost_equal(loaded_df["mean"].values, [0.1, 0.2, 0.3])
        np.testing.assert_array_almost_equal(loaded_df["norm"].values, [1.0, 1.1, 1.2])

    # ==================== Correlation I/O Tests ====================

    def test_save_and_load_correlation_matrix(self):
        """Test saving and loading correlation matrices in different formats."""
        # Create test correlation matrix
        n_vars = 10
        test_matrix = np.random.rand(n_vars, n_vars)
        test_matrix = (test_matrix + test_matrix.T) / 2  # Make symmetric
        np.fill_diagonal(test_matrix, 1.0)

        variant_info = pd.DataFrame(
            {
                "id": [f"var_{i}" for i in range(n_vars)],
                "chrom": ["01"] * n_vars,
                "pos": range(1000, 1000 + n_vars),
                "ref": ["A"] * n_vars,
                "alt": ["G"] * n_vars,
            }
        )

        # Test matrix format
        matrix_file = os.path.join(self.temp_dir, "test.ld")
        save_correlation_matrix(
            test_matrix, matrix_file, variant_info=variant_info, output_format="matrix"
        )
        self.assertTrue(os.path.exists(matrix_file))

        loaded_matrix, loaded_variant_info = load_correlation_matrix(matrix_file)
        np.testing.assert_array_almost_equal(test_matrix, loaded_matrix)

        # Test long format
        long_file = os.path.join(self.temp_dir, "test.ld.gz")
        save_correlation_matrix(
            test_matrix, long_file, variant_info=variant_info, output_format="long"
        )
        self.assertTrue(os.path.exists(long_file))

        # Test bcor format (binary)
        bcor_file = os.path.join(self.temp_dir, "test.bcor")
        save_correlation_matrix(
            test_matrix, bcor_file, variant_info=variant_info, output_format="bcor"
        )
        self.assertTrue(os.path.exists(bcor_file))

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


if __name__ == "__main__":
    unittest.main()
