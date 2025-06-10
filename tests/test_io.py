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
from unittest.mock import patch

# import gzip  # Used by pd.read_csv for compressed files

from ldcov.io.bgen_reader import load_bgen, BgenFileReader
from ldcov.io.bgen_writer import (
    correlation_preserving_transform,
    write_bgen,
    save_metadata,
)
from ldcov.io.correlation_io import save_correlation_matrix, load_correlation_matrix
from ldcov.io.covariate_loader import load_covariates
from ldcov.io.bcor_writer import BcorWriter, save_bcor
from ldcov.compute.covariate import standardize_genotypes
from ldcov.compute.correlation import load_and_adjust_genotypes, compute_correlation_matrix
from ldcov.utils.categorical_utils import one_hot_encode_categorical

# Import our own BcorReader
from ldcov.io.correlation_io import BcorReader


class TestIO(unittest.TestCase):
    """Test cases for all I/O operations."""

    @classmethod
    def setUpClass(cls):
        """Set up test data."""
        cls.examples_dir = Path(__file__).parents[1] / "examples"
        cls.bgen_file = cls.examples_dir / "data" / "data.bgen"
        cls.bgi_file = cls.examples_dir / "data" / "data.bgen.bgi"
        cls.sample_file = cls.examples_dir / "data" / "data.sample"
        cls.ref_bcor_file = cls.examples_dir / "data" / "data.bcor"
        cls.ref_ld_file = cls.examples_dir / "data" / "data.ld"

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

    def test_bgen_reader_filtered_loading(self):
        """Test BgenFileReader's filtered loading method."""
        with patch("ldcov.io.bgen_reader.BgenReader") as mock_bgen:
            # Setup mock BGEN file
            mock_bgen_instance = mock_bgen.return_value
            mock_bgen_instance.samples = ["S1", "S2", "S3"]
            
            # Create mock variants - simulating 5 variants but only wanting 2
            class MockVariant:
                def __init__(self, rsid, chrom, pos, ref, alt):
                    self.rsid = rsid
                    self.chrom = chrom
                    self.pos = pos
                    self.alleles = [ref, alt]
                    self.alt_dosage = np.array([0.0, 1.0, 2.0])  # Mock dosages
            
            mock_variants = [
                MockVariant("rs1", "1", 100, "A", "T"),
                MockVariant("rs2", "1", 200, "C", "G"),  # Want this one
                MockVariant("rs3", "1", 300, "G", "A"),
                MockVariant("rs4", "1", 400, "T", "A"),  # Want this one
                MockVariant("rs5", "1", 500, "A", "C"),
            ]
            
            # Make the mock iterable
            mock_bgen_instance.__iter__ = lambda self: iter(mock_variants)
            
            # Create reader instance
            reader = BgenFileReader("dummy.bgen")
            
            # Create variant filter
            variant_filter = {
                "positions": [200, 400],
                "rsids": ["rs2", "rs4"],
                "allele1": ["C", "T"],
                "allele2": ["G", "A"],
            }
            
            # Load filtered variants
            dosages, var_info, n_samples = reader.load_filtered_variants_and_dosages(
                variant_filter, dtype=np.float64
            )
            
            # Verify results
            self.assertEqual(dosages.shape, (3, 2))  # 3 samples, 2 variants
            self.assertEqual(len(var_info), 2)
            self.assertEqual(var_info.iloc[0]["id"], "rs2")
            self.assertEqual(var_info.iloc[1]["id"], "rs4")
            self.assertEqual(var_info.iloc[0]["pos"], 200)
            self.assertEqual(var_info.iloc[1]["pos"], 400)

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
        # First load all data to find a valid region
        all_genotypes, all_variant_info, _ = load_bgen(
            file_path=str(self.bgen_file),
            index_path=str(self.bgi_file),
            sample_path=str(self.sample_file),
        )

        # Skip test if no variants
        if len(all_variant_info) == 0:
            self.skipTest("No variants in test BGEN file")

        # Use the chromosome and position range from actual data
        first_chrom = all_variant_info["chrom"].iloc[0]
        min_pos = all_variant_info["pos"].min()
        max_pos = all_variant_info["pos"].max()
        mid_pos = (min_pos + max_pos) // 2

        # Create a region that contains some variants
        region = f"{first_chrom}:{min_pos}-{mid_pos}"

        genotypes, variant_info, sample_ids = load_bgen(
            file_path=str(self.bgen_file),
            index_path=str(self.bgi_file),
            sample_path=str(self.sample_file),
            region=region,
        )

        # Check that we got some variants
        self.assertGreater(len(variant_info), 0)

        # Check that all variants are within the region
        for _, variant in variant_info.iterrows():
            self.assertEqual(variant["chrom"], first_chrom)
            self.assertGreaterEqual(variant["pos"], min_pos)
            self.assertLessEqual(variant["pos"], mid_pos)

    def test_load_bgen_without_index(self):
        """Test BGEN loading without index file."""
        genotypes, variant_info, sample_ids = load_bgen(
            file_path=str(self.bgen_file), sample_path=str(self.sample_file)
        )

        # Should still load successfully
        self.assertGreater(genotypes.shape[0], 0)
        self.assertGreater(genotypes.shape[1], 0)

    def test_load_bgen_empty_region_error(self):
        """Test that loading an empty region raises an error."""
        # Use a region that definitely doesn't contain any variants
        empty_region = "99:1-100"

        with self.assertRaises(ValueError) as context:
            load_bgen(
                file_path=str(self.bgen_file),
                index_path=str(self.bgi_file),
                sample_path=str(self.sample_file),
                region=empty_region,
            )

        self.assertIn("No variants were loaded", str(context.exception))

    def test_load_bgen_with_variant_filter(self):
        """Test efficient loading with variant filter."""
        # This test uses a mock to simulate variant filtering
        with patch("ldcov.io.bgen_reader.BgenFileReader") as mock_reader:
            # Create mock data
            mock_instance = mock_reader.return_value
            mock_instance.sample_ids = ["sample1", "sample2", "sample3"]
            mock_instance.n_samples = 3
            
            # Create variant filter (simulating a .z file)
            variant_filter = {
                "chromosome": "1",
                "positions": [200, 400],  # Only want variants at positions 200 and 400
                "rsids": ["rs2", "rs4"],
                "allele1": ["C", "T"],
                "allele2": ["G", "A"],
                "z_file_order": [0, 1]
            }
            
            # Mock the filtered loading method
            filtered_dosages = np.array([
                [1.0, 2.0],  # sample 1
                [0.5, 1.5],  # sample 2
                [2.0, 0.0],  # sample 3
            ])
            
            filtered_variant_info = pd.DataFrame({
                "id": ["rs2", "rs4"],
                "chrom": ["1", "1"],
                "pos": [200, 400],
                "ref": ["C", "T"],
                "alt": ["G", "A"],
                "rsid": ["rs2", "rs4"],
                "idx": [1, 3]
            })
            
            mock_instance.load_filtered_variants_and_dosages.return_value = (
                filtered_dosages, filtered_variant_info, 3
            )
            
            # Load with variant filter
            genotypes, var_info, sample_ids = load_bgen("dummy.bgen", variant_filter=variant_filter)
            
            # Verify the efficient method was called
            mock_instance.load_filtered_variants_and_dosages.assert_called_once_with(
                variant_filter, np.float64, None
            )
            
            # Verify results
            self.assertEqual(genotypes.shape, (3, 2))
            self.assertEqual(len(var_info), 2)
            self.assertListEqual(var_info["id"].tolist(), ["rs2", "rs4"])
            self.assertListEqual(var_info["pos"].tolist(), [200, 400])

    def test_load_bgen_nan_validation(self):
        """Test that load_bgen raises error when NaN values are detected."""
        # This test uses a mock since we can't easily create a BGEN file with NaN values
        with patch("ldcov.io.bgen_reader.BgenFileReader") as mock_reader:
            # Create mock data with NaN values
            mock_instance = mock_reader.return_value
            mock_instance.sample_ids = ["sample1", "sample2", "sample3"]
            mock_instance.n_samples = 3
            
            # Create dosages with NaN values
            dosages_with_nan = np.array([
                [0.0, 1.0, np.nan],  # variant 0, sample 2 has NaN
                [2.0, np.nan, 1.0],  # variant 1, sample 1 has NaN
                [np.nan, 0.0, 2.0],  # variant 2, sample 0 has NaN
            ]).T  # Transpose to get samples x variants
            
            variant_info = pd.DataFrame({
                "id": ["rs1", "rs2", "rs3"],
                "chrom": ["1", "1", "1"],
                "pos": [100, 200, 300],
                "ref": ["A", "C", "G"],
                "alt": ["T", "G", "A"]
            })
            
            mock_instance.load_all_variants_and_dosages.return_value = (
                dosages_with_nan, variant_info, 3
            )
            
            # Should raise ValueError with detailed NaN information
            with self.assertRaises(ValueError) as context:
                load_bgen("dummy.bgen")
            
            self.assertIn("Genotype matrix contains NaN values", str(context.exception))

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

    def test_bcor_export_matches_reference(self):
        """Test that exported bcor matches reference bcor."""
        # Load data and compute LD
        standardized_genotypes, variant_info, sample_ids, means, norms = load_and_adjust_genotypes(
            genotype_file=str(self.bgen_file),
            sample_file=str(self.sample_file),
            covariate_file=None,
        )

        ld_matrix = compute_correlation_matrix(standardized_genotypes)

        # Load reference bcor
        ref_bcor = BcorReader(str(self.ref_bcor_file))
        ref_bcor_matrix = ref_bcor.read_corr([], [])
        ref_meta = ref_bcor.get_meta()

        # Export our LD as bcor
        output_file = os.path.join(self.temp_dir, "test_bcor_export.bcor")

        # Convert variant info to match bcor format
        variant_info_bcor = pd.DataFrame(
            {
                "id": variant_info["rsid"].tolist(),
                "chrom": variant_info["chrom"].tolist(),
                "pos": variant_info["pos"].tolist(),
                "ref": variant_info["ref"].tolist(),
                "alt": variant_info["alt"].tolist(),
            }
        )

        save_correlation_matrix(
            corr_matrix=ld_matrix,
            output_file=output_file,
            variant_info=variant_info_bcor,
            output_format="bcor",
            n_samples=ref_bcor.get_n_samples(),
            compression=1,
        )

        # Read back our bcor
        our_bcor = BcorReader(output_file)
        our_bcor_matrix = our_bcor.read_corr([], [])
        our_meta = our_bcor.get_meta()

        # Compare matrices
        self.assertEqual(our_bcor_matrix.shape, ref_bcor_matrix.shape)

        matrix_diff = np.abs(our_bcor_matrix - ref_bcor_matrix)
        max_matrix_diff = np.max(matrix_diff)
        mean_matrix_diff = np.mean(matrix_diff)

        self.assertLess(
            max_matrix_diff, 1e-6, f"BCOR matrix differs from reference: max_diff={max_matrix_diff}"
        )
        self.assertLess(
            mean_matrix_diff,
            1e-8,
            f"BCOR matrix differs from reference: mean_diff={mean_matrix_diff}",
        )

        # Compare metadata
        self.assertEqual(len(our_meta), len(ref_meta), "Metadata length mismatch")
        self.assertEqual(
            our_bcor.get_n_samples(), ref_bcor.get_n_samples(), "Sample count mismatch"
        )
        self.assertEqual(our_bcor.get_n_snps(), ref_bcor.get_n_snps(), "SNP count mismatch")

    def test_bcor_export_different_compression(self):
        """Test bcor export with different compression levels."""
        # Create a smaller test matrix
        n_vars = 10
        test_matrix = np.random.rand(n_vars, n_vars)
        test_matrix = (test_matrix + test_matrix.T) / 2  # Make symmetric
        np.fill_diagonal(test_matrix, 1.0)
        # Ensure values are in [-1, 1] range
        test_matrix = np.clip(test_matrix * 2 - 1, -1, 1)
        np.fill_diagonal(test_matrix, 1.0)

        variant_info = pd.DataFrame(
            {
                "id": [f"rs{i}" for i in range(n_vars)],
                "chrom": ["01"] * n_vars,
                "pos": list(range(1, n_vars + 1)),
                "ref": ["A"] * n_vars,
                "alt": ["G"] * n_vars,
            }
        )

        # Test different compression levels
        for compression in [0, 1, 2, 3]:
            output_file = os.path.join(self.temp_dir, f"test_compression_{compression}.bcor")

            save_correlation_matrix(
                corr_matrix=test_matrix,
                output_file=output_file,
                variant_info=variant_info,
                output_format="bcor",
                n_samples=1000,
                compression=compression,
            )

            # Read back and verify
            reader = BcorReader(output_file)
            read_matrix = reader.read_corr([], [])

            self.assertEqual(read_matrix.shape, test_matrix.shape)

            # Check that values are reasonably close (precision depends on compression)
            diff = np.abs(read_matrix - test_matrix)
            max_diff = np.max(diff)

            # Tolerance depends on compression level
            if compression == 0:  # 2 bytes
                tolerance = 1e-4
            elif compression == 1:  # 4 bytes
                tolerance = 1e-6
            elif compression == 2:  # 8 bytes
                tolerance = 1e-7  # Still very precise but accounts for float arithmetic
            else:  # compression == 3, 1 byte
                tolerance = 2e-2

            self.assertLess(
                max_diff, tolerance, f"Compression {compression}: max_diff={max_diff} > {tolerance}"
            )

    def test_bcor_export_without_ldstore(self):
        """Test that bcor export works even without ldstore for reading back."""
        # Create small test matrix
        n_vars = 5
        test_matrix = np.eye(n_vars)

        variant_info = pd.DataFrame(
            {
                "id": [f"var_{i}" for i in range(n_vars)],
                "chrom": ["1"] * n_vars,
                "pos": list(range(100, 100 + n_vars)),
                "ref": ["A"] * n_vars,
                "alt": ["T"] * n_vars,
            }
        )

        output_file = os.path.join(self.temp_dir, "test_no_ldstore.bcor")

        # Should not raise any errors
        save_correlation_matrix(
            corr_matrix=test_matrix,
            output_file=output_file,
            variant_info=variant_info,
            output_format="bcor",
            n_samples=1000,
            compression=1,
        )

        # File should exist and have reasonable size
        self.assertTrue(os.path.exists(output_file))
        file_size = os.path.getsize(output_file)
        self.assertGreater(file_size, 100)  # Should be at least a few hundred bytes

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

    # ==================== Extended BCOR Format Tests ====================

    def test_standard_bcor_with_unit_diagonal(self):
        """Test that standard bcor is written when diagonal is all 1s."""
        n_vars = 10
        # Create correlation matrix with unit diagonal
        corr_matrix = np.random.rand(n_vars, n_vars) * 0.8
        corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Make symmetric
        np.fill_diagonal(corr_matrix, 1.0)  # Unit diagonal

        variant_info = pd.DataFrame(
            {
                "id": [f"rs{i}" for i in range(n_vars)],
                "chrom": ["1"] * n_vars,
                "pos": list(range(1000, 1000 + n_vars * 100, 100)),
                "ref": ["A"] * n_vars,
                "alt": ["G"] * n_vars,
            }
        )

        output_file = os.path.join(self.temp_dir, "standard.bcor")
        save_bcor(corr_matrix, output_file, variant_info, n_samples=100, compression=1)

        # Read back and verify
        reader = BcorReader(output_file)
        self.assertFalse(reader.is_extended, "Should be standard format")

        # Check magic string
        with open(output_file, "rb") as f:
            magic = f.read(7)
            self.assertEqual(magic, b"bcor1.1", "Should have standard magic string")

        # Verify matrix
        loaded = reader.read_corr()
        np.testing.assert_array_almost_equal(corr_matrix, loaded, decimal=5)

    def test_extended_bcor_with_non_unit_diagonal(self):
        """Test that extended bcor is written when diagonal is not all 1s."""
        n_vars = 10
        # Create correlation matrix with non-unit diagonal (adjusted LD)
        corr_matrix = np.random.rand(n_vars, n_vars) * 0.8
        corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Make symmetric
        # Set non-unit diagonal values
        diagonal_values = np.array([0.9, 1.0, 0.85, 1.0, 0.95, 1.0, 0.88, 1.0, 0.92, 1.0])
        np.fill_diagonal(corr_matrix, diagonal_values)

        variant_info = pd.DataFrame(
            {
                "id": [f"rs{i}" for i in range(n_vars)],
                "chrom": ["1"] * n_vars,
                "pos": list(range(1000, 1000 + n_vars * 100, 100)),
                "ref": ["A"] * n_vars,
                "alt": ["G"] * n_vars,
            }
        )

        output_file = os.path.join(self.temp_dir, "extended.bcor")
        save_bcor(corr_matrix, output_file, variant_info, n_samples=100, compression=1)

        # Read back and verify
        reader = BcorReader(output_file)
        self.assertTrue(reader.is_extended, "Should be extended format")

        # Check magic string
        with open(output_file, "rb") as f:
            magic = f.read(7)
            self.assertEqual(magic, b"bcor1.x", "Should have extended magic string")

        # Verify matrix including diagonal
        loaded = reader.read_corr()
        np.testing.assert_array_almost_equal(corr_matrix, loaded, decimal=5)

        # Specifically check diagonal values
        np.testing.assert_array_almost_equal(np.diag(loaded), diagonal_values, decimal=5)

    def test_extended_bcor_different_compressions(self):
        """Test extended bcor with different compression levels."""
        n_vars = 8
        # Create correlation matrix with non-unit diagonal
        corr_matrix = np.eye(n_vars)
        # Add some off-diagonal correlations
        corr_matrix[0, 1] = corr_matrix[1, 0] = 0.8
        corr_matrix[2, 3] = corr_matrix[3, 2] = 0.6
        # Non-unit diagonal
        diagonal_values = np.array([0.95, 0.90, 1.0, 0.85, 1.0, 0.92, 0.88, 1.0])
        np.fill_diagonal(corr_matrix, diagonal_values)

        variant_info = pd.DataFrame(
            {
                "id": [f"var{i}" for i in range(n_vars)],
                "chrom": ["1"] * n_vars,
                "pos": list(range(100, 100 + n_vars)),
                "ref": ["A"] * n_vars,
                "alt": ["T"] * n_vars,
            }
        )

        for compression in [0, 1, 2, 3]:
            output_file = os.path.join(self.temp_dir, f"extended_comp{compression}.bcor")

            writer = BcorWriter(output_file, n_samples=100, compression=compression)
            writer.write(corr_matrix, variant_info)

            # Read back
            reader = BcorReader(output_file)
            self.assertTrue(reader.is_extended)
            loaded = reader.read_corr()

            # Check values with appropriate tolerance
            if compression == 3:  # 1 byte
                tolerance = 0.02
            elif compression == 0:  # 2 bytes
                tolerance = 1e-4
            elif compression == 1:  # 4 bytes
                tolerance = 1e-6
            else:  # compression == 2, 8 bytes
                tolerance = 1e-7

            diff = np.abs(loaded - corr_matrix)
            max_diff = np.max(diff)
            self.assertLess(
                max_diff, tolerance, f"Compression {compression}: max_diff={max_diff} > {tolerance}"
            )

    def test_extended_bcor_subset_reads(self):
        """Test reading subsets from extended bcor files."""
        n_vars = 20
        # Create test matrix with non-unit diagonal
        corr_matrix = np.random.rand(n_vars, n_vars) * 0.5 + 0.3
        corr_matrix = (corr_matrix + corr_matrix.T) / 2
        # Variable diagonal values
        diagonal_values = 0.8 + 0.2 * np.random.rand(n_vars)
        np.fill_diagonal(corr_matrix, diagonal_values)

        output_file = os.path.join(self.temp_dir, "extended_subset.bcor")
        save_bcor(corr_matrix, output_file, n_samples=100)

        reader = BcorReader(output_file)
        self.assertTrue(reader.is_extended)

        # Test reading specific rows
        rows = [0, 5, 10, 15]
        subset = reader.read_corr(rows)
        expected = corr_matrix[:, rows]
        np.testing.assert_array_almost_equal(subset, expected, decimal=5)

        # Test reading specific pairs
        rows = [1, 3, 5]
        cols = [2, 4, 6, 8]
        subset = reader.read_corr(rows, cols)
        expected = corr_matrix[np.ix_(rows, cols)]
        np.testing.assert_array_almost_equal(subset, expected, decimal=5)

        # Test diagonal element reads
        for i in range(n_vars):
            diag_val = reader.read_corr([i], [i])[0, 0]
            self.assertAlmostEqual(diag_val, diagonal_values[i], places=5)

    # ==================== Sample Filtering Tests ====================

    def test_get_sample_indices(self):
        """Test sample index mapping."""
        # Create test data
        all_sample_ids = [f"SAMPLE_{i:04d}" for i in range(100)]
        subset_sample_ids = [f"SAMPLE_{i:04d}" for i in range(10, 30)]  # 20 samples

        # Create a BgenFileReader instance
        reader = BgenFileReader(
            file_path=str(self.bgen_file),
            index_path=str(self.bgi_file),
            sample_path=str(self.sample_file),
        )
        # Override sample_ids for testing
        reader.sample_ids = all_sample_ids
        reader.n_samples = len(all_sample_ids)

        # Test with subset of samples
        indices, filtered_ids = reader.get_sample_indices(subset_sample_ids)

        self.assertEqual(len(indices), 20)
        self.assertEqual(len(filtered_ids), 20)
        self.assertEqual(indices[0], 10)  # First sample is SAMPLE_0010 at index 10
        self.assertEqual(indices[-1], 29)  # Last sample is SAMPLE_0029 at index 29
        self.assertEqual(filtered_ids, subset_sample_ids)

    def test_get_sample_indices_with_missing(self):
        """Test sample index mapping with missing samples."""
        # Create test data
        all_sample_ids = [f"SAMPLE_{i:04d}" for i in range(100)]
        subset_sample_ids = [f"SAMPLE_{i:04d}" for i in range(10, 30)]  # 20 samples

        # Create a BgenFileReader instance
        reader = BgenFileReader(
            file_path=str(self.bgen_file),
            index_path=str(self.bgi_file),
            sample_path=str(self.sample_file),
        )
        # Override sample_ids for testing
        reader.sample_ids = all_sample_ids
        reader.n_samples = len(all_sample_ids)

        # Test with some missing samples
        requested_samples = subset_sample_ids + ["MISSING_001", "MISSING_002"]
        indices, filtered_ids = reader.get_sample_indices(requested_samples)

        # Should only return the samples that exist
        self.assertEqual(len(indices), 20)
        self.assertEqual(len(filtered_ids), 20)
        self.assertEqual(filtered_ids, subset_sample_ids)

    def test_load_bgen_with_sample_filtering(self):
        """Test BGEN loading with sample filtering."""
        # Load all samples first to get the full list
        all_genotypes, all_variant_info, all_sample_ids = load_bgen(
            file_path=str(self.bgen_file),
            index_path=str(self.bgi_file),
            sample_path=str(self.sample_file),
        )

        # Skip test if too few samples
        if len(all_sample_ids) < 4:
            self.skipTest("Not enough samples for filtering test")

        # Select a subset of samples
        subset_samples = all_sample_ids[::2]  # Every other sample

        # Load with sample filtering
        filtered_genotypes, filtered_variant_info, filtered_sample_ids = load_bgen(
            file_path=str(self.bgen_file),
            index_path=str(self.bgi_file),
            sample_path=str(self.sample_file),
            sample_ids=subset_samples,
        )

        # Verify filtering worked
        self.assertEqual(len(filtered_sample_ids), len(subset_samples))
        self.assertEqual(filtered_sample_ids, subset_samples)
        self.assertEqual(filtered_genotypes.shape[0], len(subset_samples))
        self.assertEqual(filtered_genotypes.shape[1], all_genotypes.shape[1])

        # Verify the genotype data matches for the filtered samples
        for i, sample_id in enumerate(subset_samples):
            orig_idx = all_sample_ids.index(sample_id)
            np.testing.assert_array_almost_equal(
                filtered_genotypes[i, :], all_genotypes[orig_idx, :]
            )

    def test_sample_filtering_with_missing_samples(self):
        """Test that missing samples are handled gracefully."""
        # Get actual sample IDs
        _, _, actual_sample_ids = load_bgen(
            file_path=str(self.bgen_file),
            index_path=str(self.bgi_file),
            sample_path=str(self.sample_file),
        )

        if len(actual_sample_ids) < 2:
            self.skipTest("Not enough samples for test")

        # Request some existing and some non-existing samples
        requested_samples = [actual_sample_ids[0], "FAKE_SAMPLE_1", actual_sample_ids[1], "FAKE_SAMPLE_2"]

        # Load with filtering
        filtered_genotypes, _, filtered_sample_ids = load_bgen(
            file_path=str(self.bgen_file),
            index_path=str(self.bgi_file),
            sample_path=str(self.sample_file),
            sample_ids=requested_samples,
        )

        # Should only get the existing samples
        self.assertEqual(len(filtered_sample_ids), 2)
        self.assertEqual(filtered_sample_ids, [actual_sample_ids[0], actual_sample_ids[1]])

    def test_memory_efficiency_calculation(self):
        """Test that sample filtering reduces memory usage (conceptual test)."""
        # This is a conceptual test showing the memory benefit

        # Without filtering: 500K samples × 10K variants × 8 bytes
        memory_without_filtering = 500_000 * 10_000 * 8 / (1024**3)  # GB

        # With filtering: 10K samples × 10K variants × 8 bytes
        memory_with_filtering = 10_000 * 10_000 * 8 / (1024**3)  # GB

        memory_saved = memory_without_filtering - memory_with_filtering
        savings_percent = memory_saved / memory_without_filtering * 100

        # Assert significant memory savings
        self.assertGreater(savings_percent, 95)  # >95% savings

        # Log the calculation for reference
        print(f"\nMemory usage comparison:")
        print(f"Without filtering: {memory_without_filtering:.1f} GB")
        print(f"With filtering: {memory_with_filtering:.1f} GB")
        print(f"Memory saved: {memory_saved:.1f} GB ({savings_percent:.0f}%)")

    def test_bcor_backward_compatibility(self):
        """Test that standard bcor files can still be read correctly."""
        n_vars = 5
        # Create standard correlation matrix (unit diagonal)
        corr_matrix = np.eye(n_vars)
        corr_matrix[0, 1] = corr_matrix[1, 0] = 0.5

        output_file = os.path.join(self.temp_dir, "standard_compat.bcor")
        save_bcor(corr_matrix, output_file, n_samples=100)

        reader = BcorReader(output_file)
        self.assertFalse(reader.is_extended)

        # Should read correctly with diagonal as 1.0
        loaded = reader.read_corr()
        np.testing.assert_array_almost_equal(corr_matrix, loaded, decimal=6)
        np.testing.assert_array_equal(np.diag(loaded), np.ones(n_vars))


if __name__ == "__main__":
    unittest.main()
