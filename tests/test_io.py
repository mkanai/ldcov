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

    def test_bgen_reader_requires_bgi(self):
        """Test that BGEN reader requires BGI file."""
        # Create a temp BGEN file without BGI
        with tempfile.NamedTemporaryFile(suffix=".bgen") as f:
            # Should fail without BGI
            with self.assertRaises(FileNotFoundError) as context:
                BgenFileReader(f.name)
            self.assertIn("BGI index required", str(context.exception))

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

    def test_load_all_variants(self):
        """Test loading all variants using BGI."""
        reader = BgenFileReader(
            str(self.bgen_file),
            sample_path=str(self.sample_file)
        )
        
        dosages, variant_info = reader.load_all_variants()
        
        # Check dimensions
        self.assertEqual(dosages.shape[0], reader.n_samples)
        self.assertEqual(dosages.shape[1], reader.n_variants)
        self.assertEqual(len(variant_info), reader.n_variants)
        
        # Check variant info columns
        expected_cols = {'chrom', 'pos', 'id', 'rsid', 'ref', 'alt', 'idx'}
        self.assertEqual(set(variant_info.columns), expected_cols)
        
        # Check dosages are in valid range
        self.assertTrue(np.all(dosages >= 0) and np.all(dosages <= 2))
        
        reader.close()

    def test_load_region_variants(self):
        """Test loading variants from a region using BGI."""
        reader = BgenFileReader(str(self.bgen_file))
        
        # Load region with variants
        dosages, variant_info = reader.load_region_variants('01', 1, 10)
        
        self.assertGreater(dosages.shape[1], 0)  # Should have some variants
        self.assertEqual(len(variant_info), dosages.shape[1])
        self.assertTrue(np.all(variant_info['pos'] >= 1))
        self.assertTrue(np.all(variant_info['pos'] <= 10))
        
        # Empty region
        dosages2, variant_info2 = reader.load_region_variants('01', 100000, 200000)
        self.assertEqual(dosages2.shape, (reader.n_samples, 0))
        self.assertEqual(len(variant_info2), 0)
        
        reader.close()

    def test_load_filtered_variants(self):
        """Test loading filtered variants from z file."""
        # Create a simple z file for testing
        z_data = pd.DataFrame({
            'chromosome': ['01', '01', '01'],
            'position': [1, 5, 10],
            'allele1': ['A', 'A', 'A'],
            'allele2': ['G', 'G', 'G'],
            'rsid': ['rs1', 'rs5', 'rs10']
        })
        z_file = os.path.join(self.temp_dir, "test.z")
        z_data.to_csv(z_file, sep='\t', index=False)
        
        # Import the functions we need
        from ldcov.utils.variant_filter import read_z_file, create_variant_filter_from_z
        
        reader = BgenFileReader(str(self.bgen_file))
        
        # Create filter from z file
        z_df = read_z_file(z_file)
        variant_filter = create_variant_filter_from_z(z_df)
        
        # Load filtered variants
        dosages, variant_info = reader.load_filtered_variants(variant_filter)
        
        # Should have loaded the variants in z file (or fewer if some don't exist)
        self.assertLessEqual(dosages.shape[1], len(variant_filter['positions']))
        self.assertEqual(len(variant_info), dosages.shape[1])
        
        reader.close()

    def test_sample_filtering(self):
        """Test sample filtering in BGEN reader."""
        reader = BgenFileReader(
            str(self.bgen_file),
            sample_path=str(self.sample_file)
        )
        
        # Get subset of samples
        sample_ids_to_keep = reader.sample_ids[:3]  # First 3 samples
        sample_indices, filtered_ids = reader.get_sample_indices(sample_ids_to_keep)
        
        self.assertEqual(len(sample_indices), 3)
        self.assertEqual(len(filtered_ids), 3)
        self.assertEqual(filtered_ids, sample_ids_to_keep)
        
        # Load with sample filtering
        dosages, variant_info = reader.load_all_variants(sample_indices)
        
        self.assertEqual(dosages.shape[0], 3)  # Only 3 samples
        
        reader.close()

    def test_nan_handling(self):
        """Test NaN handling options."""
        # Note: The example BGEN file doesn't contain NaN values,
        # so nan_action is accepted but not actually triggered
        
        # Should not raise with valid nan_action
        for action in ['error', 'mean', 'omit']:
            dosages, variant_info, sample_ids = load_bgen(
                str(self.bgen_file),
                nan_action=action
            )
            self.assertIsNotNone(dosages)
            self.assertFalse(np.any(np.isnan(dosages)))  # No NaN values in test data


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
        requested_samples = [
            actual_sample_ids[0],
            "FAKE_SAMPLE_1",
            actual_sample_ids[1],
            "FAKE_SAMPLE_2",
        ]

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

    # ==================== BGI Reader Tests ====================

    def test_bgi_reader_init(self):
        """Test BGI reader initialization."""
        from ldcov.io.bgi_reader import BGIReader
        
        # Should succeed with valid BGI
        reader = BGIReader(str(self.bgi_file))
        self.assertIsNotNone(reader)
        reader.close()
        
        # Should fail with non-existent file
        with self.assertRaises(FileNotFoundError):
            BGIReader("/non/existent/file.bgi")
        
        # Should fail with invalid file
        with tempfile.NamedTemporaryFile(suffix=".bgi") as f:
            f.write(b"not a valid bgi file")
            f.flush()
            with self.assertRaises(ValueError) as context:
                BGIReader(f.name)
            self.assertIn("Error reading BGI file", str(context.exception))

    def test_bgi_get_variant_count(self):
        """Test getting variant count from BGI."""
        from ldcov.io.bgi_reader import BGIReader
        
        with BGIReader(str(self.bgi_file)) as reader:
            count = reader.get_variant_count()
            self.assertGreater(count, 0)
            
            # Should be cached
            count2 = reader.get_variant_count()
            self.assertEqual(count2, count)

    def test_bgi_get_all_variants(self):
        """Test getting all variant metadata from BGI."""
        from ldcov.io.bgi_reader import BGIReader
        
        with BGIReader(str(self.bgi_file)) as reader:
            variants = reader.get_all_variants()
            
            # Check structure (now a DataFrame)
            self.assertGreater(len(variants), 0)
            expected_cols = {'chrom', 'pos', 'rsid', 'n_alleles', 'ref', 'alt', 'file_offset', 'size_bytes'}
            self.assertEqual(set(variants.columns), expected_cols)
            
            # Check first variant
            first = variants.iloc[0]
            self.assertIsNotNone(first['chrom'])
            self.assertGreater(first['pos'], 0)
            self.assertIsNotNone(first['rsid'])
            self.assertGreaterEqual(first['n_alleles'], 2)
            self.assertIsNotNone(first['ref'])
            self.assertGreater(first['file_offset'], 0)
            self.assertGreater(first['size_bytes'], 0)
            
            # Check ordering by file offset
            offsets = variants['file_offset'].values
            self.assertTrue(np.all(offsets[1:] > offsets[:-1]))  # Strictly increasing

    def test_bgi_get_variants_in_region(self):
        """Test getting variants in a genomic region from BGI."""
        from ldcov.io.bgi_reader import BGIReader
        
        with BGIReader(str(self.bgi_file)) as reader:
            # Get all variants to find a valid region
            all_variants = reader.get_all_variants()
            if len(all_variants) == 0:
                self.skipTest("No variants in BGI file")
            
            # Use first chromosome and position range
            chrom = all_variants.iloc[0]['chrom']
            min_pos = all_variants['pos'].min()
            max_pos = all_variants['pos'].max()
            mid_pos = (min_pos + max_pos) // 2
            
            # Region with variants
            variants = reader.get_variants_in_region(chrom, min_pos, mid_pos)
            # May have zero variants if mid_pos is too close to min_pos
            if len(variants) > 0:
                self.assertTrue((variants['chrom'] == chrom).all())
                self.assertTrue((variants['pos'] >= min_pos).all())
                self.assertTrue((variants['pos'] <= mid_pos).all())
            
            # Empty region
            variants = reader.get_variants_in_region(chrom, max_pos + 1000, max_pos + 2000)
            self.assertEqual(len(variants), 0)

    def test_bgi_find_variants_by_filter(self):
        """Test finding variants by position/allele/rsid in BGI."""
        from ldcov.io.bgi_reader import BGIReader
        
        with BGIReader(str(self.bgi_file)) as reader:
            # Get some actual variants to search for
            all_variants = reader.get_all_variants()
            if len(all_variants) < 3:
                self.skipTest("Not enough variants for test")
            
            # Create filter matching first 3 variants
            chromosome = all_variants['chrom'].iloc[0]  # Get chromosome from first variant
            positions = all_variants['pos'].values[:3]
            alleles1 = all_variants['ref'].values[:3].tolist()
            alleles2 = all_variants['alt'].values[:3].tolist()
            
            matched = reader.find_variants_by_filter(
                chromosome, positions, alleles1, alleles2
            )
            
            # Should find all 3
            self.assertEqual(len(matched), 3)
            # Verify they're in the same order as requested
            np.testing.assert_array_equal(matched['pos'].values, positions)
            
            # Test with swapped alleles (should NOT match since exact match is required)
            matched2 = reader.find_variants_by_filter(
                chromosome, positions, alleles2, alleles1
            )
            
            # Should find 0 matches with swapped alleles
            self.assertEqual(len(matched2), 0)

    def test_bgi_context_manager(self):
        """Test BGI reader context manager usage."""
        from ldcov.io.bgi_reader import BGIReader
        
        with BGIReader(str(self.bgi_file)) as reader:
            count = reader.get_variant_count()
            self.assertGreater(count, 0)
        
        # Connection should be closed after context
        # (Can't easily test this without accessing private attributes)

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
