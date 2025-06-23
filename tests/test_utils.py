"""
Consolidated utility tests for the ldcov package.

This module combines tests for:
- Region parsing
- Variant filtering
- Z-file functionality
- Categorical utilities
"""

import unittest
import tempfile
import os
import pandas as pd

# import numpy as np  # Used indirectly by pandas and other functions
from pathlib import Path

from ldcov.utils.region_utils import parse_region
from ldcov.utils.variant_filter import load_variant_filter
from ldcov.utils.categorical_utils import one_hot_encode_categorical
from ldcov.io import load_bgen


class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""

    @classmethod
    def setUpClass(cls):
        """Set up test data."""
        cls.temp_dir = tempfile.mkdtemp(prefix="ldcov_test_utils_")

        # Get example data path
        cls.examples_dir = Path(__file__).parents[1] / "examples"
        cls.bgen_file = cls.examples_dir / "data" / "data.bgen"
        cls.bgi_file = cls.examples_dir / "data" / "data.bgen.bgi"

    @classmethod
    def tearDownClass(cls):
        """Clean up."""
        import shutil

        shutil.rmtree(cls.temp_dir)

    # ==================== Region Parsing Tests ====================

    def test_parse_region_basic(self):
        """Test basic region parsing."""
        chrom, (start, end) = parse_region("1:1000000-2000000")
        self.assertEqual(chrom, "1")
        self.assertEqual(start, 1000000)
        self.assertEqual(end, 2000000)

    def test_parse_region_with_chr_prefix(self):
        """Test region parsing with chr prefix."""
        chrom, (start, end) = parse_region("chr1:1000000-2000000")
        self.assertEqual(chrom, "chr1")
        self.assertEqual(start, 1000000)
        self.assertEqual(end, 2000000)

    def test_parse_region_padded_chromosome(self):
        """Test region parsing with zero-padded chromosome."""
        chrom, (start, end) = parse_region("01:1000000-2000000")
        self.assertEqual(chrom, "01")
        self.assertEqual(start, 1000000)
        self.assertEqual(end, 2000000)

    def test_parse_region_sex_chromosomes(self):
        """Test parsing sex chromosome regions."""
        # X chromosome
        chrom, (start, end) = parse_region("X:1000000-2000000")
        self.assertEqual(chrom, "X")

        # Y chromosome
        chrom, (start, end) = parse_region("Y:500000-1000000")
        self.assertEqual(chrom, "Y")

    def test_parse_region_invalid_format(self):
        """Test error handling for invalid region formats."""
        # Missing colon
        with self.assertRaises(ValueError):
            parse_region("1-1000000-2000000")

        # Missing hyphen
        with self.assertRaises(ValueError):
            parse_region("1:10000002000000")

        # Invalid positions
        with self.assertRaises(ValueError):
            parse_region("1:abc-def")

    def test_parse_region_edge_cases(self):
        """Test edge cases in region parsing."""
        # Single position (start = end)
        chrom, (start, end) = parse_region("1:1000000-1000000")
        self.assertEqual(start, end)

        # Large positions
        chrom, (start, end) = parse_region("1:200000000-300000000")
        self.assertEqual(start, 200000000)
        self.assertEqual(end, 300000000)

    # ==================== Z-file Tests ====================

    def test_load_z_file_basic(self):
        """Test basic Z-file loading."""
        # Create test Z-file with correct column names
        z_data = pd.DataFrame(
            {
                "rsid": ["rs1", "rs2"],
                "chromosome": ["1", "1"],
                "position": ["100", "200"],
                "allele1": ["A", "C"],
                "allele2": ["G", "T"],
            }
        )
        z_file = os.path.join(self.temp_dir, "test.z")
        z_data.to_csv(z_file, sep="\t", index=False)

        # Load Z-file
        variant_filter = load_variant_filter(z_file)

        # Check filter structure
        self.assertIsInstance(variant_filter, dict)
        self.assertEqual(len(variant_filter["rsids"]), 2)
        self.assertEqual(variant_filter["rsids"][0], "rs1")
        self.assertEqual(variant_filter["chromosome"], "1")

    def test_z_file_original_chromosome_format(self):
        """Test that Z-files preserve original chromosome format."""
        # Test chr1 format
        z_data1 = pd.DataFrame(
            {
                "rsid": ["rs1", "rs2"],
                "chromosome": ["chr1", "chr1"],
                "position": ["100", "200"],
                "allele1": ["A", "C"],
                "allele2": ["G", "T"],
            }
        )
        z_file1 = os.path.join(self.temp_dir, "test_chr1.z")
        z_data1.to_csv(z_file1, sep="\t", index=False)

        variant_filter1 = load_variant_filter(z_file1)
        self.assertEqual(variant_filter1["chromosome"], "chr1")  # Keeps original format

        # Test 01 format
        z_data2 = pd.DataFrame(
            {
                "rsid": ["rs3", "rs4"],
                "chromosome": ["01", "01"],
                "position": ["100", "200"],
                "allele1": ["A", "C"],
                "allele2": ["G", "T"],
            }
        )
        z_file2 = os.path.join(self.temp_dir, "test_01.z")
        z_data2.to_csv(z_file2, sep="\t", index=False)

        variant_filter2 = load_variant_filter(z_file2)
        self.assertEqual(variant_filter2["chromosome"], "01")  # Keeps original format

    def test_load_variant_filter(self):
        """Test loading variant filter from Z-file."""
        # Load actual variant info
        _, variant_info, _ = load_bgen(
            file_path=str(self.bgen_file),
            index_path=str(self.bgi_file),
        )

        # Create Z-file with subset of variants
        subset_variants = variant_info.iloc[:2]
        z_data = pd.DataFrame(
            {
                "rsid": subset_variants["rsid"].tolist(),
                "chromosome": subset_variants["chrom"].tolist(),
                "position": subset_variants["pos"].astype(str).tolist(),
                "allele1": subset_variants["ref"].tolist(),
                "allele2": subset_variants["alt"].tolist(),
            }
        )
        z_file = os.path.join(self.temp_dir, "test_filter.z")
        z_data.to_csv(z_file, sep="\t", index=False)

        # Load filter in one step
        variant_filter = load_variant_filter(z_file)

        # Check that filter is a dictionary with expected keys
        self.assertIsInstance(variant_filter, dict)
        self.assertIn("chromosome", variant_filter)
        self.assertIn("positions", variant_filter)
        self.assertIn("rsids", variant_filter)

        # Should have same number of positions as input
        self.assertEqual(len(variant_filter["positions"]), len(z_data))

    def test_z_file_ordering(self):
        """Test that Z-file requires sorted positions."""
        # Create Z-file with unsorted positions (should fail)
        z_data = pd.DataFrame(
            {
                "rsid": ["rs3", "rs1", "rs2"],
                "chromosome": ["01", "01", "01"],
                "position": ["300", "100", "200"],  # Not in position order
                "allele1": ["G", "A", "C"],
                "allele2": ["A", "G", "T"],
            }
        )
        z_file = os.path.join(self.temp_dir, "test_order.z")
        z_data.to_csv(z_file, sep="\t", index=False)

        # Should raise error for unsorted positions
        with self.assertRaises(ValueError) as cm:
            load_variant_filter(z_file)
        self.assertIn("not sorted", str(cm.exception))

    # ==================== Categorical Utils Tests ====================

    def test_one_hot_encode_simple(self):
        """Test simple one-hot encoding."""
        df = pd.DataFrame(
            {
                "numeric": [1.0, 2.0, 3.0],
                "category": ["A", "B", "A"],
            }
        )

        encoded = one_hot_encode_categorical(df)

        # Check structure
        self.assertIn("numeric", encoded.columns)
        # For binary variables, only one column is kept (the second alphabetically)
        self.assertIn("category_B", encoded.columns)
        self.assertNotIn("category_A", encoded.columns)  # Dropped to avoid collinearity
        self.assertNotIn("category", encoded.columns)

        # Check values
        self.assertEqual(encoded["category_B"].tolist(), [0, 1, 0])

    def test_one_hot_encode_binary(self):
        """Test one-hot encoding of binary variables."""
        df = pd.DataFrame(
            {
                "binary": ["yes", "no", "yes", "no"],
                "value": [1, 2, 3, 4],
            }
        )

        encoded = one_hot_encode_categorical(df)

        # Should only create one column for binary (keeps the second alphabetically)
        self.assertIn("binary_yes", encoded.columns)
        self.assertNotIn("binary_no", encoded.columns)  # Dropped

    def test_one_hot_encode_multiple_categories(self):
        """Test one-hot encoding with multiple categorical columns."""
        df = pd.DataFrame(
            {
                "cat1": ["A", "B", "C", "A"],
                "cat2": ["X", "Y", "X", "Y"],
                "numeric": [1, 2, 3, 4],
            }
        )

        encoded = one_hot_encode_categorical(df)

        # Check categories are encoded (first column dropped for each)
        self.assertIn("numeric", encoded.columns)
        # cat1: A dropped (first alphabetically), B and C kept
        self.assertNotIn("cat1_A", encoded.columns)
        self.assertIn("cat1_B", encoded.columns)
        self.assertIn("cat1_C", encoded.columns)
        # cat2: X dropped (first alphabetically), Y kept
        self.assertNotIn("cat2_X", encoded.columns)
        self.assertIn("cat2_Y", encoded.columns)

    def test_one_hot_encode_with_nan(self):
        """Test one-hot encoding with missing values."""
        df = pd.DataFrame(
            {
                "category": ["A", "B", None, "A"],
                "numeric": [1, 2, 3, 4],
            }
        )

        encoded = one_hot_encode_categorical(df)

        # NaN should be encoded as 0 in remaining one-hot columns
        # A is dropped, B is kept
        self.assertNotIn("category_A", encoded.columns)
        self.assertIn("category_B", encoded.columns)
        self.assertEqual(encoded["category_B"].iloc[2], 0)

    def test_one_hot_encode_numeric_strings(self):
        """Test that numeric strings are treated as categorical."""
        df = pd.DataFrame(
            {
                "str_numeric": ["1", "2", "3"],
                "actual_string": ["A", "B", "C"],
            }
        )

        encoded = one_hot_encode_categorical(df)

        # Numeric strings are treated as categorical (object dtype)
        # str_numeric: 1 dropped, 2 and 3 kept
        self.assertNotIn("str_numeric_1", encoded.columns)
        self.assertIn("str_numeric_2", encoded.columns)
        self.assertIn("str_numeric_3", encoded.columns)

        # Actual strings: A dropped, B and C kept
        self.assertNotIn("actual_string_A", encoded.columns)
        self.assertIn("actual_string_B", encoded.columns)
        self.assertIn("actual_string_C", encoded.columns)

    def test_one_hot_encode_edge_cases(self):
        """Test edge cases in one-hot encoding."""
        # Single unique value
        df1 = pd.DataFrame({"single": ["A", "A", "A"]})
        encoded1 = one_hot_encode_categorical(df1)
        self.assertIn("single_A", encoded1.columns)

        # Empty dataframe
        df2 = pd.DataFrame()
        encoded2 = one_hot_encode_categorical(df2)
        self.assertEqual(len(encoded2.columns), 0)

        # All numeric
        df3 = pd.DataFrame({"num1": [1, 2, 3], "num2": [4, 5, 6]})
        encoded3 = one_hot_encode_categorical(df3)
        self.assertEqual(list(encoded3.columns), ["num1", "num2"])


if __name__ == "__main__":
    unittest.main()
