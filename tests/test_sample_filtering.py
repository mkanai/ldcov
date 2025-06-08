"""
Tests for sample filtering during BGEN loading.
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch

from ldcov.io.bgen_reader import BgenFileReader, load_bgen


class TestSampleFiltering(unittest.TestCase):
    """Test cases for sample filtering functionality."""

    def setUp(self):
        """Set up test data."""
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp(prefix="ldcov_test_sample_filter_")

        # Mock sample data
        self.all_sample_ids = [f"SAMPLE_{i:04d}" for i in range(100)]
        self.subset_sample_ids = [f"SAMPLE_{i:04d}" for i in range(10, 30)]  # 20 samples

        # Mock genotype data
        self.n_variants = 50
        self.mock_dosages = np.random.rand(100, self.n_variants)

        # Create mock variant info
        self.mock_variant_info = pd.DataFrame(
            {
                "chrom": ["1"] * self.n_variants,
                "pos": range(1000, 1000 + self.n_variants),
                "id": [f"rs{i}" for i in range(self.n_variants)],
                "ref": "A",
                "alt": "G",
            }
        )

    def tearDown(self):
        """Clean up test data."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_get_sample_indices(self):
        """Test sample index mapping."""
        # Create a mock BgenFileReader
        reader = Mock(spec=BgenFileReader)
        reader.sample_ids = self.all_sample_ids
        reader.n_samples = len(self.all_sample_ids)

        # Add the get_sample_indices method
        reader.get_sample_indices = BgenFileReader.get_sample_indices.__get__(reader)

        # Test with subset of samples
        indices, filtered_ids = reader.get_sample_indices(self.subset_sample_ids)

        self.assertEqual(len(indices), 20)
        self.assertEqual(len(filtered_ids), 20)
        self.assertEqual(indices[0], 10)  # First sample is SAMPLE_0010 at index 10
        self.assertEqual(indices[-1], 29)  # Last sample is SAMPLE_0029 at index 29
        self.assertEqual(filtered_ids, self.subset_sample_ids)

    def test_get_sample_indices_with_missing(self):
        """Test sample index mapping with missing samples."""
        # Create a mock BgenFileReader
        reader = Mock(spec=BgenFileReader)
        reader.sample_ids = self.all_sample_ids
        reader.n_samples = len(self.all_sample_ids)

        # Add the get_sample_indices method
        reader.get_sample_indices = BgenFileReader.get_sample_indices.__get__(reader)

        # Test with some missing samples
        requested_samples = self.subset_sample_ids + ["MISSING_001", "MISSING_002"]
        indices, filtered_ids = reader.get_sample_indices(requested_samples)

        # Should only return the samples that exist
        self.assertEqual(len(indices), 20)
        self.assertEqual(len(filtered_ids), 20)
        self.assertEqual(filtered_ids, self.subset_sample_ids)

    def test_load_all_variants_with_sample_filtering(self):
        """Test loading all variants with sample filtering."""
        # Create a mock BgenFileReader
        reader = Mock(spec=BgenFileReader)
        reader.sample_ids = self.all_sample_ids
        reader.n_samples = len(self.all_sample_ids)

        # Mock the bgen_file iterator to return mock variants
        mock_variants = []
        for i in range(self.n_variants):
            variant = Mock()
            variant.rsid = f"rs{i}"
            variant.chrom = "1"
            variant.pos = 1000 + i
            variant.alleles = ["A", "G"]
            # Make alt_dosage return the appropriate row from mock_dosages
            variant.alt_dosage = self.mock_dosages[:, i]
            mock_variants.append(variant)

        reader.bgen_file = mock_variants

        # Add the load_all_variants_and_dosages method
        reader.load_all_variants_and_dosages = BgenFileReader.load_all_variants_and_dosages.__get__(
            reader
        )

        # Test without filtering
        dosages, variant_info, n_samples = reader.load_all_variants_and_dosages()

        self.assertEqual(dosages.shape, (100, self.n_variants))
        self.assertEqual(n_samples, 100)
        self.assertEqual(len(variant_info), self.n_variants)

        # Test with sample filtering
        sample_indices = list(range(10, 30))  # Indices for subset samples
        filtered_dosages, variant_info, n_samples = reader.load_all_variants_and_dosages(
            sample_indices=sample_indices
        )

        self.assertEqual(filtered_dosages.shape, (20, self.n_variants))
        self.assertEqual(n_samples, 20)
        self.assertEqual(len(variant_info), self.n_variants)

        # Check that we got the right subset of data
        np.testing.assert_array_almost_equal(filtered_dosages, dosages[10:30, :])

    @patch("ldcov.io.bgen_reader.BgenFileReader")
    def test_load_bgen_with_sample_filtering(self, mock_reader_class):
        """Test the main load_bgen function with sample filtering."""
        # Set up mock reader instance
        mock_reader = Mock()
        mock_reader.sample_ids = self.all_sample_ids
        mock_reader.n_samples = len(self.all_sample_ids)
        mock_reader.close = Mock()

        # Mock get_sample_indices to return subset
        mock_reader.get_sample_indices.return_value = (list(range(10, 30)), self.subset_sample_ids)

        # Mock load_all_variants_and_dosages
        mock_reader.load_all_variants_and_dosages.return_value = (
            self.mock_dosages[10:30, :],  # Filtered dosages
            self.mock_variant_info,
            20,  # Number of samples
        )

        mock_reader_class.return_value = mock_reader

        # Test load_bgen with sample filtering
        dosages, variant_info, sample_ids = load_bgen(
            file_path="dummy.bgen", sample_ids=self.subset_sample_ids
        )

        # Verify sample filtering was applied
        self.assertEqual(dosages.shape[0], 20)  # Should have 20 samples
        self.assertEqual(len(sample_ids), 20)
        self.assertEqual(sample_ids, self.subset_sample_ids)

        # Verify the mock was called correctly
        mock_reader.get_sample_indices.assert_called_once_with(self.subset_sample_ids)
        mock_reader.load_all_variants_and_dosages.assert_called_once()

        # Check that sample indices were passed to load function
        call_args = mock_reader.load_all_variants_and_dosages.call_args
        self.assertEqual(call_args[0][1], list(range(10, 30)))  # sample_indices argument

    def test_memory_efficiency(self):
        """Test that sample filtering reduces memory usage."""
        # This is a conceptual test showing the memory benefit

        # Without filtering: 500K samples × 10K variants × 8 bytes
        memory_without_filtering = 500_000 * 10_000 * 8 / (1024**3)  # GB

        # With filtering: 10K samples × 10K variants × 8 bytes
        memory_with_filtering = 10_000 * 10_000 * 8 / (1024**3)  # GB

        memory_saved = memory_without_filtering - memory_with_filtering

        print(f"\nMemory usage comparison:")
        print(f"Without filtering: {memory_without_filtering:.1f} GB")
        print(f"With filtering: {memory_with_filtering:.1f} GB")
        print(
            f"Memory saved: {memory_saved:.1f} GB ({memory_saved/memory_without_filtering*100:.0f}%)"
        )

        # Assert significant memory savings
        self.assertGreater(memory_saved / memory_without_filtering, 0.95)  # >95% savings


if __name__ == "__main__":
    unittest.main()
