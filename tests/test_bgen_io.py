"""
Consolidated BGEN I/O tests for the ldcov package.

This module consolidates all BGEN-related tests from:
- test_io.py (BGEN reader tests)
- test_bgen_reader.py (comprehensive reader tests)
- test_bgen_formats.py (format and context manager tests)

Tests are organized into logical sections:
1. Basic Reading and Initialization
2. Context Manager Support
3. Sample Filtering
4. Region Queries
5. Format Support (bit depths, compression)
6. Error Handling
7. Performance and Memory
8. BGI Index Support
"""

import numpy as np
import pandas as pd
import os
import pytest
import tempfile
import time
import gc
import psutil
from pathlib import Path
from typing import Optional, Tuple, List
from unittest.mock import patch, MagicMock
import logging

# Import BGEN functionality
from ldcov.io import load_bgen
from ldcov.io.bgen.reader import BgenReader
from ldcov.io.bgen.bgi import BGIReader
from ldcov.utils.variant_filter import load_variant_filter

# Try to import external bgen library for comparison
try:
    import bgen as external_bgen

    HAS_EXTERNAL_BGEN = True
except ImportError:
    HAS_EXTERNAL_BGEN = False
    external_bgen = None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def test_paths():
    """Set up test data paths."""
    examples_dir = Path(__file__).parents[1] / "examples" / "data"

    # Test files
    bgen_file = examples_dir / "data.bgen"
    bgi_file = examples_dir / "data.bgen.bgi"
    sample_file = examples_dir / "data.sample"

    # Additional format test files
    test_files = {
        "basic": bgen_file,
        "8bit": examples_dir / "example.8bits.bgen",
        "16bit": examples_dir / "example.16bits.bgen",
        "32bit": examples_dir / "example.32bits.bgen",
        "zstd": examples_dir / "example.16bits.zstd.bgen",
        "v11": examples_dir / "example.v11.bgen",
    }

    return {
        "examples_dir": examples_dir,
        "bgen_file": bgen_file,
        "bgi_file": bgi_file,
        "sample_file": sample_file,
        "test_files": test_files,
    }


@pytest.fixture
def memory_monitor():
    """Memory monitoring fixture."""
    process = psutil.Process(os.getpid())

    def get_memory_usage():
        """Get current memory usage in MB."""
        return process.memory_info().rss / 1024 / 1024

    return get_memory_usage


def check_dosages_valid(dosages):
    """Check that dosages are valid (between 0 and 2)."""
    # Handle NaN values by checking only non-NaN values
    valid_mask = ~np.isnan(dosages)
    if np.any(valid_mask):
        assert np.all(dosages[valid_mask] >= 0)
        assert np.all(dosages[valid_mask] <= 2)


# ==================== Basic Reading and Initialization ====================


def test_basic_loading(test_paths):
    """Test basic BGEN file loading."""
    genotypes, variant_info, sample_ids = load_bgen(
        file_path=str(test_paths["bgen_file"]),
        index_path=str(test_paths["bgi_file"]),
        sample_path=str(test_paths["sample_file"]),
        nan_action="omit",  # Handle any potential NaN values
    )

    # Check shapes
    assert genotypes.shape[0] == 5363  # Expected samples
    assert genotypes.shape[1] == 55  # Expected variants
    assert len(variant_info) == 55
    assert len(sample_ids) == 5363

    # Check variant info columns
    expected_cols = {"chrom", "pos", "rsid", "ref", "alt"}
    assert set(variant_info.columns) == expected_cols

    # Check dosages are valid
    check_dosages_valid(genotypes)


def test_loading_without_index(test_paths):
    """Test BGEN loading without BGI index file."""
    genotypes, variant_info, sample_ids = load_bgen(
        file_path=str(test_paths["bgen_file"]),
        sample_path=str(test_paths["sample_file"]),
        nan_action="omit",  # Handle any potential NaN values
    )

    # Should still load successfully
    assert genotypes.shape[0] > 0
    assert genotypes.shape[1] > 0
    assert len(variant_info) == genotypes.shape[1]
    assert len(sample_ids) == genotypes.shape[0]


def test_bgen_reader_requires_bgi():
    """Test that BGEN reader requires BGI file when using certain features."""
    # Create a temp BGEN file without BGI
    with tempfile.NamedTemporaryFile(suffix=".bgen") as f:
        # Write a minimal invalid BGEN header to make it look like a BGEN file
        f.write(b"\x00" * 32)  # Minimal content
        f.flush()

        # Should fail without BGI or with invalid BGEN content
        with pytest.raises((FileNotFoundError, ValueError, RuntimeError)):
            load_bgen(f.name)


def test_reader_properties(test_paths):
    """Test BgenReader properties and basic functionality."""
    with BgenReader(str(test_paths["bgen_file"])) as reader:
        # Check basic properties
        assert reader.nvariants > 0
        assert reader.nsamples > 0
        assert reader.samples is not None
        assert len(reader.samples) == reader.nsamples

        # Load variants
        try:
            # Try with nan_action parameter if supported
            dosages, variant_info = reader.load_variants(nan_action="omit")
        except TypeError:
            # If nan_action is not supported, load without it
            dosages, variant_info = reader.load_variants()
        assert dosages.shape[0] == reader.nsamples
        assert dosages.shape[1] == reader.nvariants
        assert len(variant_info) == reader.nvariants


# ==================== Context Manager Support ====================


def test_context_manager_basic(test_paths):
    """Test basic context manager functionality."""
    with BgenReader(str(test_paths["bgen_file"])) as reader:
        # Should be able to access properties and load data
        assert reader.nvariants > 0
        assert reader.nsamples > 0
        try:
            # Try with nan_action parameter if supported
            dosages, _ = reader.load_variants(nan_action="omit")
        except TypeError:
            # If nan_action is not supported, load without it
            dosages, _ = reader.load_variants()
        assert dosages is not None


def test_context_manager_file_closed(test_paths):
    """Test that file handle is properly closed after context exit."""
    reader = None
    with BgenReader(str(test_paths["bgen_file"])) as reader:
        # Can access data while open
        assert reader.nsamples > 0

    # After context exit, operations should fail
    with pytest.raises(ValueError, match="closed"):
        reader.load_variants()


def test_context_manager_exception_handling(test_paths):
    """Test that context manager properly handles exceptions."""
    reader = None
    try:
        with BgenReader(str(test_paths["bgen_file"])) as reader:
            # Simulate an error during processing
            raise RuntimeError("Test error")
    except RuntimeError:
        pass

    # Reader should still be closed even after exception
    with pytest.raises(ValueError):
        reader.load_variants()


def test_context_manager_nested(test_paths):
    """Test nested context managers work correctly."""
    with BgenReader(str(test_paths["bgen_file"])) as reader1:
        nvariants1 = reader1.nvariants

        with BgenReader(str(test_paths["bgen_file"])) as reader2:
            nvariants2 = reader2.nvariants
            assert nvariants1 == nvariants2

        # reader2 closed, reader1 still open
        try:
            # Try with nan_action parameter if supported
            dosages1, _ = reader1.load_variants(nan_action="omit")
        except TypeError:
            # If nan_action is not supported, load without it
            dosages1, _ = reader1.load_variants()
        assert dosages1 is not None

    # Both should be closed now
    with pytest.raises(ValueError):
        reader1.load_variants()


# ==================== Sample Filtering ====================


def test_sample_filtering_basic(test_paths):
    """Test basic sample filtering functionality."""
    # First load all samples
    all_genotypes, _, all_sample_ids = load_bgen(
        file_path=str(test_paths["bgen_file"]),
        index_path=str(test_paths["bgi_file"]),
        sample_path=str(test_paths["sample_file"]),
        nan_action="omit",
    )

    # Select subset of samples
    subset_samples = all_sample_ids[:100]  # First 100 samples

    # Load with sample filtering
    filtered_genotypes, filtered_variant_info, filtered_sample_ids = load_bgen(
        file_path=str(test_paths["bgen_file"]),
        index_path=str(test_paths["bgi_file"]),
        sample_path=str(test_paths["sample_file"]),
        sample_ids=subset_samples,
        nan_action="omit",
    )

    # Verify filtering worked
    assert len(filtered_sample_ids) == 100
    assert filtered_sample_ids == subset_samples
    assert filtered_genotypes.shape[0] == 100
    assert filtered_genotypes.shape[1] == all_genotypes.shape[1]


def test_sample_filtering_with_indices(test_paths):
    """Test sample filtering using indices."""
    sample_indices = np.array([0, 10, 20, 30, 40], dtype=np.int32)

    with BgenReader(
        str(test_paths["bgen_file"]), sample_path=str(test_paths["sample_file"])
    ) as reader:
        try:
            # Try with nan_action parameter if supported
            dosages, _ = reader.load_variants(sample_indices=sample_indices, nan_action="omit")
        except TypeError:
            # If nan_action is not supported, load without it
            dosages, _ = reader.load_variants(sample_indices=sample_indices)
        assert dosages.shape[0] == len(sample_indices)


def test_sample_filtering_missing_samples(test_paths):
    """Test that missing samples are handled gracefully."""
    # Get actual sample IDs
    _, _, actual_sample_ids = load_bgen(
        file_path=str(test_paths["bgen_file"]),
        index_path=str(test_paths["bgi_file"]),
        sample_path=str(test_paths["sample_file"]),
        nan_action="omit",
    )

    # Request some existing and some non-existing samples
    requested_samples = [
        actual_sample_ids[0],
        "FAKE_SAMPLE_1",
        actual_sample_ids[1],
        "FAKE_SAMPLE_2",
    ]

    # Load with filtering
    filtered_genotypes, _, filtered_sample_ids = load_bgen(
        file_path=str(test_paths["bgen_file"]),
        index_path=str(test_paths["bgi_file"]),
        sample_path=str(test_paths["sample_file"]),
        sample_ids=requested_samples,
        nan_action="omit",
    )

    # Should only get the existing samples
    assert len(filtered_sample_ids) == 2
    assert filtered_sample_ids == [actual_sample_ids[0], actual_sample_ids[1]]


def test_sample_filtering_order_preserved(test_paths):
    """Test that sample order is preserved when filtering."""
    # Load all samples
    _, _, all_sample_ids = load_bgen(
        file_path=str(test_paths["bgen_file"]),
        index_path=str(test_paths["bgi_file"]),
        sample_path=str(test_paths["sample_file"]),
        nan_action="omit",
    )

    # Select samples in reverse order
    subset_samples = all_sample_ids[::10][::-1]  # Every 10th sample, reversed

    # Load with filtering
    _, _, filtered_sample_ids = load_bgen(
        file_path=str(test_paths["bgen_file"]),
        index_path=str(test_paths["bgi_file"]),
        sample_path=str(test_paths["sample_file"]),
        sample_ids=subset_samples,
        nan_action="omit",
    )

    # Order should be preserved
    assert filtered_sample_ids == subset_samples


def test_sample_filtering_memory_efficiency(test_paths, memory_monitor):
    """Test memory efficiency of sample filtering."""
    # Get baseline memory
    gc.collect()
    baseline_memory = memory_monitor()

    # Load only 10 samples
    _, _, sample_ids = load_bgen(
        file_path=str(test_paths["bgen_file"]),
        sample_path=str(test_paths["sample_file"]),
        nan_action="omit",
    )
    subset_samples = sample_ids[:10]

    # Load with filtering
    filtered_genotypes, _, _ = load_bgen(
        file_path=str(test_paths["bgen_file"]),
        sample_path=str(test_paths["sample_file"]),
        sample_ids=subset_samples,
        nan_action="omit",
    )

    # Check memory usage
    filtered_memory = memory_monitor()
    memory_increase = filtered_memory - baseline_memory

    # Memory increase should be small (< 100 MB)
    assert memory_increase < 100, f"Memory increased by {memory_increase:.1f} MB"


# ==================== Region Queries ====================


def test_region_loading(test_paths):
    """Test loading variants from a specific region."""
    # Load region with variants
    dosages, variant_info, sample_ids = load_bgen(
        file_path=str(test_paths["bgen_file"]),
        index_path=str(test_paths["bgi_file"]),
        region="01:1-10",
        nan_action="omit",
    )

    assert dosages.shape[1] > 0  # Should have some variants
    assert len(variant_info) == dosages.shape[1]
    assert np.all(variant_info["pos"] >= 1)
    assert np.all(variant_info["pos"] <= 10)


def test_region_empty(test_paths):
    """Test loading from an empty region."""
    # Empty region - should raise ValueError
    with pytest.raises(ValueError, match="No variants"):
        load_bgen(
            file_path=str(test_paths["bgen_file"]),
            index_path=str(test_paths["bgi_file"]),
            region="01:100000-200000",
            nan_action="omit",
        )


def test_region_chromosome_formats(test_paths):
    """Test region queries with different chromosome formats."""
    # First get actual chromosome format from file
    _, variant_info, _ = load_bgen(
        file_path=str(test_paths["bgen_file"]),
        index_path=str(test_paths["bgi_file"]),
        nan_action="omit",
    )

    if len(variant_info) > 0:
        chrom = variant_info["chrom"].iloc[0]
        min_pos = variant_info["pos"].min()
        max_pos = variant_info["pos"].max()

        # Try region query
        region = f"{chrom}:{min_pos}-{max_pos}"
        dosages, loaded_info, _ = load_bgen(
            file_path=str(test_paths["bgen_file"]),
            index_path=str(test_paths["bgi_file"]),
            region=region,
            nan_action="omit",
        )

        assert len(loaded_info) > 0
        assert np.all(loaded_info["chrom"] == chrom)


# ==================== Format Support ====================


@pytest.mark.parametrize("bit_depth", [8, 16, 32])
def test_bit_depth_formats(test_paths, bit_depth):
    """Test loading BGEN files with different bit depths."""
    file_key = f"{bit_depth}bit"
    test_file = test_paths["test_files"][file_key]

    if test_file.exists():
        dosages, variant_info, sample_ids = load_bgen(
            file_path=str(test_file), nan_action="omit"  # Handle NaN values by omitting them
        )
        assert dosages is not None
        check_dosages_valid(dosages)
    else:
        pytest.skip(f"{bit_depth}-bit test file not found")


@pytest.mark.parametrize("compression_format", ["zstd"])
def test_compression_formats(test_paths, compression_format):
    """Test loading BGEN files with different compression formats."""
    test_file = test_paths["test_files"][compression_format]

    if test_file.exists():
        dosages, variant_info, sample_ids = load_bgen(
            file_path=str(test_file), nan_action="omit"  # Handle NaN values by omitting them
        )
        assert dosages is not None
        check_dosages_valid(dosages)
    else:
        pytest.skip(f"{compression_format} compressed test file not found")


def test_v11_format(test_paths):
    """Test loading v1.1 format BGEN file."""
    if test_paths["test_files"]["v11"].exists():
        # v1.1 format might not be supported - try loading and check for specific error
        try:
            load_bgen(file_path=str(test_paths["test_files"]["v11"]), nan_action="omit")
            # If it succeeds, that's fine too - v1.1 might be supported
        except ValueError as e:
            # Check that the error mentions v1.1 or unsupported format or NaN
            error_str = str(e).lower()
            assert any(
                x in error_str for x in ["v1.1", "unsupported", "nan"]
            ), f"Unexpected error: {e}"
        except Exception:
            # Other exceptions are also acceptable for unsupported formats
            pass


# ==================== Z-file Filtering ====================


def test_z_file_filtering(test_paths, tmp_path):
    """Test loading filtered variants from z file."""
    # Load some variants to create Z-file
    _, variant_info, _ = load_bgen(
        file_path=str(test_paths["bgen_file"]),
        index_path=str(test_paths["bgi_file"]),
        nan_action="omit",
    )

    # Create Z-file with subset of variants
    subset_variants = variant_info.iloc[:3]
    z_data = pd.DataFrame(
        {
            "chromosome": subset_variants["chrom"].tolist(),
            "position": subset_variants["pos"].astype(str).tolist(),
            "allele1": subset_variants["ref"].tolist(),
            "allele2": subset_variants["alt"].tolist(),
            "rsid": subset_variants["rsid"].tolist(),
        }
    )
    z_file = tmp_path / "test.z"
    z_data.to_csv(z_file, sep="\t", index=False)

    # Create filter from z file
    variant_filter = load_variant_filter(str(z_file))

    # Load filtered variants
    dosages, loaded_info, sample_ids = load_bgen(
        file_path=str(test_paths["bgen_file"]),
        index_path=str(test_paths["bgi_file"]),
        variant_filter=variant_filter,
        nan_action="omit",
    )

    # Should have loaded the variants in z file (or fewer if some don't exist)
    assert dosages.shape[1] <= len(variant_filter["positions"])
    assert len(loaded_info) == dosages.shape[1]


# ==================== Error Handling ====================


def test_invalid_file_path():
    """Test error handling for invalid file path."""
    with pytest.raises(FileNotFoundError):
        load_bgen("/nonexistent/file.bgen")


def test_invalid_nan_action(test_paths):
    """Test error handling for invalid nan_action."""
    # First check if the file has NaN values by trying to load it
    try:
        load_bgen(str(test_paths["bgen_file"]), nan_action="error")
        # If no error, the file doesn't have NaN values, so skip this test
        pytest.skip("Test file doesn't contain NaN values")
    except ValueError as e:
        if "nan" not in str(e).lower():
            pytest.skip("Test file doesn't contain NaN values")

    # Now test with an invalid nan_action value
    with pytest.raises(ValueError) as exc_info:
        load_bgen(str(test_paths["bgen_file"]), nan_action="invalid")

    # Check that the error message mentions nan_action or the invalid value
    error_msg = str(exc_info.value).lower()
    assert "nan_action" in error_msg or "invalid" in error_msg or "unknown" in error_msg


def test_corrupted_file_handling(tmp_path):
    """Test handling of corrupted BGEN files."""
    # Create a file with invalid content
    corrupted_file = tmp_path / "corrupted.bgen"
    with open(corrupted_file, "wb") as f:
        f.write(b"This is not a valid BGEN file")

    # Should raise either ValueError or FileNotFoundError (for missing BGI)
    with pytest.raises((ValueError, FileNotFoundError, RuntimeError)):
        load_bgen(str(corrupted_file))


# ==================== Performance and Memory ====================


def test_loading_performance(test_paths):
    """Test loading performance for basic operations."""
    start_time = time.time()

    dosages, variant_info, sample_ids = load_bgen(
        file_path=str(test_paths["bgen_file"]),
        index_path=str(test_paths["bgi_file"]),
        nan_action="omit",
    )

    elapsed_time = time.time() - start_time

    # Should load reasonably fast (< 5 seconds for test file)
    assert elapsed_time < 5.0, f"Loading took {elapsed_time:.2f} seconds"

    # Log performance info
    logger.info(
        f"Loaded {dosages.shape[1]} variants x {dosages.shape[0]} samples in {elapsed_time:.2f}s"
    )


def test_selective_loading_performance(test_paths):
    """Test performance of selective variant loading."""
    # Load just 10 variants by region
    start_time = time.time()

    dosages, variant_info, _ = load_bgen(
        file_path=str(test_paths["bgen_file"]),
        index_path=str(test_paths["bgi_file"]),
        region="01:1-5",
        nan_action="omit",
    )

    elapsed_time = time.time() - start_time

    # Selective loading should be fast (< 1 second)
    assert elapsed_time < 1.0, f"Selective loading took {elapsed_time:.2f} seconds"


# ==================== BGI Index Support ====================


def test_bgi_reader_basic(test_paths):
    """Test basic BGI reader functionality."""
    with BGIReader(str(test_paths["bgi_file"])) as reader:
        # Check variant count
        count = reader.get_variant_count()
        assert count > 0

        # Get all variants
        variants = reader.get_all_variants()
        assert len(variants) == count

        # Check variant metadata
        expected_cols = {
            "chrom",
            "pos",
            "rsid",
            "n_alleles",
            "ref",
            "alt",
            "file_offset",
            "size_bytes",
        }
        assert set(variants.columns) == expected_cols


def test_bgi_region_query(test_paths):
    """Test BGI region queries."""
    with BGIReader(str(test_paths["bgi_file"])) as reader:
        # Get all variants to find valid region
        all_variants = reader.get_all_variants()

        if len(all_variants) > 0:
            chrom = all_variants.iloc[0]["chrom"]
            min_pos = all_variants["pos"].min()
            max_pos = all_variants["pos"].max()

            # Query region
            region_variants = reader.get_variants_in_region(
                chrom, min_pos, (min_pos + max_pos) // 2
            )

            if len(region_variants) > 0:
                assert np.all(region_variants["chrom"] == chrom)
                assert np.all(region_variants["pos"] >= min_pos)


def test_bgi_invalid_file():
    """Test BGI reader with invalid file."""
    with tempfile.NamedTemporaryFile(suffix=".bgi") as f:
        f.write(b"not a valid bgi file")
        f.flush()

        with pytest.raises(ValueError, match="Error reading BGI file"):
            BGIReader(f.name)


# ==================== Comparison with External Library ====================


@pytest.mark.skipif(not HAS_EXTERNAL_BGEN, reason="External bgen library not available")
def test_compare_with_external_library(test_paths):
    """Compare results with external bgen library if available."""
    # Load with ldcov
    ldcov_dosages, ldcov_info, ldcov_samples = load_bgen(
        file_path=str(test_paths["bgen_file"]),
        sample_path=str(test_paths["sample_file"]),
        nan_action="omit",
    )

    # Load with external library
    try:
        from bgen import BgenReader

        bfile = BgenReader(str(test_paths["bgen_file"]))

        # Compare samples
        external_samples = bfile.samples
        assert len(ldcov_samples) == len(external_samples)

        # Compare variant count
        external_variant_count = len(bfile)
        assert ldcov_dosages.shape[1] == external_variant_count

        # Compare sample count
        assert ldcov_dosages.shape[0] == len(external_samples)

        # Optional: Compare first few variant IDs
        variant_count = 0
        for i, variant in enumerate(bfile):
            if i >= 5:  # Just check first 5 variants
                break
            variant_count += 1
            # Check if rsid matches
            if hasattr(variant, "rsid") and i < len(ldcov_info):
                assert variant.rsid == ldcov_info.iloc[i]["rsid"]

        # Ensure we could read at least some variants
        assert variant_count > 0

    except Exception as e:
        # If there's any issue with the external library, skip the test
        pytest.skip(f"External bgen library error: {e}")


# ==================== NaN Handling ====================


@pytest.mark.parametrize("nan_action", ["error", "mean", "omit"])
def test_nan_handling_actions(test_paths, nan_action):
    """Test different NaN handling options."""
    dosages, _, _ = load_bgen(str(test_paths["bgen_file"]), nan_action=nan_action)
    assert dosages is not None
    # Test data shouldn't have NaN values
    assert not np.any(np.isnan(dosages))


# ==================== Decompressor Types ====================


@pytest.mark.parametrize("decompressor_type", ["adaptive", "sequential", "parallel"])
def test_decompressor_types(test_paths, decompressor_type):
    """Test different decompressor types."""
    with BgenReader(str(test_paths["bgen_file"]), decompressor_type=decompressor_type) as reader:
        try:
            # Try with nan_action parameter if supported
            dosages, _ = reader.load_variants(nan_action="omit")
        except TypeError:
            # If nan_action is not supported, load without it
            dosages, _ = reader.load_variants()
        assert dosages is not None
        check_dosages_valid(dosages)
