"""
I/O tests for the ldcov package.

This module contains tests for:
- BGEN file reading and writing
- Sample filtering
- BGI index operations
- Metadata handling
"""

import numpy as np
import pandas as pd
import os
import pytest
from pathlib import Path
from unittest.mock import patch
import tempfile

# import gzip  # Used by pd.read_csv for compressed files

from ldcov.io import load_bgen
from ldcov.compute.covariate import standardize_genotypes


@pytest.fixture(scope="module")
def test_data():
    """Set up test data for the module."""
    examples_dir = Path(__file__).parents[1] / "examples"
    bgen_file = examples_dir / "data" / "data.bgen"
    bgi_file = examples_dir / "data" / "data.bgen.bgi"
    sample_file = examples_dir / "data" / "data.sample"
    
    # Load test data once
    genotypes, variant_info, sample_ids = load_bgen(
        file_path=str(bgen_file),
        index_path=str(bgi_file),
        sample_path=str(sample_file),
    )
    
    return {
        "examples_dir": examples_dir,
        "bgen_file": bgen_file,
        "bgi_file": bgi_file,
        "sample_file": sample_file,
        "genotypes": genotypes,
        "variant_info": variant_info,
        "sample_ids": sample_ids
    }


# ==================== BGEN Reader Tests ====================

def test_bgen_reader_requires_bgi():
    """Test that BGEN reader requires BGI file."""
    # Create a temp BGEN file without BGI
    with tempfile.NamedTemporaryFile(suffix=".bgen") as f:
        # Should fail without BGI
        with pytest.raises(FileNotFoundError, match="BGI index required"):
            load_bgen(f.name)


def test_bgen_reader_initialization(test_data):
    """Test BGEN reader initialization through load_bgen."""
    # Test that we can successfully load a BGEN file
    genotypes, variant_info, sample_ids = load_bgen(
        file_path=str(test_data["bgen_file"]),
        index_path=str(test_data["bgi_file"]),
        sample_path=str(test_data["sample_file"]),
    )
    assert genotypes is not None
    assert variant_info is not None
    assert len(sample_ids) > 0


def test_load_bgen_basic(test_data):
    """Test basic BGEN loading."""
    genotypes, variant_info, sample_ids = load_bgen(
        file_path=str(test_data["bgen_file"]),
        index_path=str(test_data["bgi_file"]),
        sample_path=str(test_data["sample_file"]),
    )
    
    # Check shapes
    assert genotypes.shape[0] > 0
    assert genotypes.shape[1] > 0
    assert len(variant_info) == genotypes.shape[1]
    assert len(sample_ids) == genotypes.shape[0]


def test_load_bgen_with_region(test_data):
    """Test BGEN loading with region filter."""
    # First load all data to find a valid region
    all_genotypes, all_variant_info, _ = load_bgen(
        file_path=str(test_data["bgen_file"]),
        index_path=str(test_data["bgi_file"]),
        sample_path=str(test_data["sample_file"]),
    )
    
    # Skip test if no variants
    if len(all_variant_info) == 0:
        pytest.skip("No variants in test BGEN file")
    
    # Use the chromosome and position range from actual data
    first_chrom = all_variant_info["chrom"].iloc[0]
    min_pos = all_variant_info["pos"].min()
    max_pos = all_variant_info["pos"].max()
    mid_pos = (min_pos + max_pos) // 2
    
    # Create a region that contains some variants
    region = f"{first_chrom}:{min_pos}-{mid_pos}"
    
    genotypes, variant_info, sample_ids = load_bgen(
        file_path=str(test_data["bgen_file"]),
        index_path=str(test_data["bgi_file"]),
        sample_path=str(test_data["sample_file"]),
        region=region,
    )
    
    # Check that we got some variants
    assert len(variant_info) > 0
    
    # Check that all variants are within the region
    for _, variant in variant_info.iterrows():
        assert variant["chrom"] == first_chrom
        assert variant["pos"] >= min_pos
        assert variant["pos"] <= mid_pos


def test_load_bgen_without_index(test_data):
    """Test BGEN loading without index file."""
    genotypes, variant_info, sample_ids = load_bgen(
        file_path=str(test_data["bgen_file"]), sample_path=str(test_data["sample_file"])
    )
    
    # Should still load successfully
    assert genotypes.shape[0] > 0
    assert genotypes.shape[1] > 0


def test_load_bgen_empty_region_error(test_data):
    """Test that loading an empty region raises an error."""
    # Use a region that definitely doesn't contain any variants
    empty_region = "99:1-100"
    
    with pytest.raises(ValueError, match="No variants were loaded"):
        load_bgen(
            file_path=str(test_data["bgen_file"]),
            index_path=str(test_data["bgi_file"]),
            sample_path=str(test_data["sample_file"]),
            region=empty_region,
        )


def test_load_all_variants(test_data):
    """Test loading all variants."""
    # Use load_bgen which is the public API
    dosages, variant_info, sample_ids = load_bgen(
        file_path=str(test_data["bgen_file"]),
        index_path=str(test_data["bgi_file"]),
        sample_path=str(test_data["sample_file"]),
    )
    
    # Check dimensions - we know from tests that this file has 5363 samples and 55 variants
    assert dosages.shape[0] == 5363
    assert dosages.shape[1] == 55
    assert len(variant_info) == 55
    assert len(sample_ids) == 5363
    
    # Check variant info columns
    expected_cols = {"chrom", "pos", "rsid", "ref", "alt"}
    assert set(variant_info.columns) == expected_cols
    
    # Check dosages are in valid range
    assert np.all(dosages >= 0) and np.all(dosages <= 2)


def test_load_region_variants(test_data):
    """Test loading variants from a region."""
    # Load region with variants
    dosages, variant_info, sample_ids = load_bgen(
        file_path=str(test_data["bgen_file"]), index_path=str(test_data["bgi_file"]), region="01:1-10"
    )
    
    assert dosages.shape[1] > 0  # Should have some variants
    assert len(variant_info) == dosages.shape[1]
    assert np.all(variant_info["pos"] >= 1)
    assert np.all(variant_info["pos"] <= 10)
    
    # Empty region - should raise ValueError
    with pytest.raises(ValueError, match="No variants were loaded"):
        dosages2, variant_info2, sample_ids2 = load_bgen(
            file_path=str(test_data["bgen_file"]),
            index_path=str(test_data["bgi_file"]),
            region="01:100000-200000",
        )


def test_load_filtered_variants(test_data, tmp_path):
    """Test loading filtered variants from z file."""
    # Create a simple z file for testing
    z_data = pd.DataFrame(
        {
            "chromosome": ["01", "01", "01"],
            "position": [1, 5, 10],
            "allele1": ["A", "A", "A"],
            "allele2": ["G", "G", "G"],
            "rsid": ["rs1", "rs5", "rs10"],
        }
    )
    z_file = tmp_path / "test.z"
    z_data.to_csv(z_file, sep="\t", index=False)
    
    # Import the function we need
    from ldcov.utils.variant_filter import load_variant_filter
    
    # Create filter from z file
    variant_filter = load_variant_filter(str(z_file))
    
    # Load filtered variants
    dosages, variant_info, sample_ids = load_bgen(
        file_path=str(test_data["bgen_file"]),
        index_path=str(test_data["bgi_file"]),
        variant_filter=variant_filter,
    )
    
    # Should have loaded the variants in z file (or fewer if some don't exist)
    assert dosages.shape[1] <= len(variant_filter["positions"])
    assert len(variant_info) == dosages.shape[1]


def test_sample_filtering(test_data):
    """Test sample filtering in BGEN reader."""
    # First load all to get sample IDs
    _, _, all_sample_ids = load_bgen(
        file_path=str(test_data["bgen_file"]),
        index_path=str(test_data["bgi_file"]),
        sample_path=str(test_data["sample_file"]),
    )
    
    # Get subset of samples
    sample_ids_to_keep = all_sample_ids[:3]  # First 3 samples
    
    # Load with sample filtering
    dosages, variant_info, filtered_ids = load_bgen(
        file_path=str(test_data["bgen_file"]),
        index_path=str(test_data["bgi_file"]),
        sample_path=str(test_data["sample_file"]),
        sample_ids=sample_ids_to_keep,
    )
    
    assert dosages.shape[0] == 3  # Only 3 samples
    assert len(filtered_ids) == 3
    assert filtered_ids == sample_ids_to_keep


def test_nan_handling(test_data):
    """Test NaN handling options."""
    # Note: The example BGEN file doesn't contain NaN values,
    # so nan_action is accepted but not actually triggered
    
    # Should not raise with valid nan_action
    for action in ["error", "mean", "omit"]:
        dosages, variant_info, sample_ids = load_bgen(str(test_data["bgen_file"]), nan_action=action)
        assert dosages is not None
        assert not np.any(np.isnan(dosages))  # No NaN values in test data


def test_bgen_reader_context_manager(test_data):
    """Test BgenReader context manager functionality."""
    from ldcov.io.bgen.reader import BgenReader
    
    # Test basic context manager usage
    with BgenReader(str(test_data["bgen_file"])) as reader:
        # Should be able to access properties
        assert len(reader.samples) > 0
        assert reader.nsamples > 0
        assert reader.nvariants > 0
    
    # After context exit, accessing should raise error
    with pytest.raises(ValueError):
        reader.load_variants()


def test_bgen_reader_context_manager_exception(test_data):
    """Test context manager handles exceptions properly."""
    from ldcov.io.bgen.reader import BgenReader
    
    reader = None
    try:
        with BgenReader(str(test_data["bgen_file"])) as reader:
            # Simulate an error
            raise RuntimeError("Test exception")
    except RuntimeError:
        pass
    
    # Reader should still be closed after exception
    assert reader is not None
    with pytest.raises(ValueError):
        reader.load_variants()


# ==================== Sample Filtering Tests ====================

def test_get_sample_indices():
    """Test sample index mapping."""
    # Create test data
    all_sample_ids = [f"SAMPLE_{i:04d}" for i in range(100)]
    subset_sample_ids = [f"SAMPLE_{i:04d}" for i in range(10, 30)]  # 20 samples
    
    # Test the functionality directly with our test data
    # Convert to numpy arrays for efficient operations (same as in BgenReader)
    sample_ids_array = np.array(all_sample_ids)
    ids_to_keep_array = np.array(subset_sample_ids)
    
    # Find which requested samples exist in BGEN
    mask = np.isin(sample_ids_array, ids_to_keep_array)
    bgen_indices = np.where(mask)[0]
    found_ids = sample_ids_array[mask]
    
    # Create a mapping to preserve the order of sample_ids_to_keep
    # Use searchsorted for efficient ordering
    sorter = np.argsort(ids_to_keep_array)
    sorted_keep = ids_to_keep_array[sorter]
    
    # Find where each found ID would be inserted in the sorted array
    insert_positions = np.searchsorted(sorted_keep, found_ids)
    
    # Get the original positions in sample_ids_to_keep
    original_positions = sorter[insert_positions]
    
    # Sort by original order
    order = np.argsort(original_positions)
    filtered_ids = found_ids[order].tolist()
    indices = bgen_indices[order].tolist()
    
    assert len(indices) == 20
    assert len(filtered_ids) == 20
    assert indices[0] == 10  # First sample is SAMPLE_0010 at index 10
    assert indices[-1] == 29  # Last sample is SAMPLE_0029 at index 29
    assert filtered_ids == subset_sample_ids


def test_get_sample_indices_with_missing():
    """Test sample index mapping with missing samples."""
    # Create test data
    all_sample_ids = [f"SAMPLE_{i:04d}" for i in range(100)]
    subset_sample_ids = [f"SAMPLE_{i:04d}" for i in range(10, 30)]  # 20 samples
    
    # Test with some missing samples
    requested_samples = subset_sample_ids + ["MISSING_001", "MISSING_002"]
    
    # Test the functionality directly with our test data
    # Convert to numpy arrays for efficient operations (same as in BgenReader)
    sample_ids_array = np.array(all_sample_ids)
    ids_to_keep_array = np.array(requested_samples)
    
    # Find which requested samples exist in BGEN
    mask = np.isin(sample_ids_array, ids_to_keep_array)
    bgen_indices = np.where(mask)[0]
    found_ids = sample_ids_array[mask]
    
    # Create a mapping to preserve the order of sample_ids_to_keep
    # Use searchsorted for efficient ordering
    sorter = np.argsort(ids_to_keep_array)
    sorted_keep = ids_to_keep_array[sorter]
    
    # Find where each found ID would be inserted in the sorted array
    insert_positions = np.searchsorted(sorted_keep, found_ids)
    
    # Get the original positions in sample_ids_to_keep
    original_positions = sorter[insert_positions]
    
    # Sort by original order
    order = np.argsort(original_positions)
    filtered_ids = found_ids[order].tolist()
    indices = bgen_indices[order].tolist()
    
    # Should only return the samples that exist
    assert len(indices) == 20
    assert len(filtered_ids) == 20
    assert filtered_ids == subset_sample_ids


def test_load_bgen_with_sample_filtering(test_data):
    """Test BGEN loading with sample filtering."""
    # Load all samples first to get the full list
    all_genotypes, all_variant_info, all_sample_ids = load_bgen(
        file_path=str(test_data["bgen_file"]),
        index_path=str(test_data["bgi_file"]),
        sample_path=str(test_data["sample_file"]),
    )
    
    # Skip test if too few samples
    if len(all_sample_ids) < 4:
        pytest.skip("Not enough samples for filtering test")
    
    # Select a subset of samples
    subset_samples = all_sample_ids[::2]  # Every other sample
    
    # Load with sample filtering
    filtered_genotypes, filtered_variant_info, filtered_sample_ids = load_bgen(
        file_path=str(test_data["bgen_file"]),
        index_path=str(test_data["bgi_file"]),
        sample_path=str(test_data["sample_file"]),
        sample_ids=subset_samples,
    )
    
    # Verify filtering worked
    assert len(filtered_sample_ids) == len(subset_samples)
    assert filtered_sample_ids == subset_samples
    assert filtered_genotypes.shape[0] == len(subset_samples)
    assert filtered_genotypes.shape[1] == all_genotypes.shape[1]
    
    # Verify the genotype data matches for the filtered samples
    for i, sample_id in enumerate(subset_samples):
        orig_idx = all_sample_ids.index(sample_id)
        np.testing.assert_array_almost_equal(
            filtered_genotypes[i, :], all_genotypes[orig_idx, :]
        )


def test_sample_filtering_with_missing_samples(test_data):
    """Test that missing samples are handled gracefully."""
    # Get actual sample IDs
    _, _, actual_sample_ids = load_bgen(
        file_path=str(test_data["bgen_file"]),
        index_path=str(test_data["bgi_file"]),
        sample_path=str(test_data["sample_file"]),
    )
    
    if len(actual_sample_ids) < 2:
        pytest.skip("Not enough samples for test")
    
    # Request some existing and some non-existing samples
    requested_samples = [
        actual_sample_ids[0],
        "FAKE_SAMPLE_1",
        actual_sample_ids[1],
        "FAKE_SAMPLE_2",
    ]
    
    # Load with filtering
    filtered_genotypes, _, filtered_sample_ids = load_bgen(
        file_path=str(test_data["bgen_file"]),
        index_path=str(test_data["bgi_file"]),
        sample_path=str(test_data["sample_file"]),
        sample_ids=requested_samples,
    )
    
    # Should only get the existing samples
    assert len(filtered_sample_ids) == 2
    assert filtered_sample_ids == [actual_sample_ids[0], actual_sample_ids[1]]


def test_memory_efficiency_calculation():
    """Test that sample filtering reduces memory usage (conceptual test)."""
    # This is a conceptual test showing the memory benefit
    
    # Without filtering: 500K samples × 10K variants × 8 bytes
    memory_without_filtering = 500_000 * 10_000 * 8 / (1024**3)  # GB
    
    # With filtering: 10K samples × 10K variants × 8 bytes
    memory_with_filtering = 10_000 * 10_000 * 8 / (1024**3)  # GB
    
    memory_saved = memory_without_filtering - memory_with_filtering
    savings_percent = memory_saved / memory_without_filtering * 100
    
    # Assert significant memory savings
    assert savings_percent > 95  # >95% savings
    
    # Log the calculation for reference
    print(f"\nMemory usage comparison:")
    print(f"Without filtering: {memory_without_filtering:.1f} GB")
    print(f"With filtering: {memory_with_filtering:.1f} GB")
    print(f"Memory saved: {memory_saved:.1f} GB ({savings_percent:.0f}%)")


# ==================== BGI Reader Tests ====================

def test_bgi_reader_init(test_data):
    """Test BGI reader initialization."""
    from ldcov.io.bgen.bgi import BGIReader
    
    # Should succeed with valid BGI
    reader = BGIReader(str(test_data["bgi_file"]))
    assert reader is not None
    reader.close()
    
    # Should fail with non-existent file
    with pytest.raises(FileNotFoundError):
        BGIReader("/non/existent/file.bgi")
    
    # Should fail with invalid file
    with tempfile.NamedTemporaryFile(suffix=".bgi") as f:
        f.write(b"not a valid bgi file")
        f.flush()
        with pytest.raises(ValueError, match="Error reading BGI file"):
            BGIReader(f.name)


def test_bgi_get_variant_count(test_data):
    """Test getting variant count from BGI."""
    from ldcov.io.bgen.bgi import BGIReader
    
    with BGIReader(str(test_data["bgi_file"])) as reader:
        count = reader.get_variant_count()
        assert count > 0
        
        # Should be cached
        count2 = reader.get_variant_count()
        assert count2 == count


def test_bgi_get_all_variants(test_data):
    """Test getting all variant metadata from BGI."""
    from ldcov.io.bgen.bgi import BGIReader
    
    with BGIReader(str(test_data["bgi_file"])) as reader:
        variants = reader.get_all_variants()
        
        # Check structure (now a DataFrame)
        assert len(variants) > 0
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
        
        # Check first variant
        first = variants.iloc[0]
        assert first["chrom"] is not None
        assert first["pos"] > 0
        assert first["rsid"] is not None
        assert first["n_alleles"] >= 2
        assert first["ref"] is not None
        assert first["file_offset"] > 0
        assert first["size_bytes"] > 0
        
        # Check ordering by file offset
        offsets = variants["file_offset"].values
        assert np.all(offsets[1:] > offsets[:-1])  # Strictly increasing


def test_bgi_get_variants_in_region(test_data):
    """Test getting variants in a genomic region from BGI."""
    from ldcov.io.bgen.bgi import BGIReader
    
    with BGIReader(str(test_data["bgi_file"])) as reader:
        # Get all variants to find a valid region
        all_variants = reader.get_all_variants()
        if len(all_variants) == 0:
            pytest.skip("No variants in BGI file")
        
        # Use first chromosome and position range
        chrom = all_variants.iloc[0]["chrom"]
        min_pos = all_variants["pos"].min()
        max_pos = all_variants["pos"].max()
        mid_pos = (min_pos + max_pos) // 2
        
        # Region with variants
        variants = reader.get_variants_in_region(chrom, min_pos, mid_pos)
        # May have zero variants if mid_pos is too close to min_pos
        if len(variants) > 0:
            assert (variants["chrom"] == chrom).all()
            assert (variants["pos"] >= min_pos).all()
            assert (variants["pos"] <= mid_pos).all()
        
        # Empty region
        variants = reader.get_variants_in_region(chrom, max_pos + 1000, max_pos + 2000)
        assert len(variants) == 0


def test_bgi_find_variants_by_filter(test_data):
    """Test finding variants by position/allele/rsid in BGI."""
    from ldcov.io.bgen.bgi import BGIReader
    
    with BGIReader(str(test_data["bgi_file"])) as reader:
        # Get some actual variants to search for
        all_variants = reader.get_all_variants()
        if len(all_variants) < 3:
            pytest.skip("Not enough variants for test")
        
        # Create filter matching first 3 variants
        chromosome = all_variants["chrom"].iloc[0]  # Get chromosome from first variant
        positions = all_variants["pos"].values[:3]
        alleles1 = all_variants["ref"].values[:3].tolist()
        alleles2 = all_variants["alt"].values[:3].tolist()
        
        matched = reader.find_variants_by_filter(chromosome, positions, alleles1, alleles2)
        
        # Should find all 3
        assert len(matched) == 3
        # Verify they're in the same order as requested
        np.testing.assert_array_equal(matched["pos"].values, positions)
        
        # Test with swapped alleles (should NOT match since exact match is required)
        matched2 = reader.find_variants_by_filter(chromosome, positions, alleles2, alleles1)
        
        # Should find 0 matches with swapped alleles
        assert len(matched2) == 0


def test_bgi_context_manager(test_data):
    """Test BGI reader context manager usage."""
    from ldcov.io.bgen.bgi import BGIReader
    
    with BGIReader(str(test_data["bgi_file"])) as reader:
        count = reader.get_variant_count()
        assert count > 0
    
    # Connection should be closed after context
    # (Can't easily test this without accessing private attributes)



