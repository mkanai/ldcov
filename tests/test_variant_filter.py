"""
Tests for variant filtering utilities in the ldcov package.

This module tests Z-file functionality and variant filtering for:
- Basic Z-file loading
- Chromosome format preservation
- Variant filtering from Z-files
- Position ordering requirements
"""

import pytest
import pandas as pd
from pathlib import Path

from ldcov.utils.variant_filter import load_variant_filter
from ldcov.io import load_bgen


@pytest.fixture(scope="module")
def test_data():
    """Set up test data."""
    # Get example data path
    examples_dir = Path(__file__).parents[1] / "examples"
    bgen_file = examples_dir / "data" / "data.bgen"
    bgi_file = examples_dir / "data" / "data.bgen.bgi"
    
    return {
        "examples_dir": examples_dir,
        "bgen_file": bgen_file,
        "bgi_file": bgi_file
    }


def test_load_z_file_basic(tmp_path):
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
    z_file = tmp_path / "test.z"
    z_data.to_csv(z_file, sep="\t", index=False)
    
    # Load Z-file
    variant_filter = load_variant_filter(str(z_file))
    
    # Check filter structure
    assert isinstance(variant_filter, dict)
    assert len(variant_filter["rsids"]) == 2
    assert variant_filter["rsids"][0] == "rs1"
    assert variant_filter["chromosome"] == "1"


def test_z_file_original_chromosome_format(tmp_path):
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
    z_file1 = tmp_path / "test_chr1.z"
    z_data1.to_csv(z_file1, sep="\t", index=False)
    
    variant_filter1 = load_variant_filter(str(z_file1))
    assert variant_filter1["chromosome"] == "chr1"  # Keeps original format
    
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
    z_file2 = tmp_path / "test_01.z"
    z_data2.to_csv(z_file2, sep="\t", index=False)
    
    variant_filter2 = load_variant_filter(str(z_file2))
    assert variant_filter2["chromosome"] == "01"  # Keeps original format


def test_load_variant_filter(test_data, tmp_path):
    """Test loading variant filter from Z-file."""
    # Load actual variant info
    _, variant_info, _ = load_bgen(
        file_path=str(test_data["bgen_file"]),
        index_path=str(test_data["bgi_file"]),
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
    z_file = tmp_path / "test_filter.z"
    z_data.to_csv(z_file, sep="\t", index=False)
    
    # Load filter in one step
    variant_filter = load_variant_filter(str(z_file))
    
    # Check that filter is a dictionary with expected keys
    assert isinstance(variant_filter, dict)
    assert "chromosome" in variant_filter
    assert "positions" in variant_filter
    assert "rsids" in variant_filter
    
    # Should have same number of positions as input
    assert len(variant_filter["positions"]) == len(z_data)


def test_z_file_ordering(tmp_path):
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
    z_file = tmp_path / "test_order.z"
    z_data.to_csv(z_file, sep="\t", index=False)
    
    # Should raise error for unsorted positions
    with pytest.raises(ValueError, match="not sorted"):
        load_variant_filter(str(z_file))