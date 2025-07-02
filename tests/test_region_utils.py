"""
Tests for region parsing utilities in the ldcov package.

This module tests the region_utils.parse_region function for:
- Different chromosome formats
- Error handling for invalid formats
- Edge cases
"""

import pytest

from ldcov.utils.region_utils import parse_region


@pytest.mark.parametrize("region_str,expected_chrom,expected_start,expected_end", [
    ("1:1000000-2000000", "1", 1000000, 2000000),
    ("chr1:1000000-2000000", "chr1", 1000000, 2000000),
    ("01:1000000-2000000", "01", 1000000, 2000000),
    ("X:1000000-2000000", "X", 1000000, 2000000),
    ("Y:500000-1000000", "Y", 500000, 1000000),
])
def test_parse_region_formats(region_str, expected_chrom, expected_start, expected_end):
    """Test region parsing with different chromosome formats."""
    chrom, (start, end) = parse_region(region_str)
    assert chrom == expected_chrom
    assert start == expected_start
    assert end == expected_end


@pytest.mark.parametrize("invalid_region", [
    "1-1000000-2000000",  # Missing colon
    "1:10000002000000",   # Missing hyphen
    "1:abc-def",          # Invalid positions
])
def test_parse_region_invalid_format(invalid_region):
    """Test error handling for invalid region formats."""
    with pytest.raises(ValueError):
        parse_region(invalid_region)


def test_parse_region_edge_cases():
    """Test edge cases in region parsing."""
    # Single position (start = end)
    chrom, (start, end) = parse_region("1:1000000-1000000")
    assert start == end
    
    # Large positions
    chrom, (start, end) = parse_region("1:200000000-300000000")
    assert start == 200000000
    assert end == 300000000