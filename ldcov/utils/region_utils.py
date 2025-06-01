"""
Utility functions for working with genomic regions.
"""

from typing import Tuple


def parse_region(region_str: str) -> Tuple[str, Tuple[int, int]]:
    """
    Parse a genomic region string.

    Parameters:
    -----------
    region_str : str
        Genomic region in format "chr:start-end"

    Returns:
    --------
    tuple
        (chrom, (start_pos, end_pos))

    Raises:
    -------
    ValueError
        If the region string is not in the expected format
    """
    if region_str is None:
        raise ValueError("Region string cannot be None")

    try:
        chrom, pos_range = region_str.split(":")
        start_pos, end_pos = map(int, pos_range.split("-"))
        return chrom, (start_pos, end_pos)
    except ValueError:
        raise ValueError(f"Invalid region format: {region_str}. Expected format: 'chr:start-end'")
