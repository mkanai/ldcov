"""
BGEN file reader for efficient reading of BGEN genotype files.

This module provides functions for reading BGEN files.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict, Any, Union
import logging

logger = logging.getLogger(__name__)

try:
    from bgen import BgenReader

    BGEN_AVAILABLE = True
except ImportError:
    logger.warning("bgen module not available; install with 'pip install bgen'")
    BGEN_AVAILABLE = False


def _extract_variant_info(variant, idx: int) -> Dict[str, Any]:
    """
    Extract variant information from a BGEN variant object.

    Parameters:
    -----------
    variant : bgen variant object
        Variant object from the BGEN library
    idx : int
        Index of the variant

    Returns:
    --------
    dict
        Variant information dictionary
    """
    alleles = variant.alleles
    return {
        "chrom": variant.chrom,
        "pos": variant.pos,
        "id": variant.rsid,
        "ref": alleles[0],
        "alt": alleles[1] if len(alleles) > 1 else "",
        "rsid": variant.rsid,
        "idx": idx,
    }


def _compute_dosage_from_variant(variant) -> np.ndarray:
    """
    Compute dosage from a BGEN variant object.

    Parameters:
    -----------
    variant : bgen variant object
        Variant object from the BGEN library

    Returns:
    --------
    np.ndarray
        Dosage array for all samples

    Raises:
    -------
    Exception
        If dosage cannot be computed
    """
    try:
        return variant.alt_dosage
    except Exception:
        # Fall back to computing dosage from probabilities
        probs = variant.probabilities

        # Calculate dosages: 0*P(AA) + 1*P(AB) + 2*P(BB)
        if probs.shape[1] == 3:  # diploid, biallelic
            return probs[:, 1] + 2 * probs[:, 2]
        elif probs.shape[1] == 2:  # haploid, biallelic
            return probs[:, 1]
        else:
            raise ValueError(f"Unsupported probabilities shape: {probs.shape}")


def _normalize_chromosome(chrom: str) -> str:
    """
    Normalize chromosome name for BGEN queries.

    Parameters:
    -----------
    chrom : str
        Chromosome name (with or without 'chr' prefix)

    Returns:
    --------
    str
        Normalized chromosome name (without 'chr' prefix)
    """
    return chrom[3:] if chrom.startswith("chr") else chrom


class BgenFileReader:
    """
    Reader for BGEN format files with support for indexed access.
    """

    def __init__(
        self, file_path: str, index_path: Optional[str] = None, sample_path: Optional[str] = None
    ):
        """
        Initialize BGEN reader.

        Parameters:
        -----------
        file_path : str
            Path to BGEN file
        index_path : str, optional
            Path to BGI index file (will use file_path + '.bgi' if not provided)
        sample_path : str, optional
            Path to sample file
        """
        self.file_path = file_path
        self.index_path = index_path
        self.sample_path = sample_path

        # Check if bgen module is available
        if not BGEN_AVAILABLE:
            raise ImportError("bgen module not available. Install with 'pip install bgen'")

        # Open BGEN file and get metadata
        self._open_bgen()

    def _open_bgen(self):
        """Open BGEN file and initialize metadata."""
        logger.info(f"Opening BGEN file: {self.file_path}")
        try:
            # BgenReader will automatically use the .bgi index file if it exists
            self.bgen_file = BgenReader(
                self.file_path, sample_path=self.sample_path if self.sample_path else ""
            )

            # Get metadata
            self.sample_ids = self.bgen_file.samples
            self.n_samples = len(self.sample_ids)

            logger.info(f"Opened BGEN file with {self.n_samples} samples")

        except Exception as e:
            logger.error(f"Error opening BGEN file: {e}")
            raise

    def close(self):
        """Close BGEN file."""
        if hasattr(self, "bgen_file"):
            self.bgen_file.close()

    def get_variants_in_region(self, chrom: str, start_pos: int, end_pos: int) -> pd.DataFrame:
        """
        Get variant information for a genomic region.

        Parameters:
        -----------
        chrom : str
            Chromosome
        start_pos : int
            Start position
        end_pos : int
            End position

        Returns:
        --------
        pandas.DataFrame
            Variant information
        """
        try:
            search_chrom = _normalize_chromosome(chrom)
            variants_in_region = self.bgen_file.fetch(search_chrom, start_pos, end_pos)

            variant_info = [
                _extract_variant_info(variant, i) for i, variant in enumerate(variants_in_region)
            ]

            if not variant_info:
                logger.warning(f"No variants found in region {chrom}:{start_pos}-{end_pos}")
                return pd.DataFrame()

            return pd.DataFrame(variant_info)

        except Exception as e:
            logger.warning(f"Error querying {chrom}:{start_pos}-{end_pos}: {e}")
            return pd.DataFrame()

    def load_all_variants_and_dosages(
        self, dtype: np.dtype = np.float64
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Load all variants and their dosages in a single efficient pass.

        Parameters:
        -----------
        dtype : numpy.dtype, optional
            Data type for the dosage array (default: np.float64)

        Returns:
        --------
        tuple
            (dosages, variant_info)
        """
        logger.info("Loading all variants and dosages in one pass")

        variant_info_list = []
        dosages_list = []

        for i, variant in enumerate(self.bgen_file):
            # Extract variant info
            variant_info_list.append(_extract_variant_info(variant, i))

            # Compute dosage
            try:
                dosage = _compute_dosage_from_variant(variant)
                dosages_list.append(dosage)
            except Exception as e:
                logger.warning(f"Error computing dosage for variant {i} ({variant.rsid}): {e}")
                dosages_list.append(np.full(self.n_samples, np.nan, dtype=dtype))

        if not variant_info_list:
            return np.zeros((self.n_samples, 0), dtype=dtype), pd.DataFrame()

        # Convert to arrays
        dosages = np.column_stack(dosages_list).astype(dtype)
        variant_info = pd.DataFrame(variant_info_list)

        return dosages, variant_info

    def load_region_variants_and_dosages(
        self, chrom: str, start_pos: int, end_pos: int, dtype: np.dtype = np.float64
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Load variants and dosages for a specific genomic region.

        Parameters:
        -----------
        chrom : str
            Chromosome
        start_pos : int
            Start position
        end_pos : int
            End position
        dtype : numpy.dtype, optional
            Data type for the dosage array (default: np.float64)

        Returns:
        --------
        tuple
            (dosages, variant_info)
        """
        try:
            search_chrom = _normalize_chromosome(chrom)
            variants_in_region = list(self.bgen_file.fetch(search_chrom, start_pos, end_pos))

            if not variants_in_region:
                logger.warning(f"No variants found in region {chrom}:{start_pos}-{end_pos}")
                return np.zeros((self.n_samples, 0), dtype=dtype), pd.DataFrame()

            variant_info_list = []
            dosages_list = []

            for i, variant in enumerate(variants_in_region):
                # Extract variant info
                variant_info_list.append(_extract_variant_info(variant, i))

                # Compute dosage
                try:
                    dosage = _compute_dosage_from_variant(variant)
                    dosages_list.append(dosage)
                except Exception as e:
                    logger.warning(f"Error computing dosage for variant {i} ({variant.rsid}): {e}")
                    dosages_list.append(np.full(self.n_samples, np.nan, dtype=dtype))

            # Convert to arrays
            dosages = np.column_stack(dosages_list).astype(dtype)
            variant_info = pd.DataFrame(variant_info_list)

            return dosages, variant_info

        except Exception as e:
            logger.error(f"Error loading region {chrom}:{start_pos}-{end_pos}: {e}")
            return np.zeros((self.n_samples, 0), dtype=dtype), pd.DataFrame()

    def __del__(self):
        """Clean up when object is deleted."""
        self.close()


def load_bgen(
    file_path: str,
    index_path: Optional[str] = None,
    sample_path: Optional[str] = None,
    region: Optional[str] = None,
    variant_filter: Optional[Dict[str, Any]] = None,
    dtype: np.dtype = np.float64,
) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    """
    Load genotype data from BGEN file.

    Parameters:
    -----------
    file_path : str
        Path to BGEN file
    index_path : str, optional
        Path to BGI index file
    sample_path : str, optional
        Path to sample file
    region : str, optional
        Genomic region in format "chr:start-end"
    variant_filter : dict, optional
        Variant filter from .z file (from create_variant_filter_from_z)
    dtype : numpy.dtype, optional
        Data type for the dosage array (default: np.float64)

    Returns:
    --------
    tuple
        (genotypes, variant_info, sample_ids)
        Note: genotypes are returned as floating point values of the specified dtype
        If variant_filter is provided, variants are ordered according to the .z file order
    """
    # Open BGEN file
    bgen_reader = BgenFileReader(
        file_path=file_path, index_path=index_path, sample_path=sample_path
    )

    try:
        if variant_filter is not None:
            # Load variants specified in .z file
            from ..utils.variant_filter import validate_variants_match_z_file

            # First load all variants to find matches with .z file
            logger.info("Loading all variants to match with .z file filter")
            dosages, variant_info = bgen_reader.load_all_variants_and_dosages(dtype)

            # Find matching variants and get mapping indices
            bgen_indices, z_indices = validate_variants_match_z_file(variant_info, variant_filter)

            # Subset and reorder according to .z file
            filtered_dosages = dosages[:, bgen_indices]
            filtered_variant_info = variant_info.iloc[bgen_indices].copy()

            # Reorder to match .z file order
            z_order = np.argsort(z_indices)  # Get indices that would sort z_indices
            final_dosages = filtered_dosages[:, z_order]
            final_variant_info = filtered_variant_info.iloc[z_order].reset_index(drop=True)

            logger.info(f"Filtered to {final_dosages.shape[1]} variants matching .z file order")
            return final_dosages, final_variant_info, bgen_reader.sample_ids

        elif region is not None:
            # Parse region string and load specific region
            from ..utils.region_utils import parse_region

            chrom, pos_range = parse_region(region)
            start_pos, end_pos = pos_range
            logger.info(f"Loading region: {chrom}:{start_pos}-{end_pos}")

            dosages, variant_info = bgen_reader.load_region_variants_and_dosages(
                chrom, start_pos, end_pos, dtype
            )
        else:
            # Load all variants and dosages in one efficient pass
            dosages, variant_info = bgen_reader.load_all_variants_and_dosages(dtype)

        # Check if we loaded any variants
        if dosages.size == 0 or dosages.shape[1] == 0:
            raise ValueError(
                "No variants were loaded from the BGEN file. "
                "This may be due to: "
                "1) An empty genomic region, "
                "2) No variants passing the filter criteria, "
                "3) Issues with the BGEN file format"
            )

        return dosages, variant_info, bgen_reader.sample_ids

    except Exception as e:
        logger.error(f"Error loading BGEN file: {e}")
        raise
    finally:
        bgen_reader.close()
