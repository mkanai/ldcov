"""
BGEN file reader for efficient reading of BGEN genotype files.

This module provides functions for reading BGEN files with mandatory BGI index support.
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict, Any, Union
import logging
from tqdm import tqdm

from .bgi_reader import BGIReader

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


def _impute_nan_with_mean(
    dosages: np.ndarray, variant_info: pd.DataFrame
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Impute NaN values with variant-wise mean.

    Parameters:
    -----------
    dosages : np.ndarray
        Genotype dosage matrix (samples x variants)
    variant_info : pd.DataFrame
        Variant information

    Returns:
    --------
    tuple
        (imputed_dosages, variant_info)
    """
    # Count NaN values for logging
    nan_mask = np.isnan(dosages)
    n_nan_total = np.sum(nan_mask)
    variants_with_nan = np.any(nan_mask, axis=0)
    n_variants_with_nan = np.sum(variants_with_nan)

    logger.warning(
        f"Found {n_nan_total} NaN values across {n_variants_with_nan} variants. "
        f"Imputing with variant-wise mean."
    )

    # Create a copy to avoid modifying original
    imputed_dosages = dosages.copy()

    # Impute variant by variant
    for j in range(dosages.shape[1]):
        if np.any(nan_mask[:, j]):
            # Calculate mean excluding NaN values
            variant_mean = np.nanmean(dosages[:, j])

            # If all values are NaN for this variant, use 0
            if np.isnan(variant_mean):
                logger.warning(
                    f"Variant {variant_info.iloc[j]['id']} at position "
                    f"{variant_info.iloc[j]['pos']} has all NaN values. Imputing with 0."
                )
                variant_mean = 0.0

            # Impute NaN values with mean
            imputed_dosages[nan_mask[:, j], j] = variant_mean

    return imputed_dosages, variant_info


def _omit_nan_samples(
    dosages: np.ndarray, variant_info: pd.DataFrame, sample_ids: List[str]
) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    """
    Omit samples with any NaN values.

    Parameters:
    -----------
    dosages : np.ndarray
        Genotype dosage matrix (samples x variants)
    variant_info : pd.DataFrame
        Variant information
    sample_ids : list of str
        Sample IDs

    Returns:
    --------
    tuple
        (filtered_dosages, variant_info, filtered_sample_ids)
    """
    # Find samples with any NaN values
    nan_mask = np.isnan(dosages)
    samples_with_nan = np.any(nan_mask, axis=1)
    n_samples_with_nan = np.sum(samples_with_nan)

    if n_samples_with_nan == 0:
        return dosages, variant_info, sample_ids

    # Log warning about samples being removed
    logger.warning(
        f"Removing {n_samples_with_nan} samples with NaN values out of {len(sample_ids)} total samples."
    )

    # Show first few sample IDs being removed
    removed_sample_ids = [sample_ids[i] for i in np.where(samples_with_nan)[0][:5]]
    if n_samples_with_nan <= 5:
        logger.warning(f"Removed samples: {', '.join(removed_sample_ids)}")
    else:
        logger.warning(
            f"First 5 removed samples: {', '.join(removed_sample_ids)} "
            f"(and {n_samples_with_nan - 5} more)"
        )

    # Keep only samples without NaN
    keep_mask = ~samples_with_nan
    filtered_dosages = dosages[keep_mask, :]
    filtered_sample_ids = [sid for i, sid in enumerate(sample_ids) if keep_mask[i]]

    # Also check if any variants now have all missing values
    all_nan_variants = np.all(np.isnan(filtered_dosages), axis=0)
    if np.any(all_nan_variants):
        n_all_nan = np.sum(all_nan_variants)
        logger.warning(
            f"After removing samples, {n_all_nan} variants have no valid data. "
            f"Consider using 'mean' imputation instead."
        )

    return filtered_dosages, variant_info, filtered_sample_ids


def _report_nan_error(
    dosages: np.ndarray, variant_info: pd.DataFrame, sample_ids: List[str]
) -> None:
    """
    Report detailed error message for NaN values in genotype matrix.

    Parameters:
    -----------
    dosages : np.ndarray
        Genotype dosage matrix
    variant_info : pd.DataFrame
        Variant information
    sample_ids : list of str
        Sample IDs

    Raises:
    -------
    ValueError
        Always raises with detailed NaN information
    """
    nan_mask = np.isnan(dosages)

    # Count samples and variants with NaN
    samples_with_nan = np.any(nan_mask, axis=1)
    variants_with_nan = np.any(nan_mask, axis=0)
    n_samples_with_nan = np.sum(samples_with_nan)
    n_variants_with_nan = np.sum(variants_with_nan)

    # Find first 5 sample/variant pairs with NaN
    nan_locations = np.argwhere(nan_mask)[:5]

    # Build detailed error message
    error_msg = (
        f"Genotype matrix contains NaN values:\n"
        f"  - {n_samples_with_nan} out of {dosages.shape[0]} samples have NaN values\n"
        f"  - {n_variants_with_nan} out of {dosages.shape[1]} variants have NaN values\n"
    )

    if len(nan_locations) > 0:
        error_msg += "\nFirst (up to 5) sample/variant pairs with NaN:\n"
        for i, (sample_idx, variant_idx) in enumerate(nan_locations):
            sample_id = sample_ids[sample_idx]
            variant_id = (
                variant_info.iloc[variant_idx]["id"]
                if "id" in variant_info
                else f"variant_{variant_idx}"
            )
            variant_pos = (
                variant_info.iloc[variant_idx]["pos"] if "pos" in variant_info else "unknown"
            )
            error_msg += f"  {i+1}. Sample '{sample_id}' (index {sample_idx}), Variant '{variant_id}' at position {variant_pos} (index {variant_idx})\n"

    error_msg += "\nThis may indicate issues with the input BGEN file or variant filtering."

    raise ValueError(error_msg)


class BgenFileReader:
    """
    Reader for BGEN format files with mandatory BGI index support.
    """

    def __init__(
        self,
        file_path: str,
        index_path: Optional[str] = None,
        sample_path: Optional[str] = None,
        show_progress: bool = True,
    ):
        """
        Initialize BGEN reader.

        Parameters:
        -----------
        file_path : str
            Path to BGEN file
        index_path : str, optional
            Path to BGI index file. If None, will look for file_path + '.bgi'
        sample_path : str, optional
            Path to sample file
        show_progress : bool, optional
            Whether to show progress bars (default: True)
            
        Raises:
        -------
        FileNotFoundError
            If BGEN file or required BGI index is not found
        ImportError
            If bgen module is not available
        """
        self.file_path = file_path
        self.sample_path = sample_path
        self.show_progress = show_progress

        # Check if bgen module is available
        if not BGEN_AVAILABLE:
            raise ImportError("bgen module not available. Install with 'pip install bgen'")
            
        # Check BGEN file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"BGEN file not found: {file_path}")
        
        # Determine BGI path
        if index_path is not None:
            self.bgi_path = index_path
        else:
            self.bgi_path = file_path + '.bgi'
            
        # BGI is mandatory
        if not os.path.exists(self.bgi_path):
            raise FileNotFoundError(
                f"BGI index required but not found: {self.bgi_path}\n"
                f"Please create index using: bgenix -g {file_path}"
            )
        
        # Open BGI for metadata access
        self.bgi = BGIReader(self.bgi_path)
        
        # Open BGEN file and get metadata
        self._open_bgen()

    def _open_bgen(self):
        """Open BGEN file and initialize metadata."""
        logger.info(f"Opening BGEN file: {self.file_path}")
        try:
            # BgenReader will use the BGI index we verified exists
            self.bgen_file = BgenReader(
                self.file_path, sample_path=self.sample_path if self.sample_path else ""
            )

            # Get metadata
            self.sample_ids = self.bgen_file.samples
            self.n_samples = len(self.sample_ids)
            
            # Get variant count from BGI
            self.n_variants = self.bgi.get_variant_count()

            logger.info(f"Opened BGEN file with {self.n_samples} samples and {self.n_variants} variants")

        except Exception as e:
            logger.error(f"Error opening BGEN file: {e}")
            raise

    def close(self):
        """Close BGEN and BGI files."""
        if hasattr(self, "bgen_file"):
            self.bgen_file.close()
        if hasattr(self, "bgi"):
            self.bgi.close()

    def _load_variants(
        self,
        variant_metadata: np.ndarray,
        sample_indices: Optional[List[int]] = None,
        dtype: np.dtype = np.float64,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Load variant dosages using metadata from BGI.
        
        This is the unified loading method that handles all scenarios.
        
        Parameters:
        -----------
        variant_metadata : np.ndarray
            Structured array from BGI with variant metadata
        sample_indices : List[int], optional
            Sample indices to keep
        dtype : np.dtype
            Data type for dosages
            
        Returns:
        --------
        Tuple[np.ndarray, pd.DataFrame]
            (dosages, variant_info)
        """
        n_variants = len(variant_metadata)
        if n_variants == 0:
            n_samples_out = len(sample_indices) if sample_indices else self.n_samples
            return np.empty((n_samples_out, 0), dtype=dtype), pd.DataFrame()
        
        # Determine output dimensions
        n_samples_out = len(sample_indices) if sample_indices else self.n_samples
        
        # Pre-allocate dosage array
        dosages = np.empty((n_samples_out, n_variants), dtype=dtype)
        
        # For efficient loading, we'll use the bgen library's index-based access
        # First, create a mapping of positions to our output columns
        position_to_col = {}
        chrom_pos_to_col = {}
        for i, meta in enumerate(variant_metadata):
            chrom_pos_to_col[(meta['chrom'], meta['pos'])] = i
            position_to_col[meta['pos']] = i
        
        # Load variants
        loaded = 0
        
        if self.show_progress:
            # For specific regions/filters, use fetch if possible
            if n_variants < self.n_variants / 2:  # Heuristic: use fetch for small subsets
                # Get unique chromosome from metadata
                chroms = np.unique(variant_metadata['chrom'])
                
                with tqdm(total=n_variants, desc="Loading variants", unit="variants") as pbar:
                    for chrom in chroms:
                        chrom_mask = variant_metadata['chrom'] == chrom
                        if not np.any(chrom_mask):
                            continue
                            
                        chrom_variants = variant_metadata[chrom_mask]
                        min_pos = np.min(chrom_variants['pos'])
                        max_pos = np.max(chrom_variants['pos'])
                        
                        # Fetch variants in this region
                        for variant in self.bgen_file.fetch(chrom, min_pos, max_pos):
                            key = (variant.chrom, variant.pos)
                            if key in chrom_pos_to_col:
                                col_idx = chrom_pos_to_col[key]
                                self._process_variant_dosage(variant, col_idx, dosages, col_idx, sample_indices)
                                loaded += 1
                                pbar.update(1)
                                
                                if loaded >= n_variants:
                                    break
                        
                        if loaded >= n_variants:
                            break
            else:
                # For all variants or large subsets, iterate through all
                with tqdm(total=n_variants, desc="Loading variants", unit="variants") as pbar:
                    for variant in self.bgen_file:
                        key = (variant.chrom, variant.pos)
                        if key in chrom_pos_to_col:
                            col_idx = chrom_pos_to_col[key]
                            self._process_variant_dosage(variant, col_idx, dosages, col_idx, sample_indices)
                            loaded += 1
                            pbar.update(1)
                            
                            if loaded >= n_variants:
                                break
        else:
            # Same logic without progress bar
            if n_variants < self.n_variants / 2:
                chroms = np.unique(variant_metadata['chrom'])
                for chrom in chroms:
                    chrom_mask = variant_metadata['chrom'] == chrom
                    if not np.any(chrom_mask):
                        continue
                        
                    chrom_variants = variant_metadata[chrom_mask]
                    min_pos = np.min(chrom_variants['pos'])
                    max_pos = np.max(chrom_variants['pos'])
                    
                    for variant in self.bgen_file.fetch(chrom, min_pos, max_pos):
                        key = (variant.chrom, variant.pos)
                        if key in chrom_pos_to_col:
                            col_idx = chrom_pos_to_col[key]
                            self._process_variant_dosage(variant, col_idx, dosages, col_idx, sample_indices)
                            loaded += 1
                            
                            if loaded >= n_variants:
                                break
                    
                    if loaded >= n_variants:
                        break
            else:
                for variant in self.bgen_file:
                    key = (variant.chrom, variant.pos)
                    if key in chrom_pos_to_col:
                        col_idx = chrom_pos_to_col[key]
                        self._process_variant_dosage(variant, col_idx, dosages, col_idx, sample_indices)
                        loaded += 1
                        
                        if loaded >= n_variants:
                            break
        
        # Convert metadata to DataFrame
        variant_info = pd.DataFrame({
            'chrom': variant_metadata['chrom'],
            'pos': variant_metadata['pos'],
            'id': variant_metadata['rsid'],
            'rsid': variant_metadata['rsid'],
            'ref': variant_metadata['ref'],
            'alt': variant_metadata['alt'],
            'idx': np.arange(n_variants)
        })
        
        return dosages, variant_info

    def get_sample_indices(self, sample_ids_to_keep: List[str]) -> Tuple[List[int], List[str]]:
        """
        Map sample IDs to their indices in the BGEN file.

        Parameters:
        -----------
        sample_ids_to_keep : list of str
            Sample IDs to keep

        Returns:
        --------
        tuple
            (indices, filtered_sample_ids) where indices are positions in BGEN file
            and filtered_sample_ids are the IDs that were found
        """
        # Convert to numpy arrays for efficient operations
        sample_ids_array = np.array(self.sample_ids)
        ids_to_keep_array = np.array(sample_ids_to_keep)

        # Find matches using numpy
        # Create a boolean mask for samples to keep
        mask = np.isin(sample_ids_array, ids_to_keep_array)
        indices = np.where(mask)[0].tolist()

        # Get the filtered IDs in the order they appear in BGEN
        filtered_ids = sample_ids_array[mask].tolist()

        # To preserve the order of sample_ids_to_keep, we need to reorder
        # Create a mapping from ID to its position in sample_ids_to_keep
        order_map = {sid: i for i, sid in enumerate(sample_ids_to_keep)}

        # Sort filtered_ids and indices by their order in sample_ids_to_keep
        if filtered_ids:
            sorted_pairs = sorted(
                zip(filtered_ids, indices), key=lambda x: order_map.get(x[0], float("inf"))
            )
            filtered_ids, indices = zip(*sorted_pairs)
            filtered_ids = list(filtered_ids)
            indices = list(indices)

        if len(indices) < len(sample_ids_to_keep):
            missing = set(sample_ids_to_keep) - set(filtered_ids)
            logger.warning(
                f"Found {len(indices)} out of {len(sample_ids_to_keep)} requested samples. "
                f"Missing {len(missing)} samples."
            )

        return indices, filtered_ids


    def _process_variant_dosage(
        self,
        variant,
        idx: int,
        dosages: np.ndarray,
        col_idx: int,
        sample_indices: Optional[List[int]] = None,
    ) -> None:
        """
        Process a single variant's dosage and store it in the dosages array.

        Parameters:
        -----------
        variant : bgen variant object
            Variant to process
        idx : int
            Original variant index
        dosages : np.ndarray
            Output array to store dosages
        col_idx : int
            Column index in output array
        sample_indices : list of int, optional
            Sample indices to filter
        """
        try:
            dosage = _compute_dosage_from_variant(variant)

            # Filter samples and assign to column
            if sample_indices is not None:
                dosages[:, col_idx] = dosage[sample_indices]
            else:
                dosages[:, col_idx] = dosage
        except Exception as e:
            logger.warning(f"Error computing dosage for variant at column {col_idx}: {e}")
            dosages[:, col_idx] = np.nan



    def load_all_variants(
        self,
        sample_indices: Optional[List[int]] = None,
        dtype: np.dtype = np.float64,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Load all variants using BGI metadata.
        
        Parameters:
        -----------
        sample_indices : List[int], optional
            Sample indices to keep
        dtype : np.dtype
            Data type for dosages
            
        Returns:
        --------
        Tuple[np.ndarray, pd.DataFrame]
            (dosages, variant_info)
        """
        logger.info("Loading all variants from BGEN file")
        variant_metadata = self.bgi.get_all_variants()
        return self._load_variants(variant_metadata, sample_indices, dtype)
    
    def load_region_variants(
        self,
        chrom: str,
        start_pos: int, 
        end_pos: int,
        sample_indices: Optional[List[int]] = None,
        dtype: np.dtype = np.float64,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Load variants in a genomic region using BGI.
        
        Parameters:
        -----------
        chrom : str
            Chromosome
        start_pos : int
            Start position (inclusive)
        end_pos : int
            End position (inclusive)
        sample_indices : List[int], optional
            Sample indices to keep
        dtype : np.dtype
            Data type for dosages
            
        Returns:
        --------
        Tuple[np.ndarray, pd.DataFrame]
            (dosages, variant_info)
        """
        search_chrom = _normalize_chromosome(chrom)
        logger.info(f"Loading variants from region {search_chrom}:{start_pos}-{end_pos}")
        
        variant_metadata = self.bgi.get_variants_in_region(search_chrom, start_pos, end_pos)
        if len(variant_metadata) == 0:
            logger.warning(f"No variants found in region {chrom}:{start_pos}-{end_pos}")
        
        return self._load_variants(variant_metadata, sample_indices, dtype)
    
    def load_filtered_variants(
        self,
        variant_filter: Dict[str, Any],
        sample_indices: Optional[List[int]] = None,
        dtype: np.dtype = np.float64,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Load variants matching a filter using BGI.
        
        Parameters:
        -----------
        variant_filter : Dict[str, Any]
            Variant filter from .z file with 'positions', 'allele1', 'allele2', 'rsids'
        sample_indices : List[int], optional
            Sample indices to keep
        dtype : np.dtype
            Data type for dosages
            
        Returns:
        --------
        Tuple[np.ndarray, pd.DataFrame]
            (dosages, variant_info)
        """
        logger.info("Loading filtered variants matching .z file")
        
        # Convert positions to numpy array if not already
        positions = np.array(variant_filter['positions'], dtype=np.int32)
        
        # Find matching variants
        matched_variants, matched_indices = self.bgi.find_variants_by_filter(
            positions,
            variant_filter['allele1'],
            variant_filter['allele2'],
            variant_filter['rsids']
        )
        
        if len(matched_variants) == 0:
            raise ValueError("No variants from filter found in BGEN file")
        
        logger.info(
            f"Found {len(matched_variants)} out of {len(positions)} variants from filter"
        )
        
        return self._load_variants(matched_variants, sample_indices, dtype)

    def __del__(self):
        """Clean up when object is deleted."""
        self.close()


def load_bgen(
    file_path: str,
    index_path: Optional[str] = None,
    sample_path: Optional[str] = None,
    region: Optional[str] = None,
    variant_filter: Optional[Dict[str, Any]] = None,
    sample_ids: Optional[List[str]] = None,
    dtype: np.dtype = np.float64,
    show_progress: bool = True,
    nan_action: str = "error",
) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    """
    Load genotype data from BGEN file.

    Parameters:
    -----------
    file_path : str
        Path to BGEN file
    index_path : str, optional
        Path to BGI index file. If None, will look for file_path + '.bgi'
    sample_path : str, optional
        Path to sample file
    region : str, optional
        Genomic region in format "chr:start-end"
    variant_filter : dict, optional
        Variant filter from .z file (from create_variant_filter_from_z)
    sample_ids : list of str, optional
        Sample IDs to keep. If None, all samples are loaded.
    dtype : numpy.dtype, optional
        Data type for the dosage array (default: np.float64)
    show_progress : bool, optional
        Whether to show progress bars during loading (default: True)
    nan_action : str, optional
        Action for handling NaN values: 'error' (default), 'mean', or 'omit'

    Returns:
    --------
    tuple
        (genotypes, variant_info, sample_ids)
        Note: genotypes are returned as floating point values of the specified dtype
        If variant_filter is provided, variants are ordered according to the .z file order
        If sample_ids is provided, only those samples are returned
    """
    # Open BGEN file (BGI index is required)
    bgen_reader = BgenFileReader(
        file_path=file_path,
        index_path=index_path,
        sample_path=sample_path,
        show_progress=show_progress,
    )

    try:
        # Process sample filtering if requested
        sample_indices = None
        filtered_sample_ids = bgen_reader.sample_ids

        if sample_ids is not None:
            logger.info(f"Filtering BGEN to {len(sample_ids)} requested samples")
            sample_indices, filtered_sample_ids = bgen_reader.get_sample_indices(sample_ids)

            if not sample_indices:
                raise ValueError(
                    "No requested samples found in BGEN file. "
                    "Please check that sample IDs match between files."
                )

            # Only log if there's a difference between requested and found
            if len(sample_indices) < len(sample_ids):
                logger.info(f"Found {len(sample_indices)} of {len(sample_ids)} requested samples")

        if variant_filter is not None:
            # Load filtered variants
            dosages, variant_info = bgen_reader.load_filtered_variants(
                variant_filter, sample_indices, dtype
            )

        elif region is not None:
            # Parse region string
            from ..utils.region_utils import parse_region

            chrom, pos_range = parse_region(region)
            start_pos, end_pos = pos_range

            # Load region variants
            dosages, variant_info = bgen_reader.load_region_variants(
                chrom, start_pos, end_pos, sample_indices, dtype
            )
        else:
            # Load all variants
            dosages, variant_info = bgen_reader.load_all_variants(
                sample_indices, dtype
            )

        # Check if we loaded any variants
        if dosages.size == 0 or dosages.shape[1] == 0:
            raise ValueError(
                "No variants were loaded from the BGEN file. "
                "This may be due to: "
                "1) An empty genomic region, "
                "2) No variants passing the filter criteria, "
                "3) Issues with the BGEN file format"
            )

        # Validate genotypes
        assert np.issubdtype(dosages.dtype, np.floating), "Genotypes must be floating point"

        # Handle NaN values based on nan_action
        if np.any(np.isnan(dosages)):
            if nan_action == "error":
                _report_nan_error(dosages, variant_info, filtered_sample_ids)
            elif nan_action == "mean":
                dosages, variant_info = _impute_nan_with_mean(dosages, variant_info)
            elif nan_action == "omit":
                dosages, variant_info, filtered_sample_ids = _omit_nan_samples(
                    dosages, variant_info, filtered_sample_ids
                )
            else:
                raise ValueError(f"Unknown nan_action: {nan_action}")

        return dosages, variant_info, filtered_sample_ids

    except Exception as e:
        logger.error(f"Error loading BGEN file: {e}")
        raise
    finally:
        bgen_reader.close()
