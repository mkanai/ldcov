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

    def _determine_n_samples_out(
        self, sample_indices: Optional[List[int]], log: bool = True
    ) -> int:
        """Determine output number of samples based on filtering."""
        if sample_indices is not None:
            n_samples_out = len(sample_indices)
            if log:
                logger.info(
                    f"Loading variants with {n_samples_out} samples (filtered from {self.n_samples})"
                )
        else:
            n_samples_out = self.n_samples
            if log:
                logger.info(f"Loading variants with {self.n_samples} samples")
        return n_samples_out

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
            logger.warning(f"Error computing dosage for variant {idx} ({variant.rsid}): {e}")
            dosages[:, col_idx] = np.nan

    def _empty_result(
        self, n_samples_out: int, dtype: np.dtype
    ) -> Tuple[np.ndarray, pd.DataFrame, int]:
        """Return empty result arrays."""
        return np.zeros((n_samples_out, 0), dtype=dtype), pd.DataFrame(), n_samples_out

    def _prepare_all_variants(self) -> Tuple[Optional[int], None, None]:
        """
        Prepare to load all variants.

        Returns:
        --------
        tuple
            (variant_count, None, None) - for all variants we just need the count
        """
        try:
            variant_count = sum(1 for _ in self.bgen_file)
            self._open_bgen()  # Re-open to reset iterator
            return variant_count, None, None
        except:
            return None, None, None

    def _prepare_region_variants(
        self, chrom: str, start_pos: int, end_pos: int
    ) -> Tuple[str, Tuple[int, int], None]:
        """
        Prepare to load variants from a region.

        Returns:
        --------
        tuple
            (normalized_chrom, (start_pos, end_pos), None)
        """
        search_chrom = _normalize_chromosome(chrom)
        return search_chrom, (start_pos, end_pos), None

    def _prepare_filtered_variants(
        self, variant_filter: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], List[int], Dict[int, int]]:
        """
        Prepare to load filtered variants.

        Returns:
        --------
        tuple
            (filtered_variant_info, bgen_indices, indices_to_output)
        """
        # First pass: collect variant info to find matches
        logger.info("First pass: scanning for variants matching filter...")
        all_variant_info = []
        variant_to_bgen_idx = {}
        rsid_to_bgen_idx = {}

        for i, variant in enumerate(self.bgen_file):
            var_info = _extract_variant_info(variant, i)
            all_variant_info.append(var_info)

            # Create lookup keys
            alleles_sorted = tuple(sorted([var_info["ref"], var_info["alt"]]))
            key = (var_info["pos"], alleles_sorted)
            variant_to_bgen_idx[key] = i
            rsid_to_bgen_idx[var_info["id"]] = i

        # Find which variants from filter are in BGEN
        matches = []

        for z_idx, (pos, a1, a2, rsid) in enumerate(
            zip(
                variant_filter["positions"],
                variant_filter["allele1"],
                variant_filter["allele2"],
                variant_filter["rsids"],
            )
        ):
            alleles_sorted = tuple(sorted([a1, a2]))
            key = (pos, alleles_sorted)

            if key in variant_to_bgen_idx:
                matches.append((z_idx, variant_to_bgen_idx[key]))
            elif rsid in rsid_to_bgen_idx:
                matches.append((z_idx, rsid_to_bgen_idx[rsid]))

        if not matches:
            raise ValueError("No variants from filter found in BGEN file")

        logger.info(
            f"Found {len(matches)} out of {len(variant_filter['positions'])} variants from filter"
        )

        # Sort matches by z_idx to maintain order
        matches.sort(key=lambda x: x[0])
        z_indices, bgen_indices = zip(*matches)

        # Get filtered variant info in correct order
        filtered_variant_info = [all_variant_info[idx] for idx in bgen_indices]

        # Create mapping for quick lookup
        indices_to_output = {idx: i for i, idx in enumerate(bgen_indices)}

        self._open_bgen()  # Reset for second pass

        return filtered_variant_info, list(bgen_indices), indices_to_output

    def _load_dosages(
        self,
        loading_mode: str,
        variant_count: Optional[int] = None,
        region_params: Optional[Tuple[str, Tuple[int, int]]] = None,
        filtered_params: Optional[Tuple[List[Dict], List[int], Dict[int, int]]] = None,
        dtype: np.dtype = np.float64,
        sample_indices: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, pd.DataFrame, int]:
        """
        Unified method to load dosages based on loading mode.

        Parameters:
        -----------
        loading_mode : str
            'all', 'region', or 'filtered'
        variant_count : int, optional
            For 'all' mode - number of variants
        region_params : tuple, optional
            For 'region' mode - (search_chrom, (start_pos, end_pos))
        filtered_params : tuple, optional
            For 'filtered' mode - (variant_info, indices, mapping)
        dtype : numpy.dtype
            Data type for dosages
        sample_indices : list of int, optional
            Sample filtering
        """
        n_samples_out = self._determine_n_samples_out(sample_indices)

        if loading_mode == "all":
            # Load all variants
            if variant_count is not None:
                # Pre-allocated approach
                logger.debug(f"Pre-allocating arrays for {variant_count} variants")
                dosages = np.empty((n_samples_out, variant_count), dtype=dtype)
                variant_info_list = []

                for i, variant in enumerate(self.bgen_file):
                    variant_info_list.append(_extract_variant_info(variant, i))
                    self._process_variant_dosage(variant, i, dosages, i, sample_indices)

                variant_info = pd.DataFrame(variant_info_list)
                return dosages, variant_info, n_samples_out
            else:
                # Fallback approach
                logger.debug("Using fallback approach without pre-allocation")
                dosage_list = []
                variant_info_list = []

                for i, variant in enumerate(self.bgen_file):
                    variant_info_list.append(_extract_variant_info(variant, i))

                    dosage = _compute_dosage_from_variant(variant)
                    if sample_indices is not None:
                        dosage = dosage[sample_indices]
                    dosage_list.append(dosage)

                if not dosage_list:
                    return self._empty_result(n_samples_out, dtype)

                dosages = np.column_stack(dosage_list).astype(dtype)
                variant_info = pd.DataFrame(variant_info_list)
                return dosages, variant_info, n_samples_out

        elif loading_mode == "region":
            # Load region variants
            search_chrom, (start_pos, end_pos) = region_params
            logger.debug(f"Loading variants from region {search_chrom}:{start_pos}-{end_pos}")

            variants_in_region = list(self.bgen_file.fetch(search_chrom, start_pos, end_pos))

            if not variants_in_region:
                logger.warning(f"No variants found in region {search_chrom}:{start_pos}-{end_pos}")
                return self._empty_result(n_samples_out, dtype)

            # Pre-allocate arrays
            n_variants = len(variants_in_region)
            dosages = np.empty((n_samples_out, n_variants), dtype=dtype)
            variant_info_list = []

            # Process variants
            for i, variant in enumerate(variants_in_region):
                variant_info_list.append(_extract_variant_info(variant, i))
                self._process_variant_dosage(variant, i, dosages, i, sample_indices)

            variant_info = pd.DataFrame(variant_info_list)
            return dosages, variant_info, n_samples_out

        elif loading_mode == "filtered":
            # Load filtered variants
            filtered_variant_info, bgen_indices, indices_to_output = filtered_params

            logger.info("Second pass: loading dosages for matched variants...")

            # Pre-allocate arrays
            n_variants = len(bgen_indices)
            dosages = np.empty((n_samples_out, n_variants), dtype=dtype)

            # Process only the variants we need
            for i, variant in enumerate(self.bgen_file):
                if i in indices_to_output:
                    output_idx = indices_to_output[i]
                    self._process_variant_dosage(variant, i, dosages, output_idx, sample_indices)

                    # Stop if we've processed all needed variants
                    if output_idx == n_variants - 1:
                        break

            variant_info = pd.DataFrame(filtered_variant_info)
            return dosages, variant_info, n_samples_out

        else:
            raise ValueError(f"Unknown loading mode: {loading_mode}")

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
    sample_ids : list of str, optional
        Sample IDs to keep. If None, all samples are loaded.
    dtype : numpy.dtype, optional
        Data type for the dosage array (default: np.float64)

    Returns:
    --------
    tuple
        (genotypes, variant_info, sample_ids)
        Note: genotypes are returned as floating point values of the specified dtype
        If variant_filter is provided, variants are ordered according to the .z file order
        If sample_ids is provided, only those samples are returned
    """
    # Open BGEN file
    bgen_reader = BgenFileReader(
        file_path=file_path, index_path=index_path, sample_path=sample_path
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
            # Prepare filtered variants
            logger.info("Loading filtered variants matching .z file")
            filtered_variant_info, bgen_indices, indices_to_output = (
                bgen_reader._prepare_filtered_variants(variant_filter)
            )

            # Load dosages
            dosages, variant_info, n_samples = bgen_reader._load_dosages(
                "filtered",
                filtered_params=(filtered_variant_info, bgen_indices, indices_to_output),
                dtype=dtype,
                sample_indices=sample_indices,
            )

        elif region is not None:
            # Parse region string
            from ..utils.region_utils import parse_region

            chrom, pos_range = parse_region(region)
            start_pos, end_pos = pos_range
            logger.info(f"Loading region: {chrom}:{start_pos}-{end_pos}")

            # Prepare region parameters
            search_chrom, region_bounds, _ = bgen_reader._prepare_region_variants(
                chrom, start_pos, end_pos
            )

            # Load dosages
            dosages, variant_info, n_samples = bgen_reader._load_dosages(
                "region",
                region_params=(search_chrom, region_bounds),
                dtype=dtype,
                sample_indices=sample_indices,
            )
        else:
            # Prepare all variants
            logger.info("Loading all variants from BGEN file")
            variant_count, _, _ = bgen_reader._prepare_all_variants()

            # Load dosages
            dosages, variant_info, n_samples = bgen_reader._load_dosages(
                "all", variant_count=variant_count, dtype=dtype, sample_indices=sample_indices
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

        if np.any(np.isnan(dosages)):
            _report_nan_error(dosages, variant_info, filtered_sample_ids)

        return dosages, variant_info, filtered_sample_ids

    except Exception as e:
        logger.error(f"Error loading BGEN file: {e}")
        raise
    finally:
        bgen_reader.close()
