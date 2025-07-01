"""
Batch decompression for BGEN reader to improve performance.

This module implements asynchronous read-ahead decompression to overlap
I/O and decompression operations with computation.

Provides both Python (ThreadPoolExecutor) and Cython (C++ threads) backends.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, Future
from typing import List, Optional, Callable, Tuple
import numpy as np
import os

logger = logging.getLogger(__name__)

# Try to import Cython implementation
try:
    from .._decompressor import CythonBatchDecompressor, CythonSequentialDecompressor

    HAS_CYTHON_DECOMPRESSOR = True
    HAS_SEQUENTIAL_DECOMPRESSOR = True
except ImportError:
    try:
        from .._decompressor import CythonBatchDecompressor

        HAS_CYTHON_DECOMPRESSOR = True
        HAS_SEQUENTIAL_DECOMPRESSOR = False
    except ImportError:
        HAS_CYTHON_DECOMPRESSOR = False
        HAS_SEQUENTIAL_DECOMPRESSOR = False
    logger.debug("Cython batch decompressor not available, using Python implementation")


class ReadAheadDecompressor:
    """
    Simple read-ahead decompressor for BGEN variants.

    This implementation prefetches and decompresses the next variant while
    the current one is being processed, providing ~10-20% speedup for
    compressed BGEN files.
    """

    def __init__(self, bgen_reader, max_workers: int = 1):
        """
        Initialize the read-ahead decompressor.

        Parameters
        ----------
        bgen_reader : BgenReader
            The BGEN reader instance
        max_workers : int
            Number of worker threads (default: 1 for simple read-ahead)
        """
        self.bgen_reader = bgen_reader
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.prefetch_future: Optional[Future] = None
        self._closed = False

    def _read_and_decompress_variant(self, offset: int):
        """
        Read and decompress a single variant.

        This method is run in a separate thread to overlap with computation.

        Parameters
        ----------
        offset : int
            File offset for the variant

        Returns
        -------
        BgenVariant
            The loaded variant object
        """
        try:
            # Create variant object using reader's method
            variant = self.bgen_reader.create_variant_at_offset(offset)

            # Force decompression by accessing alt_dosage
            # This triggers _load_genotypes() which performs decompression in the thread
            _ = variant.alt_dosage

            return variant
        except Exception as e:
            logger.warning(f"Failed to prefetch variant at offset {offset}: {e}")
            return None

    def process_variants_with_readahead(
        self,
        offsets: List[int],
        sample_indices: Optional[np.ndarray] = None,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> Tuple[np.ndarray, List]:
        """
        Process variants with read-ahead decompression.

        Parameters
        ----------
        offsets : List[int]
            File offsets for variants to process
        sample_indices : np.ndarray, optional
            Sample indices to extract
        progress_callback : callable, optional
            Function to call with progress updates

        Returns
        -------
        Tuple[np.ndarray, List]
            (dosages, variant_info)
        """
        n_variants = len(offsets)
        if n_variants == 0:
            n_samples = (
                len(sample_indices) if sample_indices is not None else self.bgen_reader.nsamples
            )
            return np.empty((n_samples, 0), dtype=np.float64), []

        # Determine output dimensions
        n_samples_out = (
            len(sample_indices) if sample_indices is not None else self.bgen_reader.nsamples
        )

        # Pre-allocate output array
        dosages = np.empty((n_samples_out, n_variants), dtype=np.float64)
        variant_info = []

        # Start prefetching first variant
        if n_variants > 0:
            self.prefetch_future = self.executor.submit(
                self._read_and_decompress_variant, offsets[0]
            )

        for i in range(n_variants):
            # Get current variant (either from prefetch or read now)
            if self.prefetch_future is not None:
                variant = self.prefetch_future.result()
                if variant is None:
                    # Fallback to synchronous read on error
                    variant = self.bgen_reader.create_variant_at_offset(offsets[i])
            else:
                # No prefetch, read synchronously
                from .variant import BgenVariant

                variant = BgenVariant(
                    self.bgen_reader.file_handle,
                    offsets[i],
                    self.bgen_reader.layout,
                    self.bgen_reader.compression,
                    self.bgen_reader.nsamples,
                )

            # Start prefetching next variant (if available)
            if i + 1 < n_variants:
                self.prefetch_future = self.executor.submit(
                    self._read_and_decompress_variant, offsets[i + 1]
                )
            else:
                self.prefetch_future = None

            # Process current variant
            if sample_indices is not None:
                dosages[:, i] = variant.get_dosage_for_samples(sample_indices)
            else:
                dosages[:, i] = variant.alt_dosage

            # Collect variant info
            variant_info.append(
                {
                    "varid": variant.varid,
                    "rsid": variant.rsid,
                    "chrom": variant.chrom,
                    "pos": variant.pos,
                    "alleles": variant.alleles,
                }
            )

            # Update progress
            if progress_callback is not None:
                progress_callback(i + 1)

        return dosages, variant_info

    def close(self):
        """Clean up resources."""
        if not self._closed:
            self.executor.shutdown(wait=True)
            self._closed = True

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Destructor."""
        self.close()


class BatchDecompressor:
    """
    Adaptive batch decompressor that selects the best backend.

    Automatically chooses between:
    - CythonSequentialDecompressor: C++ sequential implementation for sequential access
    - CythonBatchDecompressor: C++ batch implementation for random access
    - ReadAheadDecompressor: Python implementation as fallback

    Selection is based on:
    - Availability of Cython modules
    - File size and compression type
    - Access pattern (sequential vs random)
    - Environmental hints
    """

    def __init__(self, bgen_reader, batch_size: int = 50, backend: str = "auto"):
        """
        Initialize batch decompressor.

        Parameters
        ----------
        bgen_reader : BgenReader
            The BGEN reader instance
        batch_size : int
            Number of variants to process in each batch
        backend : str
            Backend selection: 'auto', 'sequential', 'batch'/'cython', or 'python'
            - 'auto': Automatically select based on file characteristics and access pattern
            - 'sequential': Use C++ sequential decompressor (if available)
            - 'batch'/'cython': Use C++ batch decompressor (if available)
            - 'python': Use Python read-ahead implementation
            Can also be set via LDCOV_DECOMPRESSOR_BACKEND environment variable
        """
        self.bgen_reader = bgen_reader
        self.batch_size = batch_size
        self.backend = backend
        self._impl = None

        # Select implementation
        self._select_implementation()

    def _select_implementation(self):
        """Select the best decompressor implementation."""
        # Check environment variable if backend is 'auto'
        backend = self.backend
        if backend == "auto":
            env_backend = os.environ.get("LDCOV_DECOMPRESSOR_BACKEND")
            if env_backend:
                backend = env_backend
                logger.debug(f"Using decompressor backend from environment: {backend}")

        # Initialize backend selection flags
        self._use_sequential = False
        self._use_batch = False
        self._use_python = False

        # Check explicit backend selection
        if backend == "sequential":
            if not HAS_SEQUENTIAL_DECOMPRESSOR:
                raise RuntimeError("Sequential decompressor requested but not available")
            self._use_sequential = True
            logger.debug("Using C++ sequential decompressor")
        elif backend == "cython" or backend == "batch":
            if not HAS_CYTHON_DECOMPRESSOR:
                raise RuntimeError("Batch decompressor requested but not available")
            self._use_batch = True
            logger.debug("Using C++ batch decompressor")
        elif backend == "python":
            self._use_python = True
            logger.debug("Using Python read-ahead decompressor")
        else:  # auto
            # Auto-select based on file characteristics and availability
            if HAS_SEQUENTIAL_DECOMPRESSOR or HAS_CYTHON_DECOMPRESSOR:
                # Use benchmark-based selection logic
                try:
                    file_size = os.path.getsize(self.bgen_reader.file_path)
                    file_size_mb = file_size / (1024 * 1024)
                    compression_type = self.bgen_reader.compression

                    # Determine if we should use C++ decompressor
                    use_cpp = False

                    # Biobank-scale always benefits from C++
                    if file_size_mb > 3000:
                        use_cpp = True
                        logger.debug(
                            f"Using C++ decompressor for biobank-scale {file_size_mb:.1f}MB file"
                        )
                    # Small files: Python is faster
                    elif file_size_mb < 100:
                        use_cpp = False
                        logger.debug(f"Using Python for small {file_size_mb:.1f}MB file")
                    elif compression_type == 0:  # uncompressed
                        # Marginal benefit for medium files
                        use_cpp = 100 <= file_size_mb <= 500
                        logger.debug(
                            f"Using {'C++' if use_cpp else 'Python'} for uncompressed {file_size_mb:.1f}MB file"
                        )
                    elif compression_type == 1:  # zlib
                        # C++ beneficial for medium files
                        use_cpp = 100 <= file_size_mb <= 500
                        logger.debug(
                            f"Using {'C++' if use_cpp else 'Python'} for zlib {file_size_mb:.1f}MB file"
                        )
                    elif compression_type == 2:  # zstd
                        # Only beneficial for very large files
                        use_cpp = file_size_mb > 900
                        logger.debug(
                            f"Using {'C++' if use_cpp else 'Python'} for zstd {file_size_mb:.1f}MB file"
                        )
                    else:
                        # Unknown compression, use Python
                        use_cpp = False
                        logger.debug(
                            f"Using Python for unknown compression type {compression_type}"
                        )

                    if use_cpp:
                        # For auto mode, prefer sequential for sequential access patterns
                        # This is a heuristic - could be improved with actual access pattern detection
                        if HAS_SEQUENTIAL_DECOMPRESSOR:
                            self._use_sequential = True
                            logger.debug("Auto-selected C++ sequential decompressor")
                        else:
                            self._use_batch = True
                            logger.debug("Auto-selected C++ batch decompressor")
                    else:
                        self._use_python = True

                except:
                    # Default to Python if we can't determine file size
                    self._use_python = True
                    logger.debug(
                        "Using Python read-ahead (failed to determine file characteristics)"
                    )
            else:
                self._use_python = True
                logger.debug("Using Python read-ahead (no C++ decompressor available)")

    def process_variants(
        self,
        offsets: List[int],
        variant_metadata: List = None,
        sample_indices: Optional[np.ndarray] = None,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> Tuple[np.ndarray, List]:
        """
        Process variants using the selected backend.

        Parameters
        ----------
        offsets : List[int]
            File offsets for variants
        variant_metadata : List, optional
            Metadata for variants (from BGI)
        sample_indices : np.ndarray, optional
            Sample indices to extract
        progress_callback : callable, optional
            Progress callback

        Returns
        -------
        Tuple[np.ndarray, List]
            (dosages, variant_info)
        """
        if self._use_sequential and variant_metadata is not None:
            # Use sequential implementation
            if self._impl is None:
                self._impl = CythonSequentialDecompressor(self.bgen_reader)
            return self._impl.process_variants_sequentially(
                offsets, variant_metadata, sample_indices, progress_callback
            )
        elif self._use_batch and variant_metadata is not None:
            # Use batch implementation
            if self._impl is None:
                num_threads = min(4, max(2, os.cpu_count() // 2))
                self._impl = CythonBatchDecompressor(
                    self.bgen_reader, num_threads=num_threads, batch_size=self.batch_size
                )
            return self._impl.process_variants_with_batch(
                offsets, variant_metadata, sample_indices, progress_callback
            )
        else:
            # Use Python implementation
            if self._impl is None:
                self._impl = ReadAheadDecompressor(self.bgen_reader)
            return self._impl.process_variants_with_readahead(
                offsets, sample_indices, progress_callback
            )

    def close(self):
        """Clean up resources."""
        if self._impl is not None:
            self._impl.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def should_use_decompressor(
    file_size_mb: float, compression_type: int, access_pattern: str = "sequential"
) -> Tuple[bool, str]:
    """
    Determine if C++ decompression should be used and which type.

    Parameters
    ----------
    file_size_mb : float
        Size of BGEN file in MB
    compression_type : int
        0=none, 1=zlib, 2=zstd
    access_pattern : str
        'sequential' or 'random'

    Returns
    -------
    Tuple[bool, str]
        (use_cpp, decompressor_type) where decompressor_type is 'sequential', 'batch', or 'python'
    """
    # Based on comprehensive benchmark results from 2025-06-14

    # Random access always benefits from C++ batch decompressor
    if access_pattern == "random":
        return True, "batch"

    # Biobank-scale always benefits
    if file_size_mb > 3000:
        # Use sequential for sequential access, batch for random
        return True, "sequential" if access_pattern == "sequential" else "batch"

    # Small files: Python is faster
    if file_size_mb < 100:
        return False, "python"

    if compression_type == 0:  # uncompressed
        # Marginal benefit, but use C++ for medium files
        use_cpp = 100 <= file_size_mb <= 500
        return use_cpp, "sequential" if use_cpp else "python"

    elif compression_type == 1:  # zlib
        # C++ beneficial for medium files
        use_cpp = 100 <= file_size_mb <= 500
        return use_cpp, "sequential" if use_cpp else "python"

    elif compression_type == 2:  # zstd
        # Only beneficial for very large files
        use_cpp = file_size_mb > 900
        return use_cpp, "sequential" if use_cpp else "python"

    # Default: use Python
    return False, "python"
