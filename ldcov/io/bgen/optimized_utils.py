"""Optimized utilities for GCS BGEN operations."""

import os
import time
import hashlib
import logging
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, Future

logger = logging.getLogger(__name__)


class OptimizedBGICache:
    """Enhanced BGI cache with parallel downloads and persistent storage."""

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the BGI cache.

        Args:
            cache_dir: Directory for persistent cache.
                      Defaults to ~/.cache/ldcov/bgi/
        """
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/ldcov/bgi")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache for frequently accessed BGI files
        self._memory_cache: Dict[str, bytes] = {}
        self._cache_lock = threading.Lock()

        # Thread pool for parallel downloads
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._download_futures: Dict[str, Future] = {}

        # GCS filesystem instance (reused for efficiency)
        self._fs = None

    def get_bgi_path(self, bgen_path: str) -> str:
        """
        Get local path to BGI file, downloading if necessary.

        Args:
            bgen_path: Path to BGEN file (local or gs://)

        Returns:
            Local path to BGI file
        """
        if not bgen_path.startswith("gs://"):
            # Local file - just change extension
            return bgen_path + ".bgi"

        bgi_path = bgen_path + ".bgi"

        # Check memory cache first
        with self._cache_lock:
            if bgi_path in self._memory_cache:
                logger.debug(f"BGI found in memory cache: {bgi_path}")
                return self._write_temp_bgi(bgi_path, self._memory_cache[bgi_path])

        # Check persistent cache
        cache_key = self._get_cache_key(bgi_path)
        cached_path = self.cache_dir / cache_key

        if cached_path.exists():
            # Verify cache is still valid
            if self._is_cache_valid(bgi_path, cached_path):
                logger.info(f"Using cached BGI: {cached_path}")
                self._load_to_memory_cache(bgi_path, cached_path)
                return str(cached_path)

        # Download BGI file
        return self._download_bgi(bgi_path, cached_path)

    def prefetch_bgi_files(self, bgen_paths: list[str]) -> None:
        """
        Prefetch multiple BGI files in parallel.

        Args:
            bgen_paths: List of BGEN file paths
        """
        gcs_paths = [p for p in bgen_paths if p.startswith("gs://")]

        for path in gcs_paths:
            bgi_path = path + ".bgi"
            cache_key = self._get_cache_key(bgi_path)
            cached_path = self.cache_dir / cache_key

            if not cached_path.exists() or not self._is_cache_valid(bgi_path, cached_path):
                # Start async download
                if bgi_path not in self._download_futures:
                    future = self._executor.submit(self._download_bgi, bgi_path, cached_path)
                    self._download_futures[bgi_path] = future

    def _get_cache_key(self, gcs_path: str) -> str:
        """Generate cache key for GCS path."""
        return hashlib.sha256(gcs_path.encode()).hexdigest() + ".bgi"

    def _is_cache_valid(self, gcs_path: str, local_path: Path) -> bool:
        """Check if cached file is still valid."""
        if not local_path.exists():
            return False

        # Check if file is complete (not truncated)
        if local_path.stat().st_size < 100:  # BGI files should be >100 bytes
            return False

        # For now, assume cache is valid if it exists
        # Could add timestamp checking if needed
        return True

    def _download_bgi(self, gcs_path: str, local_path: Path) -> str:
        """Download BGI file from GCS with retry logic."""
        logger.info(f"Downloading BGI: {gcs_path}")

        # Initialize GCS filesystem if needed
        if self._fs is None:
            import gcsfs
            self._fs = gcsfs.GCSFileSystem()

        # Retry logic
        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                # Download to temporary file first
                temp_path = local_path.with_suffix(".tmp")

                start_time = time.time()
                self._fs.get(gcs_path, str(temp_path))
                download_time = time.time() - start_time

                # Verify download
                if temp_path.stat().st_size < 100:
                    raise ValueError(
                        f"Downloaded BGI file too small: {temp_path.stat().st_size} bytes"
                    )

                # Move to final location
                temp_path.rename(local_path)

                logger.info(
                    f"BGI download complete: {download_time:.2f}s, "
                    f"{local_path.stat().st_size / 1024:.1f}KB"
                )

                # Load into memory cache
                self._load_to_memory_cache(gcs_path, local_path)

                return str(local_path)

            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"BGI download attempt {attempt + 1} failed: {e}")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise RuntimeError(f"Failed to download BGI after {max_retries} attempts: {e}")

    def _load_to_memory_cache(self, gcs_path: str, local_path: Path) -> None:
        """Load BGI file into memory cache."""
        try:
            with open(local_path, "rb") as f:
                data = f.read()

            with self._cache_lock:
                # Limit memory cache size
                if len(self._memory_cache) >= 10:  # Keep max 10 BGI files in memory
                    # Remove oldest entry
                    oldest = next(iter(self._memory_cache))
                    del self._memory_cache[oldest]

                self._memory_cache[gcs_path] = data

        except Exception as e:
            logger.warning(f"Failed to load BGI into memory cache: {e}")

    def _write_temp_bgi(self, gcs_path: str, data: bytes) -> str:
        """Write BGI data to temporary file."""
        temp_path = Path(".") / os.path.basename(gcs_path)
        with open(temp_path, "wb") as f:
            f.write(data)
        return str(temp_path)

    def cleanup(self) -> None:
        """Clean up resources."""
        self._executor.shutdown(wait=False)
        if self._fs is not None:
            self._fs.close()


# Global cache instance
_bgi_cache = None


def get_bgi_cache() -> OptimizedBGICache:
    """Get global BGI cache instance."""
    global _bgi_cache
    if _bgi_cache is None:
        _bgi_cache = OptimizedBGICache()
    return _bgi_cache


def ensure_local_bgi_optimized(bgi_path: str) -> str:
    """
    Optimized version of ensure_local_bgi using persistent cache.

    Args:
        bgi_path: Path to BGI file (local or gs://)

    Returns:
        Local path to BGI file
    """
    return get_bgi_cache().get_bgi_path(bgi_path.rstrip(".bgi"))


def prefetch_bgi_files(bgen_paths: list[str]) -> None:
    """
    Prefetch BGI files for multiple BGEN files.

    Args:
        bgen_paths: List of BGEN file paths
    """
    get_bgi_cache().prefetch_bgi_files(bgen_paths)


class GCSOptimizationConfig:
    """Configuration for GCS-specific optimizations."""

    # Buffer sizes
    DEFAULT_BUFFER_SIZE = 1 * 1024 * 1024  # 1MB
    LARGE_FILE_BUFFER_SIZE = 10 * 1024 * 1024  # 10MB

    # Prefetch settings
    PREFETCH_ENABLED = True
    PREFETCH_SIZE = 5  # Number of variants to prefetch

    # Cache settings
    CACHE_ENABLED = True
    CACHE_SIZE = 100 * 1024 * 1024  # 100MB

    # Retry settings
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0  # seconds

    @classmethod
    def get_optimal_buffer_size(cls, file_size: int) -> int:
        """Get optimal buffer size based on file size."""
        if file_size > 100 * 1024 * 1024:  # >100MB
            return cls.LARGE_FILE_BUFFER_SIZE
        return cls.DEFAULT_BUFFER_SIZE

    @classmethod
    def should_use_parallel_reads(cls, n_variants: int, consecutive: bool) -> bool:
        """Determine if parallel reads would be beneficial."""
        # Use parallel reads for non-consecutive access of many variants
        return not consecutive and n_variants > 10
