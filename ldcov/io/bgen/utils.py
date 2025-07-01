"""Utilities for BGEN file handling."""

import os
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Global BGI memory cache to avoid repeated downloads
_bgi_memory_cache: Dict[str, str] = {}  # Maps GCS path -> local path


def ensure_local_bgi(bgi_path: str) -> str:
    """
    Ensure BGI file is available locally, downloading from GCS if needed.

    Similar to bcftools approach - downloads to current directory.
    Uses memory cache to avoid redundant checks.

    Parameters
    ----------
    bgi_path : str
        Path to BGI file (local or gs://)

    Returns
    -------
    str
        Local path to BGI file
    """
    # If already local, return as-is
    if not bgi_path.startswith("gs://"):
        return bgi_path

    # Check memory cache first
    if bgi_path in _bgi_memory_cache:
        cached_path = _bgi_memory_cache[bgi_path]
        # Verify the cached file still exists
        if os.path.exists(cached_path):
            logger.debug(f"Using BGI from memory cache: {cached_path}")
            return cached_path
        else:
            # File was removed, clear from cache
            del _bgi_memory_cache[bgi_path]

    # Extract filename from GCS path
    filename = os.path.basename(bgi_path)
    local_path = os.path.join(".", filename)

    # Check if already exists locally
    if os.path.exists(local_path):
        logger.info(f"Using existing BGI file: {local_path}")
        # Add to memory cache
        _bgi_memory_cache[bgi_path] = local_path
        return local_path

    # Download from GCS with retry logic
    max_retries = 3
    retry_delay = 1.0

    for attempt in range(max_retries):
        try:
            import gcsfs

            logger.info(f"Downloading BGI index from {bgi_path} to {local_path}...")

            fs = gcsfs.GCSFileSystem()
            fs.get(bgi_path, local_path)

            logger.info("BGI index downloaded successfully")
            # Add to memory cache
            _bgi_memory_cache[bgi_path] = local_path
            return local_path

        except ImportError:
            raise ImportError("gcsfs is required for GCS support. Install with: pip install gcsfs")
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"BGI download attempt {attempt + 1} failed: {e}. Retrying...")
                import time

                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                raise RuntimeError(
                    f"Failed to download BGI file from {bgi_path} after {max_retries} attempts: {e}"
                )


def clear_bgi_cache() -> None:
    """Clear the BGI memory cache."""
    global _bgi_memory_cache  # noqa: F824
    _bgi_memory_cache.clear()
    logger.debug("BGI memory cache cleared")


def get_bgi_cache_info() -> Dict[str, str]:
    """Get information about cached BGI files."""
    return _bgi_memory_cache.copy()
