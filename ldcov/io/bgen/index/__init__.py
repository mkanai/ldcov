"""BGI index module."""

import os
import tempfile
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class BGICache:
    """Simple BGI cache implementation for GCS paths."""
    
    _instance = None
    
    def __init__(self):
        self._cache: Dict[str, str] = {}
        self._cache_dir = os.path.join(tempfile.gettempdir(), "ldcov_bgi_cache")
        os.makedirs(self._cache_dir, exist_ok=True)
        
    @classmethod
    def getInstance(cls):
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
        
    def get_local_path(self, gcs_path: str) -> str:
        """Get local path for BGI file, downloading if needed.
        
        Args:
            gcs_path: GCS path to BGI file
            
        Returns:
            Local path to BGI file
        """
        # For local paths, just return as-is
        if not gcs_path.startswith("gs://"):
            return gcs_path
            
        # Check memory cache first
        if gcs_path in self._cache:
            return self._cache[gcs_path]
            
        # Determine local path
        basename = os.path.basename(gcs_path)
        local_path = os.path.join(os.getcwd(), basename)
        
        # Check if already downloaded
        if os.path.exists(local_path):
            self._cache[gcs_path] = local_path
            return local_path
            
        # Download from GCS
        logger.info(f"Downloading BGI file from {gcs_path} to {local_path}")
        try:
            import gcsfs
            fs = gcsfs.GCSFileSystem()
            fs.get(gcs_path, local_path)
            self._cache[gcs_path] = local_path
            return local_path
        except Exception as e:
            raise RuntimeError(f"Failed to download BGI file from {gcs_path}: {e}")


# Create module-level instance
bgi_cache = BGICache.getInstance()