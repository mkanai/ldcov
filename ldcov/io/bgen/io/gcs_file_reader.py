"""Python wrapper for GCS file reader functionality."""

import os
from typing import Optional


class GCSFileReader:
    """GCS file reader implementation for BGEN files."""

    def __init__(self, path: str):
        """Initialize GCS file reader.

        Args:
            path: GCS path (must start with gs://)
        """
        if not path.startswith("gs://"):
            raise ValueError(f"Path must start with gs://: {path}")

        self.path = path
        self.fs = None  # Will be initialized lazily
        self._file = None
        self._buffer_size = 10 * 1024 * 1024  # 10MB buffer as per Phase 1 optimization

    def _ensure_fs(self):
        """Lazily initialize GCS filesystem."""
        if self.fs is None:
            import gcsfs
            self.fs = gcsfs.GCSFileSystem()
    
    def open(self):
        """Open the GCS file."""
        self._ensure_fs()
        self._file = self.fs.open(self.path, "rb")

    def close(self):
        """Close the GCS file."""
        if self._file:
            self._file.close()
            self._file = None

    def read(self, size: int) -> bytes:
        """Read bytes from the file."""
        if not self._file:
            raise RuntimeError("File not opened")
        return self._file.read(size)

    def seek(self, offset: int, whence: int = 0):
        """Seek to position in file."""
        if not self._file:
            raise RuntimeError("File not opened")
        self._file.seek(offset, whence)

    def tell(self) -> int:
        """Get current position in file."""
        if not self._file:
            raise RuntimeError("File not opened")
        return self._file.tell()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def create_file_reader(path: str):
    """Factory function to create appropriate file reader based on path.

    Args:
        path: File path (local or GCS)

    Returns:
        Appropriate file reader instance
    """
    if path.startswith("gs://"):
        return GCSFileReader(path)
    else:
        # For local files, return a simple file object
        # This would normally use the C++ readers but for testing we'll use Python
        return open(path, "rb")
