"""Memory-mapped file reader for BGEN files."""

import mmap
import os
from typing import Optional, Union


class MMapBgenFile:
    """Memory-mapped BGEN file reader for efficient random access."""

    def __init__(self, filepath: str):
        """
        Initialize memory-mapped BGEN file.

        Parameters
        ----------
        filepath : str
            Path to BGEN file
        """
        self.filepath = filepath
        self.file = None
        self.mmap = None
        self._position = 0
        self._length = 0

        # Open file and create memory map
        self._open()

    def _open(self):
        """Open file and create memory map."""
        self.file = open(self.filepath, "rb")
        self._length = os.path.getsize(self.filepath)

        # Create memory map
        # Use ACCESS_READ for read-only access
        if self._length > 0:
            self.mmap = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        else:
            raise ValueError(f"Empty BGEN file: {self.filepath}")

    def seek(self, offset: int, whence: int = 0) -> int:
        """
        Seek to position in file.

        Parameters
        ----------
        offset : int
            Offset to seek to
        whence : int
            0 = absolute, 1 = relative to current, 2 = relative to end

        Returns
        -------
        int
            New position
        """
        if whence == 0:  # Absolute
            self._position = offset
        elif whence == 1:  # Relative to current
            self._position += offset
        elif whence == 2:  # Relative to end
            self._position = self._length + offset
        else:
            raise ValueError(f"Invalid whence value: {whence}")

        # Clamp to valid range
        self._position = max(0, min(self._position, self._length))
        return self._position

    def tell(self) -> int:
        """Get current position."""
        return self._position

    def read(self, size: Optional[int] = None) -> bytes:
        """
        Read bytes from current position.

        Parameters
        ----------
        size : int, optional
            Number of bytes to read. If None, read to end.

        Returns
        -------
        bytes
            Data read
        """
        if size is None:
            size = self._length - self._position

        # Ensure we don't read past end
        size = min(size, self._length - self._position)

        if size <= 0:
            return b""

        # Read from memory map
        data = self.mmap[self._position : self._position + size]
        self._position += size

        return data

    def read_at(self, offset: int, size: int) -> bytes:
        """
        Read bytes at specific offset without changing position.

        Parameters
        ----------
        offset : int
            Offset to read from
        size : int
            Number of bytes to read

        Returns
        -------
        bytes
            Data read
        """
        if offset < 0 or offset >= self._length:
            raise ValueError(f"Invalid offset: {offset}")

        size = min(size, self._length - offset)
        return self.mmap[offset : offset + size]

    def close(self):
        """Close memory map and file."""
        if self.mmap is not None:
            self.mmap.close()
            self.mmap = None
        if self.file is not None:
            self.file.close()
            self.file = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()

    @property
    def closed(self) -> bool:
        """Check if file is closed."""
        return self.file is None or self.file.closed
