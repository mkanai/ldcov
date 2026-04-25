"""File-handle abstraction supporting local files (with optional mmap) and GCS via gcsfs."""

import mmap
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Sequence, Tuple, Union


class BcorFileHandle:
    _LOCAL_MMAP_MIN_BYTES = 100 * 1024 * 1024  # 100 MB
    _DEFAULT_GCS_CONCURRENCY = 8

    def __init__(
        self,
        path: str,
        use_mmap: Optional[bool] = None,
        gcs_concurrency: int = _DEFAULT_GCS_CONCURRENCY,
        gcs_fs=None,
    ):
        """gcs_fs: an optional pre-constructed gcsfs.GCSFileSystem instance to reuse
        (avoids repeated auth setup when opening sidecars adjacent to a parent .bcor)."""
        self._path = path
        self._is_remote = path.startswith("gs://")
        self._use_mmap_hint = use_mmap
        self._fh = None
        self._mmap = None
        self._gcs_fs = gcs_fs
        self._gcs_size = None
        self._gcs_concurrency = gcs_concurrency

    def is_remote(self) -> bool:
        return self._is_remote

    @property
    def gcs_fs(self):
        """The gcsfs.GCSFileSystem in use, or None for local paths. Available after open."""
        return self._gcs_fs

    @property
    def size(self) -> int:
        if self._is_remote:
            self._ensure_open()
            return int(self._gcs_size)
        return os.path.getsize(self._path)

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _ensure_open(self):
        if self._fh is not None:
            return
        if self._is_remote:
            if self._gcs_fs is None:
                import gcsfs

                self._gcs_fs = gcsfs.GCSFileSystem()
            info = self._gcs_fs.info(self._path)
            self._gcs_size = info["size"]
            self._fh = self._gcs_fs.open(self._path, "rb")
        else:
            self._fh = open(self._path, "rb")
            use_mmap = self._use_mmap_hint
            if use_mmap is None:
                use_mmap = os.path.getsize(self._path) > self._LOCAL_MMAP_MIN_BYTES
            if use_mmap:
                try:
                    self._mmap = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
                except Exception:
                    # mmap failed (e.g. empty file, unmappable fs); release the fd we
                    # just opened so callers don't leak it when __enter__ propagates.
                    self._fh.close()
                    self._fh = None
                    raise

    def close(self):
        if self._mmap is not None:
            try:
                self._mmap.close()
            except BufferError:
                # Exported memoryviews (e.g. numpy arrays) still reference this mmap.
                # Closing is unsafe; drop our reference and let the GC collect it once
                # all consumer buffers are released.
                pass
            self._mmap = None
        if self._fh is not None:
            try:
                self._fh.close()
            finally:
                self._fh = None

    # ---- streaming API (compatible with existing BcorReader usage) ----

    def read(self, n: int) -> bytes:
        self._ensure_open()
        return self._fh.read(n)

    def seek(self, offset: int, whence: int = 0) -> None:
        self._ensure_open()
        self._fh.seek(offset, whence)

    def tell(self) -> int:
        self._ensure_open()
        return self._fh.tell()

    # ---- range API ----

    def read_range(self, offset: int, length: int) -> Union[memoryview, bytes]:
        """Return a buffer-protocol object covering [offset, offset+length).

        For mmap-backed local files, returns a zero-copy memoryview into the mmap.
        For non-mmap local files and GCS, returns a freshly read bytes object.
        Callers that need to retain data beyond the file handle's lifetime should
        wrap the result with `bytes(...)` explicitly.
        """
        self._ensure_open()
        if self._mmap is not None:
            return memoryview(self._mmap)[offset : offset + length]
        self._fh.seek(offset)
        return self._fh.read(length)

    def read_ranges(self, ranges: Sequence[Tuple[int, int]]) -> List[Union[memoryview, bytes]]:
        """Fetch multiple (offset, length) ranges. Parallel for remote handles."""
        self._ensure_open()
        if not self._is_remote or len(ranges) <= 1:
            return [self.read_range(o, l) for o, l in ranges]

        # Parallel range reads over GCS. Each worker opens its own file object so seeks don't
        # collide; gcsfs file objects are not thread-safe on a single instance. The underlying
        # GCSFileSystem auth state is shared (no re-auth per worker).
        gcs_fs = self._gcs_fs
        path = self._path

        def _fetch(ol):
            offset, length = ol
            with gcs_fs.open(path, "rb") as fh:
                fh.seek(offset)
                return fh.read(length)

        max_workers = min(self._gcs_concurrency, max(1, len(ranges)))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(_fetch, ranges))

    # ---- mmap passthrough (for legacy callers) ----

    @property
    def mmap(self):
        """Underlying mmap object (None for non-mmap local files and remote)."""
        return self._mmap
