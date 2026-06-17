"""Filesystem-agnostic reader for a Hail BlockMatrix directory."""

import json
import threading
from collections import OrderedDict
from typing import Optional

import numpy as np

from ldcov.io.fs_utils import resolve_filesystem

from .block_codec import decode_block
from .partitioner import GridPartitioner


class HailBlockMatrixReader:
    """Read blocks of a Hail BlockMatrix from local or remote storage.

    Remote backends are resolved via fsspec (gs:// needs gcsfs, s3:// needs s3fs).
    """

    def __init__(self, path: str, storage_options: Optional[dict] = None, block_cache: int = 4):
        self.path = path.rstrip("/")
        self._fs, _ = resolve_filesystem(self.path, storage_options)
        self._cache_size = max(1, block_cache)
        self._cache = OrderedDict()  # (i, j) -> ndarray
        self._lock = threading.Lock()

        with self._fs.open(self.path + "/metadata.json", "rb") as fh:
            meta = json.loads(fh.read().decode("utf-8"))
        self.block_size = meta["blockSize"]
        self.n_rows = meta["nRows"]
        self.n_cols = meta["nCols"]
        self._part_files = meta["partFiles"]
        self.partitioner = GridPartitioner(
            self.block_size, self.n_rows, self.n_cols, meta.get("maybeFiltered")
        )

    def read_block(self, i: int, j: int) -> Optional[np.ndarray]:
        """Return block (i, j) as a 2D float64 array, or None if not stored.

        Thread-safe: the LRU cache is guarded by a lock, but the fetch+decode runs
        outside the lock so concurrent reads of different blocks proceed in parallel.
        """
        key = (i, j)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        slot = self.partitioner.part_slot(i, j)
        if slot is None:
            return None
        part_path = self.path + "/parts/" + self._part_files[slot]
        with self._fs.open(part_path, "rb") as fh:
            raw = fh.read()
        block = decode_block(raw)
        with self._lock:
            self._cache[key] = block
            self._cache.move_to_end(key)
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return block
