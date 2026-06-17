"""Grid partitioner mirroring Hail's column-major block numbering."""

from typing import List, Optional


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


class GridPartitioner:
    def __init__(
        self, block_size: int, n_rows: int, n_cols: int, maybe_filtered: Optional[List[int]]
    ):
        self.block_size = block_size
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_block_rows = _ceil_div(n_rows, block_size)
        self.n_block_cols = _ceil_div(n_cols, block_size)
        self.maybe_filtered = maybe_filtered
        if maybe_filtered is None:
            self._slot_of = None
        else:
            self._slot_of = {linear: slot for slot, linear in enumerate(maybe_filtered)}

    def block_of(self, index: int) -> int:
        """Block index along a single axis for a global row/col index."""
        return index // self.block_size

    def linear_id(self, i: int, j: int) -> int:
        """Column-major linear block id for block (i, j)."""
        return i + j * self.n_block_rows

    def part_slot(self, i: int, j: int) -> Optional[int]:
        """Part-file slot for block (i, j), or None if not stored on disk."""
        if i < 0 or j < 0 or i >= self.n_block_rows or j >= self.n_block_cols:
            return None
        linear = self.linear_id(i, j)
        if self._slot_of is None:
            return linear
        return self._slot_of.get(linear)
