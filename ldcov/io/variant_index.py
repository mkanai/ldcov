"""Variant ↔ BlockMatrix-index lookup backed by a Parquet variant index."""

from typing import Optional, Tuple

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.compute as pc

from ldcov.io.fs_utils import resolve_filesystem


class VariantIndex:
    """Map (contig, position, ref, alt) <-> BlockMatrix idx using a Parquet variant index."""

    def __init__(self, parquet_path: str, storage_options: Optional[dict] = None):
        self.parquet_path = parquet_path
        fs, path = resolve_filesystem(parquet_path, storage_options)
        self._dataset = ds.dataset(path, format="parquet", filesystem=fs)

    def query_region(self, chrom: str, start: int, end: int) -> pd.DataFrame:
        """All variant-index rows on `chrom` with start <= position <= end, sorted by idx."""
        flt = (
            (pc.field("contig") == str(chrom))
            & (pc.field("position") >= int(start))
            & (pc.field("position") <= int(end))
        )
        df = self._dataset.to_table(filter=flt).to_pandas()
        return df.sort_values("idx").reset_index(drop=True)

    def by_idx_range(self, start: int, end: int) -> pd.DataFrame:
        """Variant-index rows with start <= idx < end, sorted by idx."""
        flt = (pc.field("idx") >= int(start)) & (pc.field("idx") < int(end))
        df = self._dataset.to_table(filter=flt).to_pandas()
        return df.sort_values("idx").reset_index(drop=True)

    def match(
        self, chrom: str, pos: int, ref: str, alt: str
    ) -> Tuple[Optional[int], Optional[bool]]:
        """Return (idx, flip) for one variant, or (None, None) if not found.

        flip is True when the variant index stores ref/alt swapped relative to the query.
        """
        flt = (pc.field("contig") == str(chrom)) & (pc.field("position") == int(pos))
        df = self._dataset.to_table(columns=["ref", "alt", "idx"], filter=flt).to_pandas()
        for _, row in df.iterrows():
            if row["ref"] == ref and row["alt"] == alt:
                return int(row["idx"]), False
            if row["ref"] == alt and row["alt"] == ref:
                return int(row["idx"]), True
        return None, None
