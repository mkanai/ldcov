"""Variant ↔ BlockMatrix-index lookup backed by a Parquet variant index."""

from typing import Optional, Tuple

import numpy as np
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
        self, chrom: str, pos: int, ref: str, alt: str, allow_allele_swap: bool = False
    ) -> Tuple[Optional[int], Optional[bool]]:
        """Return (idx, flip) for one variant, or (None, None) if not found.

        With ``allow_allele_swap=False`` (default) only the exact ``(ref, alt)`` orientation
        matches, ``flip`` is always False, and a variant present only in swapped orientation
        returns ``(None, None)``. With ``allow_allele_swap=True`` the query is also matched
        against the ``(alt, ref)`` swap, and ``flip`` is True for such a match.
        """
        flt = (pc.field("contig") == str(chrom)) & (pc.field("position") == int(pos))
        df = self._dataset.to_table(columns=["ref", "alt", "idx"], filter=flt).to_pandas()
        for _, row in df.iterrows():
            if row["ref"] == ref and row["alt"] == alt:
                return int(row["idx"]), False
            if allow_allele_swap and row["ref"] == alt and row["alt"] == ref:
                return int(row["idx"]), True
        return None, None

    def match_variants(
        self, variants: pd.DataFrame, allow_allele_swap: bool = False
    ) -> pd.DataFrame:
        """Match a batch of variants to BlockMatrix indices with one query per contig.

        A vectorized counterpart to :meth:`match` for locus-scale work: instead of one
        pyarrow scan per variant, it issues a single ``query_region`` per distinct contig
        and maps every row against an in-memory lookup.

        Parameters
        ----------
        variants : pandas.DataFrame
            Must contain columns ``contig``, ``position``, ``ref``, ``alt`` (extras ignored).
        allow_allele_swap : bool
            As in :meth:`match`. Default False returns only exact-orientation matches; set
            True to also match the ref/alt swap (reported via ``flip``).

        Returns
        -------
        pandas.DataFrame
            Aligned row-for-row with ``variants`` (same index), with columns:
            ``idx`` (int64; -1 where unmatched) and ``flip`` (bool; True where the panel
            stores ref/alt swapped relative to the query, always False when
            ``allow_allele_swap=False``).
        """
        required = {"contig", "position", "ref", "alt"}
        missing = required - set(variants.columns)
        if missing:
            raise ValueError(
                f"match_variants requires columns {sorted(required)}; missing {sorted(missing)}"
            )

        n = len(variants)
        idx = np.full(n, -1, dtype=np.int64)
        flip = np.zeros(n, dtype=bool)

        contig = variants["contig"].astype(str).to_numpy()
        position = variants["position"].to_numpy()
        ref = variants["ref"].astype(str).to_numpy()
        alt = variants["alt"].astype(str).to_numpy()

        for c in np.unique(contig):
            rows = np.where(contig == c)[0]
            pos = position[rows]
            panel = self.query_region(c, int(pos.min()), int(pos.max()))
            forward = {}
            reverse = {}
            for p, r, a, gi in zip(
                panel["position"].tolist(),
                panel["ref"].tolist(),
                panel["alt"].tolist(),
                panel["idx"].tolist(),
            ):
                # First occurrence wins (index is expected biallelic-unique per orientation).
                forward.setdefault((int(p), r, a), int(gi))
                if allow_allele_swap:
                    reverse.setdefault((int(p), a, r), int(gi))
            for i in rows:
                key = (int(position[i]), ref[i], alt[i])
                if key in forward:
                    idx[i] = forward[key]
                elif key in reverse:
                    idx[i] = reverse[key]
                    flip[i] = True

        return pd.DataFrame({"idx": idx, "flip": flip}, index=variants.index)
