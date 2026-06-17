"""Assemble a submatrix from a Hail BlockMatrix and export it to bcor."""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import logging
import numpy as np
import pandas as pd

from ldcov.io.blockmatrix.reader import HailBlockMatrixReader
from ldcov.io.variant_index import VariantIndex
from ldcov.io.bcor_writer import save_bcor
from ldcov.utils.region_utils import parse_region

logger = logging.getLogger(__name__)


def _as_slice(seq):
    """Return slice(start, stop) if `seq` is a contiguous +1 ascending run, else None.

    Region / idx-range selectors produce contiguous local offsets and output positions
    within each block, which lets the scatter use slice views instead of np.ix_ gathers.
    """
    if not seq:
        return None
    arr = np.asarray(seq)
    start = int(arr[0])
    if int(arr[-1]) - start + 1 == len(arr) and bool(np.all(np.diff(arr) == 1)):
        return slice(start, start + len(arr))
    return None


def _assemble(
    reader: HailBlockMatrixReader, global_idxs: List[int], max_workers: int = 4
) -> np.ndarray:
    """Build a symmetric submatrix for the given global indices (NaN where not stored).

    Assumes the BlockMatrix is stored upper-triangular (LD convention): off-diagonal
    blocks fully populated, diagonal blocks upper-triangular (symmetrized on read).
    Caller must pass deduplicated global_idxs. Stored blocks are fetched concurrently
    (up to max_workers) and scattered in the main thread, so peak memory is bounded by
    ~max_workers blocks and there are no concurrent writes to the output.
    """
    bs = reader.block_size
    n = len(global_idxs)
    out = np.full((n, n), np.nan, dtype=np.float64)

    groups = defaultdict(list)  # block_id -> list of (out_pos, local_offset)
    for pos, g in enumerate(global_idxs):
        groups[g // bs].append((pos, g % bs))
    block_ids = sorted(groups)

    # Upper-triangular block pairs that are actually stored on disk.
    pairs = [
        (p, q)
        for a_i, p in enumerate(block_ids)
        for q in block_ids[a_i:]
        if reader.partitioner.part_slot(p, q) is not None
    ]

    def _scatter(p, q, block):
        a_positions = [pos for pos, _ in groups[p]]
        a_locals = [loc for _, loc in groups[p]]
        b_positions = [pos for pos, _ in groups[q]]
        b_locals = [loc for _, loc in groups[q]]
        a_ps, a_ls = _as_slice(a_positions), _as_slice(a_locals)
        b_ps, b_ls = _as_slice(b_positions), _as_slice(b_locals)
        if a_ps and a_ls and b_ps and b_ls:
            # Fast path: contiguous selection (region / idx-range). Use slice VIEWS and
            # write directly into `out`, avoiding the np.ix_ gather and intermediate copies.
            src = block[a_ls, b_ls]
            if p == q:
                # Diagonal block is stored upper-triangular; symmetrize in place. src+src.T
                # doubles the diagonal, so reset it (np.fill_diagonal avoids building a
                # 134 MB np.diag(np.diag(...)) matrix).
                dst = out[a_ps, b_ps]
                np.add(src, src.T, out=dst)
                np.fill_diagonal(dst, np.diagonal(src))
            else:
                out[a_ps, b_ps] = src
                out[b_ps, a_ps] = src.T
            return
        # General path: arbitrary order (e.g. z-file). Fancy indexing copies the sub-block.
        sub = block[np.ix_(a_locals, b_locals)]
        if p == q:
            res = sub + sub.T
            np.fill_diagonal(res, np.diagonal(sub))
            out[np.ix_(a_positions, b_positions)] = res
        else:
            out[np.ix_(a_positions, b_positions)] = sub
            out[np.ix_(b_positions, a_positions)] = sub.T

    if len(pairs) <= 1 or max_workers <= 1:
        # Single block (or serial requested): no pool overhead.
        for p, q in pairs:
            block = reader.read_block(p, q)
            if block is not None:
                _scatter(p, q, block)
        return out

    workers = min(len(pairs), max_workers)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(reader.read_block, p, q): (p, q) for (p, q) in pairs}
        for fut in as_completed(futures):
            p, q = futures[fut]
            block = fut.result()  # re-raises a worker's fetch error in the main thread
            if block is not None:
                _scatter(p, q, block)
    return out


def _apply_flips(matrix: np.ndarray, flip_positions: List[int]) -> np.ndarray:
    """Negate rows/cols for swapped-allele variants (double-flip cancels, as intended)."""
    if not flip_positions:
        return matrix
    f = np.asarray(flip_positions, dtype=int)
    matrix[f, :] *= -1.0
    matrix[:, f] *= -1.0
    return matrix


def _read_zfile(path: str) -> pd.DataFrame:
    """Read a FINEMAP/SuSiE-style whitespace z-file with a header row.

    Required columns (case-insensitive): rsid, chromosome, position, allele1, allele2.
    allele1 is the reference allele, allele2 is the effect/alternative allele.
    """
    df = pd.read_csv(path, sep=r"\s+")
    df.columns = [c.lower() for c in df.columns]
    required = {"rsid", "chromosome", "position", "allele1", "allele2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"z-file {path} missing columns: {sorted(missing)}")
    return df


def _resolve_targets(vi, region, z, idx_range, on_missing):
    """Return a DataFrame with columns: idx, chrom, pos, ref, alt, rsid, flipped (in matrix order)."""
    if region is not None:
        chrom, (start, end) = parse_region(region)
        df = vi.query_region(chrom, start, end)
        df = df.rename(columns={"contig": "chrom", "position": "pos"})
        df["rsid"] = (
            df["chrom"].astype(str)
            + ":"
            + df["pos"].astype(str)
            + ":"
            + df["ref"].astype(str)
            + ":"
            + df["alt"].astype(str)
        )
        df["flipped"] = False
        return df[["idx", "chrom", "pos", "ref", "alt", "rsid", "flipped"]].reset_index(drop=True)

    if idx_range is not None:
        start, end = idx_range
        if vi is None:
            # No variant index: synthesize placeholder metadata so outputs still write.
            idxs = list(range(int(start), int(end)))
            return pd.DataFrame(
                {
                    "idx": idxs,
                    "chrom": [""] * len(idxs),
                    "pos": [0] * len(idxs),
                    "ref": [""] * len(idxs),
                    "alt": [""] * len(idxs),
                    "rsid": ["idx_%d" % i for i in idxs],
                    "flipped": [False] * len(idxs),
                }
            )
        df = vi.by_idx_range(start, end)
        df = df.rename(columns={"contig": "chrom", "position": "pos"})
        df["rsid"] = "idx_" + df["idx"].astype(str)
        df["flipped"] = False
        return df[["idx", "chrom", "pos", "ref", "alt", "rsid", "flipped"]].reset_index(drop=True)

    if z is not None:
        zdf = _read_zfile(z)
        rows = []
        unmatched = []
        for _, zr in zdf.iterrows():
            # z allele1 = ref, allele2 = effect/alt
            idx, flip = vi.match(
                str(zr["chromosome"]),
                int(zr["position"]),
                ref=str(zr["allele1"]),
                alt=str(zr["allele2"]),
            )
            if idx is None:
                unmatched.append(str(zr["rsid"]))
                continue
            rows.append(
                {
                    "idx": idx,
                    "chrom": str(zr["chromosome"]),
                    "pos": int(zr["position"]),
                    "ref": str(zr["allele1"]),
                    "alt": str(zr["allele2"]),
                    "rsid": str(zr["rsid"]),
                    "flipped": bool(flip),
                }
            )
        if unmatched:
            msg = f"{len(unmatched)} of {len(zdf)} z-file variants not found in the matrix"
            if on_missing == "error":
                raise ValueError(msg + f": {unmatched[:10]}")
            if on_missing == "warn":
                logger.warning(msg)
            # on_missing == "drop": silently drop unmatched
        return pd.DataFrame(
            rows, columns=["idx", "chrom", "pos", "ref", "alt", "rsid", "flipped"]
        ).reset_index(drop=True)

    raise ValueError("Exactly one of region, z, or idx_range must be provided")


def extract_ld(
    bm_path,
    variant_index_path,
    out,
    region=None,
    z=None,
    idx_range=None,
    output_format="bcor",
    compression=1,
    write_index=True,
    on_missing="warn",
    fill="nan",
    block_cache=4,
    max_workers=4,
    storage_options=None,
    n_samples=0,
):
    """Extract a submatrix from a Hail BlockMatrix LD store and write outputs.

    Exactly one of region / z / idx_range must be given. Returns (matrix, variant_df).
    """
    selectors = [s is not None for s in (region, z, idx_range)]
    if sum(selectors) != 1:
        raise ValueError("Exactly one of region, z, or idx_range must be provided")

    vi = (
        None
        if variant_index_path is None
        else VariantIndex(variant_index_path, storage_options=storage_options)
    )
    if vi is None and idx_range is None:
        raise ValueError("variant_index is required for region/z selectors")
    # idx_range works without a variant index (placeholder metadata is synthesized).
    targets = _resolve_targets(vi, region, z, idx_range, on_missing)
    if len(targets) == 0:
        logger.warning("No variants matched the selector; output will be empty")

    reader = HailBlockMatrixReader(
        bm_path, storage_options=storage_options, block_cache=block_cache
    )
    global_idxs = [int(x) for x in targets["idx"].tolist()]
    seen, dups = set(), []
    for gi in global_idxs:
        if gi in seen:
            dups.append(gi)
        seen.add(gi)
    if dups:
        raise ValueError(
            f"Selection maps to duplicate matrix indices {sorted(set(dups))[:10]}; "
            "each variant must map to a unique row (deduplicate the z-file)."
        )
    matrix = _assemble(reader, global_idxs, max_workers=max_workers)
    matrix = _apply_flips(matrix, list(np.where(targets["flipped"].to_numpy())[0]))

    if fill == "zero":
        matrix = np.nan_to_num(matrix, nan=0.0)
    n_missing = int(np.isnan(matrix).sum())
    if n_missing:
        logger.warning(
            "%d of %d requested cells are not stored in the matrix (off-band); " "filled with NaN",
            n_missing,
            matrix.size,
        )

    _write_outputs(matrix, targets, out, output_format, compression, write_index, n_samples)
    return matrix, targets


def _write_outputs(matrix, targets, out, output_format, compression, write_index, n_samples):
    """Write .bcor / .npz plus the .variants.tsv companion."""
    targets.to_csv(out + ".variants.tsv", sep="\t", index=False)
    if output_format in ("bcor", "both"):
        variant_info = targets[["rsid", "chrom", "pos", "ref", "alt"]]
        save_bcor(
            corr_matrix=matrix,
            output_file=out + ".bcor",
            variant_info=variant_info,
            n_samples=n_samples,
            compression=compression,
            write_index=write_index,
        )
    if output_format in ("npz", "both"):
        np.savez(out + ".npz", matrix=matrix, idx=targets["idx"].to_numpy())
