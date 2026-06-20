#!/usr/bin/env python
"""Offline builder: Hail variant_indices.ht -> sorted Parquet variant index for ldcov --ld-bm.

Run ONCE per matrix on a machine with Hail installed. This is the only Hail-dependent
code in ldcov; the runtime --ld-bm path never imports Hail.

Usage:
    python scripts/make_bm_variant_index.py \
        --ht gs://.../...ld.variant_indices.ht \
        --out gnomad_nfe.variant_index.parquet
"""

import argparse


# Columns the --ld-bm runtime actually needs (variant<->idx matching + ref/alt sign-flip).
ESSENTIAL_COLUMNS = ["contig", "position", "ref", "alt", "idx"]


def write_variant_index_from_table(
    ht, out_path: str, minimal: bool = False, compression: str = "zstd"
) -> None:  # pragma: no cover (needs Hail)
    """Transform an already-loaded Hail Table into a sorted Parquet variant index.

    Split out from build_variant_index so callers that need a custom hl.init() (tmp dir,
    Spark/driver memory, requester-pays) can configure Hail themselves, then reuse
    this single copy of the transform/guard/export logic.
    """
    import os

    import hail as hl
    import pandas as pd

    if "idx" not in ht.row:
        raise ValueError("variant_indices.ht has no 'idx' field; cannot build variant index")
    # Fail loudly on any non-biallelic row before exporting.
    bad = ht.filter(hl.len(ht.alleles) != 2)
    n_bad = bad.count()
    if n_bad:
        sample = bad.head(5).collect()
        raise ValueError(
            f"{n_bad} multiallelic/monomorphic variants found; refusing to build variant index. "
            f"Examples: {[str(r.locus) for r in sample]}"
        )
    ht = ht.annotate(
        contig=ht.locus.contig,
        position=ht.locus.position,
        ref=ht.alleles[0],
        alt=ht.alleles[1],
    )
    if minimal:
        # Only the runtime-essential columns -> small, shippable variant index.
        flat = ht.key_by().select(*ESSENTIAL_COLUMNS)
    else:
        # Keep idx + flattened non-key fields (e.g. pop_freq.*, rsid, AF) for QC.
        flat = ht.flatten()
    # to_pandas() collects through the driver and is unstable/slow for ~10-24M rows.
    # Export to a block-gzipped TSV (distributed, stable) and read it back with pandas.
    tmp_tsv = out_path + ".export.tsv.bgz"
    flat.export(tmp_tsv)
    df = pd.read_csv(
        tmp_tsv,
        sep="\t",
        compression="gzip",
        dtype={"contig": str, "ref": str, "alt": str},
    )
    os.remove(tmp_tsv)
    # Order columns: essential keys first, then everything else (full mode).
    cols = ESSENTIAL_COLUMNS + [c for c in df.columns if c not in ESSENTIAL_COLUMNS]
    df = df[[c for c in cols if c in df.columns]]
    df = df.sort_values(["contig", "position"]).reset_index(drop=True)
    df.to_parquet(out_path, index=False, compression=compression)
    print(f"Wrote {len(df)} variants ({len(df.columns)} cols, {compression}) to {out_path}")


def build_variant_index(
    ht_path: str, out_path: str, minimal: bool = False, compression: str = "zstd"
) -> None:  # pragma: no cover (needs Hail)
    import hail as hl

    hl.init()
    write_variant_index_from_table(
        hl.read_table(ht_path), out_path, minimal=minimal, compression=compression
    )


def main():  # pragma: no cover
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ht",
        required=True,
        help="Path to the variant index .ht (local, gs://, or s3://; "
        "s3:// needs Hail/Spark configured for S3 access)",
    )
    ap.add_argument("--out", required=True, help="Output Parquet variant index path")
    ap.add_argument(
        "--minimal",
        action="store_true",
        help="Write only the runtime-essential columns "
        f"({', '.join(ESSENTIAL_COLUMNS)}) for a small, shippable variant index",
    )
    ap.add_argument(
        "--compression", default="zstd", help="Parquet compression codec (default: zstd)"
    )
    args = ap.parse_args()
    build_variant_index(args.ht, args.out, minimal=args.minimal, compression=args.compression)


if __name__ == "__main__":
    main()
