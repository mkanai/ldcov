"""CLI glue for the `--ld-bm` mode."""

import json
import logging

from .extract import extract_ld

logger = logging.getLogger(__name__)


def add_ld_bm_arguments(parser):
    """Register the NEW --ld-bm flags on the existing ldcov argparse parser.

    Reuses existing --region, --z, --out, --no-bcor-idx (bcor_index), and --output-format.
    """
    group = parser.add_argument_group("ld-bm (Hail BlockMatrix LD extraction)")
    group.add_argument(
        "--ld-bm", action="store_true", help="Extract a submatrix from a Hail BlockMatrix LD store"
    )
    group.add_argument(
        "--bm", help="Path to the BlockMatrix directory (local or gs://; s3:// requires s3fs)"
    )
    group.add_argument("--variant-index", help="Path to the Parquet variant index")
    group.add_argument("--idx-range", help="BlockMatrix index range selector, format A:B")
    group.add_argument(
        "--on-missing",
        choices=["warn", "error", "drop"],
        default="warn",
        help="Behavior for unmatched z-file variants (default: warn)",
    )
    group.add_argument(
        "--fill",
        choices=["nan", "zero"],
        default="nan",
        help="Fill value for off-band pairs in --ld-bm (default: nan)",
    )
    group.add_argument(
        "--block-cache",
        type=int,
        default=4,
        help="Number of decoded blocks to cache in --ld-bm (default: 4)",
    )
    group.add_argument(
        "--fetch-workers",
        type=int,
        default=4,
        help="Concurrent block fetches for --ld-bm (default: 4)",
    )
    group.add_argument(
        "--compression",
        type=int,
        default=1,
        help="bcor compression level for --ld-bm output (default: 1)",
    )
    group.add_argument(
        "--storage-options",
        help="JSON dict of fsspec storage options for --bm/--variant-index "
        "(e.g. '{\"anon\": true}' for credentials/endpoint). "
        "s3:// defaults to anonymous when omitted.",
    )


def run_ld_bm(args):
    """Dispatch target for `ldcov --ld-bm`."""
    if not args.bm:
        raise ValueError("--ld-bm requires --bm")
    if not args.out:
        raise ValueError("--ld-bm requires --out")

    idx_range = None
    if getattr(args, "idx_range", None):
        a, b = args.idx_range.split(":")
        idx_range = (int(a), int(b))

    storage_options = None
    if getattr(args, "storage_options", None):
        try:
            storage_options = json.loads(args.storage_options)
        except json.JSONDecodeError as exc:
            raise ValueError(f"--storage-options is not valid JSON: {exc}") from exc
        if not isinstance(storage_options, dict):
            raise ValueError("--storage-options must be a JSON object (dict)")

    # --output-format is shared with the BGEN flow (matrix/long/bcor); for --ld-bm only
    # bcor/npz/both are meaningful. Anything else (e.g. the global default "matrix") -> bcor.
    fmt = args.output_format
    if fmt not in ("bcor", "npz", "both"):
        fmt = "bcor"

    extract_ld(
        bm_path=args.bm,
        variant_index_path=args.variant_index,
        out=args.out,
        region=args.region,
        z=args.z,
        idx_range=idx_range,
        output_format=fmt,
        compression=args.compression,
        write_index=args.bcor_index,
        on_missing=args.on_missing,
        fill=args.fill,
        block_cache=args.block_cache,
        max_workers=args.fetch_workers,
        storage_options=storage_options,
    )
    logger.info("ld-bm extraction complete: %s", args.out)
