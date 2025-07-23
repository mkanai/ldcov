"""
Command-line interface for the ldcov package.

This module provides a command-line interface for computing
linkage disequilibrium and adjusting genotypes with ldcov.
"""

import argparse
import logging
import sys
import os
from typing import Optional, List  # noqa: F401

from ..compute.correlation import (
    load_and_adjust_genotypes,
    compute_ld_from_standardized,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """
    Parse command-line arguments.

    Returns:
    --------
    argparse.Namespace
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="ldcov: Compute linkage disequilibrium with optional covariate adjustment.",
        epilog="""
Examples:
  # Compute LD only (no covariate adjustment)
  ldcov --bgen input.bgen --out output --compute-ld

  # Compute LD with covariate adjustment
  ldcov --bgen input.bgen --out output --compute-ld -c covariates.txt

  # Use specific columns as covariates
  ldcov --bgen input.bgen --out output --compute-ld -c covariates.txt --covariate-cols PC1 PC2 PC3
  
  # Pre-compute projection matrix for later use
  ldcov --precompute-projection -c covariates.txt --sample data.sample --out myproject
  
  # Use pre-computed projection matrix
  ldcov --bgen input.bgen --projection-matrix myproject.proj.npz --compute-ld --out output
  
  # Compute LD and save projection matrix for future use
  ldcov --bgen input.bgen -c covariates.txt --compute-ld --save-projection --out output
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments (bgen is conditionally required)
    parser.add_argument("--bgen", help="Input BGEN genotype file")

    parser.add_argument(
        "--out",
        required=True,
        help="Output file prefix (extensions will be added automatically)",
    )

    # Mode flags (at least one required)
    parser.add_argument(
        "--compute-ld", action="store_true", help="Compute linkage disequilibrium matrix"
    )

    parser.add_argument(
        "--precompute-projection",
        action="store_true",
        help="Pre-compute projection matrix from covariates for later use",
    )

    # Optional arguments
    parser.add_argument("--bgi", help="Path to BGEN index file (.bgi, optional)")

    parser.add_argument("--covariates", "-c", help="Covariate file for adjustment")

    parser.add_argument(
        "--projection-matrix",
        help="Pre-computed projection matrix file (.proj.npz) to use instead of covariates",
    )

    parser.add_argument(
        "--save-projection",
        action="store_true",
        help="Save projection matrix when computing LD or adjusting genotypes",
    )

    parser.add_argument(
        "--covariate-id-col",
        default="IID",
        help="Column name for sample IDs in covariate file (default: IID)",
    )

    parser.add_argument(
        "--covariate-cols",
        nargs="+",
        help="Specific columns to use as covariates (default: all columns except ID)",
    )

    parser.add_argument("--region", "-r", help='Genomic region in format "chr:start-end"')

    parser.add_argument(
        "--output-format",
        choices=["matrix", "long", "bcor"],
        default="matrix",
        help="Output format for LD matrix (default: matrix)",
    )

    parser.add_argument("--sample", help="Path to sample file (.sample, optional)")

    parser.add_argument(
        "--z",
        "--z-file",
        help="Path to .z file specifying variants to extract and their order (optional)",
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    parser.add_argument(
        "--progress",
        action="store_true",
        help="Enable progress bars during loading (disabled by default)",
    )

    parser.add_argument(
        "--nan-action",
        choices=["error", "mean", "omit"],
        default="error",
        help="Action for handling NaN values in genotypes: "
        "'error' (default) - raise error if NaN values are found; "
        "'mean' - impute NaN values with variant mean; "
        "'omit' - remove samples with NaN values (with warnings)",
    )

    return parser.parse_args()


def validate_args(args):
    """
    Validate command-line arguments based on flags.

    Parameters:
    -----------
    args : argparse.Namespace
        Parsed arguments

    Raises:
    -------
    ValueError
        If arguments are invalid for the selected options
    """
    # At least one mode flag must be specified
    if not args.compute_ld and not args.precompute_projection:
        raise ValueError(
            "At least one of --compute-ld or --precompute-projection must be specified"
        )

    # Precompute projection requires covariates
    if args.precompute_projection:
        if not args.covariates:
            raise ValueError("--covariates is required when using --precompute-projection")
        if args.bgen:
            raise ValueError("--bgen should not be specified with --precompute-projection")

    # Cannot use both covariates and projection matrix
    if args.covariates and args.projection_matrix:
        raise ValueError("Cannot specify both --covariates and --projection-matrix")

    # Save projection only makes sense with covariates
    if args.save_projection and not args.covariates:
        raise ValueError("--save-projection requires --covariates")

    # BGEN is required for compute-ld
    if args.compute_ld and not args.bgen:
        raise ValueError("--bgen is required for --compute-ld")


def run_cli():
    """
    Main CLI entry point.
    """
    # Parse arguments
    args = parse_args()

    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate arguments
    try:
        validate_args(args)
    except ValueError as e:
        logger.error(f"Invalid arguments: {e}")
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(os.path.abspath(args.out))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Handle precompute projection mode
    if args.precompute_projection:
        from ..compute.projection import compute_projection_matrix, save_projection_matrix

        logger.info("Pre-computing projection matrix from covariates")
        projection_data = compute_projection_matrix(
            covariate_file=args.covariates,
            sample_file=args.sample,
            covariate_id_col=args.covariate_id_col,
            covariate_cols=args.covariate_cols,
        )

        output_file = f"{args.out}.proj.npz"
        save_projection_matrix(projection_data, output_file)
        logger.info(f"Projection matrix saved to {output_file}")
        return

    # For other modes, we need BGEN file
    # Auto-detect BGI file if not specified
    bgi_file = args.bgi
    if bgi_file is None and args.bgen:
        potential_bgi = f"{args.bgen}.bgi"
        if os.path.exists(potential_bgi):
            logger.info(f"Using auto-detected BGI file: {potential_bgi}")
            bgi_file = potential_bgi

    # Load and adjust genotypes (common step for compute-ld and export-adjusted-bgen)
    result = load_and_adjust_genotypes(
        genotype_file=args.bgen,
        covariate_file=args.covariates,
        projection_matrix_file=args.projection_matrix,
        region=args.region,
        index_file=bgi_file,
        sample_file=args.sample,
        z_file=args.z,
        covariate_id_col=args.covariate_id_col,
        covariate_cols=args.covariate_cols,
        show_progress=args.progress,
        nan_action=args.nan_action,
    )
    standardized_genotypes, variant_info, sample_ids, means, norms = result

    # Save projection matrix if requested and covariates were used
    if args.save_projection and args.covariates:
        from ..compute.projection import compute_projection_matrix, save_projection_matrix

        logger.info("Computing and saving projection matrix")
        projection_data = compute_projection_matrix(
            covariate_file=args.covariates,
            sample_ids=sample_ids,
            covariate_id_col=args.covariate_id_col,
            covariate_cols=args.covariate_cols,
        )

        projection_output_file = f"{args.out}.proj.npz"
        save_projection_matrix(projection_data, projection_output_file)
        logger.info(f"Projection matrix saved to {projection_output_file}")

    # Compute LD if requested
    if args.compute_ld:
        # Determine LD output file extension based on format
        if args.output_format == "matrix":
            ld_output_file = f"{args.out}.ld"
        elif args.output_format == "long":
            ld_output_file = f"{args.out}.ld.gz"
        elif args.output_format == "bcor":
            ld_output_file = f"{args.out}.bcor"

        logger.info("Computing LD matrix")
        compute_ld_from_standardized(
            standardized_genotypes=standardized_genotypes,
            variant_info=variant_info,
            output_file=ld_output_file,
            output_format=args.output_format,
        )
        logger.info(f"LD matrix saved to {ld_output_file}")


if __name__ == "__main__":
    run_cli()
