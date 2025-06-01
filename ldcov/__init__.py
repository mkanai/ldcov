"""
ldcov: A Python package for efficient linkage disequilibrium calculation with covariate adjustment.

This package provides tools for:
1. LD calculation for large BGEN genotype datasets
2. Covariate adjustment of genotypes using FWL projection
3. Efficient handling of BGEN files with support for regions and indexing
"""

__version__ = "0.1.0"

# Import core functionality to make it accessible at the top level
from .compute.correlation import (
    load_and_adjust_genotypes,
    save_adjusted_genotypes,
    compute_ld_from_standardized,
    compute_correlation_matrix,
)
from .compute.covariate import regress_out_covariates, standardize_genotypes
from .io.bgen_reader import load_bgen
from .io.covariate_loader import load_covariates
from .io.correlation_io import save_correlation_matrix, load_correlation_matrix
from .io.bgen_writer import write_bgen, save_metadata
