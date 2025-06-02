"""
Input/output operations for the ldcov package.

This package provides utilities for reading and writing BGEN genotype data,
covariate data, and correlation matrices.
"""

from .bgen_reader import load_bgen, BgenFileReader
from .covariate_loader import load_covariates
from .correlation_io import save_correlation_matrix, load_correlation_matrix
from .bgen_writer import correlation_preserving_transform, write_bgen, save_metadata
from .bcor_reader import BcorReader
from .bcor_writer import BcorWriter, save_bcor
