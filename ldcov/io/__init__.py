"""
Input/output operations for the ldcov package.

This package provides utilities for reading and writing BGEN genotype data,
covariate data, and correlation matrices.
"""

from .bgen import load_bgen
from .covariate_loader import load_covariates
from .correlation_io import save_correlation_matrix, load_correlation_matrix
from .bcor_reader import BcorReader
from .bcor_writer import BcorWriter, save_bcor
