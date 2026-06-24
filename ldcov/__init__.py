"""
ldcov: A Python package for efficient linkage disequilibrium calculation with covariate adjustment.

This package provides tools for:
1. LD calculation for large BGEN genotype datasets
2. Covariate adjustment of genotypes using FWL projection
3. Efficient handling of BGEN files with support for regions and indexing
"""

from importlib.metadata import version as _version, PackageNotFoundError as _PackageNotFoundError

try:
    __version__ = _version("ldcov")
except _PackageNotFoundError:  # package not installed (e.g. running from a source tree)
    __version__ = "0.0.0+unknown"

# Lazy import support for faster startup
_LAZY_IMPORTS = {
    "load_and_adjust_genotypes": ".compute.correlation",
    "compute_ld_from_standardized": ".compute.correlation",
    "compute_correlation_matrix": ".compute.correlation",
    "regress_out_covariates": ".compute.covariate",
    "standardize_genotypes": ".compute.covariate",
    "load_bgen": ".io",
    "load_covariates": ".io.covariate_loader",
    "save_correlation_matrix": ".io.correlation_io",
    "load_correlation_matrix": ".io.correlation_io",
}


def __getattr__(name):
    """Lazy loading of module attributes."""
    if name in _LAZY_IMPORTS:
        module_path = _LAZY_IMPORTS[name]
        if module_path.startswith("."):
            # Relative import
            import importlib

            module = importlib.import_module(module_path, package=__name__)
        else:
            # Absolute import
            import importlib

            module = importlib.import_module(module_path)
        attr = getattr(module, name)
        # Cache it for future use
        globals()[name] = attr
        return attr
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
