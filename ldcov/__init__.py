"""
ldcov: A Python package for efficient linkage disequilibrium calculation with covariate adjustment.

This package provides tools for:
1. LD calculation for large BGEN genotype datasets
2. Covariate adjustment of genotypes using FWL projection
3. Efficient handling of BGEN files with support for regions and indexing
"""

__version__ = "0.1.0"

# Ultra-aggressive lazy import support - defer ALL imports
_LAZY_IMPORTS = {
    # Compute functions
    'load_and_adjust_genotypes': lambda: _import('.compute.correlation', 'load_and_adjust_genotypes'),
    'compute_ld_from_standardized': lambda: _import('.compute.correlation', 'compute_ld_from_standardized'),
    'compute_correlation_matrix': lambda: _import('.compute.correlation', 'compute_correlation_matrix'),
    'regress_out_covariates': lambda: _import('.compute.covariate', 'regress_out_covariates'),
    'standardize_genotypes': lambda: _import('.compute.covariate', 'standardize_genotypes'),
    'compute_projection': lambda: _import('.compute.projection', 'compute_projection'),
    'save_projection': lambda: _import('.compute.projection', 'save_projection'),
    'load_projection': lambda: _import('.compute.projection', 'load_projection'),
    
    # IO functions
    'load_bgen': lambda: _import('.io', 'load_bgen'),
    'load_covariates': lambda: _import('.io.covariate_loader', 'load_covariates'),
    'save_correlation_matrix': lambda: _import('.io.correlation_io', 'save_correlation_matrix'),
    'load_correlation_matrix': lambda: _import('.io.correlation_io', 'load_correlation_matrix'),
    'BgenReader': lambda: _import('.io.bgen', 'BgenReader'),
    'BGIReader': lambda: _import('.io.bgen', 'BGIReader'),
    'BcorReader': lambda: _import('.io', 'BcorReader'),
    'BcorWriter': lambda: _import('.io', 'BcorWriter'),
    'save_bcor': lambda: _import('.io', 'save_bcor'),
    
    # Utility functions
    'parse_region': lambda: _import('.utils.region_utils', 'parse_region'),
    'get_region_variants': lambda: _import('.utils.region_utils', 'get_region_variants'),
    'load_variant_filter': lambda: _import('.utils.z_utils', 'load_variant_filter'),
}

def _import(module_path, attr_name):
    """Helper to perform the actual import."""
    import importlib
    if module_path.startswith('.'):
        module = importlib.import_module(module_path, package=__name__)
    else:
        module = importlib.import_module(module_path)
    return getattr(module, attr_name)

def __getattr__(name):
    """Ultra-lazy loading of module attributes."""
    if name in _LAZY_IMPORTS:
        # Call the lambda to import and get the attribute
        attr = _LAZY_IMPORTS[name]()
        # Cache it for future use
        globals()[name] = attr
        return attr
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
