"""
Computation modules for LD calculation and covariate adjustment.
"""

# Ultra-lazy imports - defer all module loading
_LAZY_IMPORTS = {
    'compute_correlation_matrix': lambda: _import('.correlation', 'compute_correlation_matrix'),
    'load_and_adjust_genotypes': lambda: _import('.correlation', 'load_and_adjust_genotypes'),
    'compute_ld_from_standardized': lambda: _import('.correlation', 'compute_ld_from_standardized'),
    'regress_out_covariates': lambda: _import('.covariate', 'regress_out_covariates'),
    'standardize_genotypes': lambda: _import('.covariate', 'standardize_genotypes'),
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
