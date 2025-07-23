"""
Input/output operations for the ldcov package.

This package provides utilities for reading and writing BGEN genotype data,
covariate data, and correlation matrices.
"""

# Ultra-lazy imports - defer all module loading
_LAZY_IMPORTS = {
    'load_bgen': lambda: _import('.bgen', 'load_bgen'),
    'load_covariates': lambda: _import('.covariate_loader', 'load_covariates'),
    'save_correlation_matrix': lambda: _import('.correlation_io', 'save_correlation_matrix'),
    'load_correlation_matrix': lambda: _import('.correlation_io', 'load_correlation_matrix'),
    'BcorReader': lambda: _import('.bcor_reader', 'BcorReader'),
    'BcorWriter': lambda: _import('.bcor_writer', 'BcorWriter'),
    'save_bcor': lambda: _import('.bcor_writer', 'save_bcor'),
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
