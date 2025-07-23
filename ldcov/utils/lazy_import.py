"""
Lazy import utilities for reducing startup time.
"""

import importlib
import sys
from typing import Any, Optional, Dict


class LazyModule:
    """Lazy module importer that defers imports until accessed."""
    
    def __init__(self, module_name: str):
        self._module_name = module_name
        self._module: Optional[Any] = None
    
    def __getattr__(self, name: str) -> Any:
        if self._module is None:
            self._module = importlib.import_module(self._module_name)
        return getattr(self._module, name)
    
    def __repr__(self) -> str:
        if self._module is None:
            return f"<LazyModule '{self._module_name}' (not loaded)>"
        return repr(self._module)


class LazyLoader:
    """Context manager for temporarily enabling lazy imports."""
    
    def __init__(self):
        self._lazy_modules: Dict[str, LazyModule] = {}
        self._original_import = None
    
    def __enter__(self):
        """Enable lazy loading."""
        self._original_import = __builtins__.__import__
        __builtins__.__import__ = self._lazy_import
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore normal import behavior."""
        __builtins__.__import__ = self._original_import
    
    def _lazy_import(self, name, *args, **kwargs):
        """Custom import function that returns lazy modules for specified packages."""
        # List of heavy modules to load lazily
        lazy_modules = {'numpy', 'pandas', 'scipy', 'gcsfs', 'tqdm'}
        
        if name in lazy_modules and name not in sys.modules:
            if name not in self._lazy_modules:
                self._lazy_modules[name] = LazyModule(name)
            return self._lazy_modules[name]
        
        # Use normal import for everything else
        return self._original_import(name, *args, **kwargs)


def lazy_import(module_name: str) -> LazyModule:
    """Create a lazy module importer for the specified module."""
    return LazyModule(module_name)