#!/usr/bin/env python3
"""Check if ldcov is using vendored libraries."""

import os
import sys

sys.path.insert(0, '/app')

try:
    from ldcov.io.bgen import BgenReader
    print('✅ BgenReader imported successfully')
    
    # Check for vendored library indicators
    sys.path.insert(0, '/app/ldcov/io/bgen')
    try:
        from _build_config import COMPRESSION_BACKEND, get_build_info
        info = get_build_info()
        print(f'Build configuration: {COMPRESSION_BACKEND}')
        print(f'Build info: {info}')
        
        if COMPRESSION_BACKEND == 'vendored':
            print('✅ Using vendored libraries!')
            sys.exit(0)
        else:
            print('⚠️  Using system libraries')
            print('This may result in ~20% performance loss')
            sys.exit(1)
    except ImportError as e:
        print(f'⚠️  Build configuration module not found: {e}')
        # Try alternative location
        try:
            import ldcov.io.bgen._build_config as config
            print(f'Found config at alternative location: {config.COMPRESSION_BACKEND}')
        except:
            pass
        sys.exit(1)
except Exception as e:
    print(f'❌ Error: {e}')
    sys.exit(1)