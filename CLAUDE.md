# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ldcov (formerly pyldbm) is a Python package for efficient linkage disequilibrium (LD) calculation with covariate adjustment. It focuses exclusively on BGEN format genetic data and implements Frisch-Waugh-Lovell (FWL) projection to remove confounding effects from covariates.

**Package was renamed from `pyldbm` to `ldcov` in January 2025.**

**NEW (January 2025)**: 
- Added pre-computed projection matrix support for efficient large-scale analyses. QR decomposition can now be computed once and reused across multiple genomic regions or distributed computing jobs.
- Implemented sample filtering during BGEN loading for memory efficiency. When working with large BGEN files (e.g., 500K samples) and smaller covariate sets (e.g., 10K samples), only the needed samples are loaded into memory.
- Added progress bars for variant loading operations using `tqdm` library. Use `--no-progress` flag to disable for scripting.
- Added `--nan-action` option for flexible NaN handling in genotype data with three strategies: 'error' (default), 'mean' (impute with variant mean), or 'omit' (remove samples with NaN).

## Development Commands

```bash
# Install for development with all dependencies
make install

# Run tests
make test
pytest                    # Run all tests
pytest tests/test_correlation.py::test_compute_ld  # Run specific test
pytest tests/test_projection.py  # Run projection matrix tests

# Code quality
make lint                 # Check code style (flake8 + black check)
make format              # Auto-format code with black
make coverage            # Run tests with coverage report

# Build
make dist                # Build distribution packages
make clean              # Clean build artifacts

# Example workflows
# Pre-compute projection matrix
ldcov --precompute-projection -c covariates.txt --sample data.sample --out myproject

# Use pre-computed projection for LD computation  
ldcov --bgen data.bgen --projection-matrix myproject.proj.npz --compute-ld --out results

# Compute LD and save projection for future use
ldcov --bgen data.bgen -c covariates.txt --compute-ld --save-projection --out results

# Handle NaN values with mean imputation
ldcov --bgen data.bgen --compute-ld --nan-action mean --out results

# Disable progress bars for scripting
ldcov --bgen data.bgen --compute-ld --no-progress --out results
```

## Architecture (Simplified in 2025)

The codebase follows a modular structure:

- **cli/**: Command-line interface with flexible flag-based operation modes
  - `main.py`: Entry point
  - `commands.py`: Simplified CLI logic using `--compute-ld`, `--export-adjusted-bgen`, and `--precompute-projection` flags
- **compute/**: Core computation logic
  - `correlation.py`: Modular functions - `load_and_adjust_genotypes()`, `save_adjusted_genotypes()`, `compute_ld_from_standardized()`
  - `covariate.py`: FWL projection for covariate adjustment, standardization with L2 normalization, supports pre-computed projection matrices
  - `projection.py`: Pre-computation and I/O for projection matrices, enabling efficient reuse across analyses
- **io/**: File format handlers (BGEN only)
  - `bgen_reader.py`: BGEN file reading with region and variant filtering, supports sample filtering during loading
  - `bgen_writer.py`: BGEN writing with correlation-preserving transformation
  - `bcor_reader.py`: BCOR format reader with support for standard and extended formats
  - `bcor_writer.py`: BCOR format writer with automatic format selection based on diagonal values
  - `covariate_loader.py`: Covariate loading with automatic categorical encoding, supports gs:// paths
  - `correlation_io.py`: LD matrix output (matrix, long, bcor formats)
- **utils/**: Supporting utilities
  - `categorical_utils.py`: One-hot encoding for categorical variables
  - `region_utils.py`: Genomic region parsing
  - `variant_filter.py`: Z-file support for variant filtering and ordering

## Key Implementation Details

1. **Genotype Standardization**: Genotypes are centered and scaled by L2 norm (not standard deviation) before correlation computation.

2. **Covariate Adjustment**: Uses FWL projection with QR decomposition for numerical stability. Categorical covariates are automatically one-hot encoded.

3. **Pre-computed Projection Matrices** (NEW):
   - QR decomposition can be pre-computed once and reused across multiple analyses
   - Projection matrices saved in compressed NPZ format with sample ID validation
   - Enables efficient distributed computing workflows
   - Use `--precompute-projection` to generate, `--projection-matrix` to use

4. **Adjusted Genotypes**: When saving adjusted genotypes to BGEN, the code converts from standardized scale back to allelic scale (0-2) while preserving correlation structure.

5. **Memory Efficiency**: 
   - Supports block-based processing and in-place operations for large datasets
   - Sample filtering during BGEN loading: when covariates/projection specify a subset of samples, only those samples are loaded into memory (variant by variant filtering)
   - Reduces memory usage by up to 98% when working with biobank-scale data

6. **BCOR Format Support**:
   - Standard BCOR (magic: "bcor1.1"): Used when all diagonal values equal 1.0
   - Extended BCOR (magic: "bcor1.x"): Automatically used when diagonal values differ from 1.0 (e.g., adjusted LD matrices)
   - Both formats support compression levels 0-3 (2, 4, 8, or 1 byte per value)
   - Reader/writer modules can be imported separately for external use

## Testing Approach

Tests use pytest and mock data generation when file readers aren't available. The test suite is organized into focused modules:

### Test Organization (January 2025)

- **test_cli.py**: Command-line interface tests
  - Flag parsing and validation
  - Different operation modes
  - Error handling
  
- **test_compute.py**: Core computation tests
  - Correlation matrix computation
  - LD computation workflows
  - Integration with covariate adjustment
  
- **test_covariate.py**: Covariate handling tests
  - Genotype standardization
  - Covariate loading from various formats
  - Categorical variable encoding
  - FWL projection and adjustment
  
- **test_io.py**: I/O operation tests
  - BGEN reading and writing
  - Sample filtering during loading
  - BCOR format (standard and extended)
  - Correlation matrix I/O
  - Metadata handling
  
- **test_projection.py**: Projection matrix tests
  - Pre-computation and saving
  - Loading and validation
  - Sample compatibility checks
  
- **test_covariate_edge_cases.py**: Edge case tests
  - Numerical stability
  - Rank-deficient systems
  - Zero-variance handling
  
- **test_ldstore_comparison.py**: Compatibility tests
  - Comparison with LDstore2 reference
  - Format compatibility
  
- **test_utils.py**: Utility function tests
  - Region parsing
  - Variant filtering
  - Z-file handling

### Coverage Metrics

As of January 2025:
- Overall coverage: 81%
- Core modules: 80-85% coverage
- Critical paths: >95% coverage
- All 116 tests pass in ~14 seconds

## Code Style

- Black formatter with 100-character line length
- Type hints throughout
- Comprehensive logging for debugging
- In-place operations where possible for memory efficiency

## Recent Changes (January 2025)

### NaN Handling Enhancement
- **Added `--nan-action` option** with three strategies for handling NaN values in genotype data:
  - `error` (default): Raises detailed error showing which samples/variants have NaN
  - `mean`: Imputes NaN values with variant-wise mean (0 if all values are NaN)
  - `omit`: Removes samples containing any NaN values with warnings
- **Enhanced error reporting** for NaN values:
  - Reports number of samples and variants with NaN values
  - Shows first 5 sample/variant pairs with NaN values including IDs and positions
  - Helps users quickly identify problematic data in BGEN files
- **Added progress bars** for all variant loading operations using `tqdm`
  - Use `--no-progress` flag to disable for scripting/logging

### BGEN Reader Architecture Refactoring
- **Consolidated API**: Removed three public methods (`load_all_variants_and_dosages`, `load_region_variants_and_dosages`, `load_filtered_variants_and_dosages`) from BgenFileReader class
- **Simplified interface**: All BGEN loading now goes through the single `load_bgen()` function
- **Clean separation of concerns**:
  - Variant discovery phase: `_prepare_all_variants()`, `_prepare_region_variants()`, `_prepare_filtered_variants()`
  - Dosage loading phase: Single `_load_dosages()` method handles all loading scenarios
- **Benefits**:
  - Eliminates code duplication (~100 lines removed)
  - Clearer architecture with separated discovery and loading phases
  - Easier to extend with new selection criteria
  - Better testability of individual components
- **Alpha version**: Breaking changes acceptable as this is pre-release software

### BGI Index-Based Optimization (January 2025)
- **Mandatory BGI files**: BGEN files must now have accompanying `.bgi` index files
- **Custom BGI reader**: Implemented pure Python SQLite-based BGI reader (`ldcov/io/bgi_reader.py`)
- **Offset-based reading**: Extended the forked bgen library with `read_variants_at_offsets()` method
- **Direct variant access**: Uses file offsets from BGI to directly seek to specific variants
- **Performance benefits**:
  - Eliminates double-scanning of BGEN files
  - O(m) complexity for reading m variants instead of O(n) where n is total variants
  - Particularly efficient for region queries and filtered variant lists
  - 10-100x faster for small variant subsets from large files
- **Implementation details**:
  - BGI reader extracts variant metadata and file offsets from SQLite database
  - Extended C++ bgen reader with offset-based variant reading capability
  - Cython bindings expose the new functionality to Python
  - All variant loading now uses direct offset access

### Efficient Variant Filtering
- **Optimized variant filtering**: When using a `.z` file filter, only the requested variants are loaded
- **Two-pass approach**: First pass identifies matching variants, second pass loads only those variants
- **Memory efficient**: Significantly reduces memory usage and improves performance for large BGEN files
- **Correct ordering**: Maintains variant ordering according to `.z` file specification

## Recent Major Refactoring (January 2025)

### What was changed:
1. **Removed complexity**:
   - Eliminated `AnalysisConfig` and configuration objects
   - Removed `compute_ld()` convenience function
   - Removed support for VCF, PLINK formats (BGEN only now)
   - Removed unused dependencies (scipy, scikit-learn, ldstore)

2. **Simplified API**:
   - CLI uses simple flags: `--compute-ld`, `--export-adjusted-bgen`, `--precompute-projection`
   - Functions take direct parameters instead of config objects
   - Modular design: separate functions for loading, adjusting, saving

3. **Key functions**:
   ```python
   # Main workflow functions
   load_and_adjust_genotypes()  # Load BGEN and optionally adjust for covariates
   save_adjusted_genotypes()    # Save to BGEN with correlation preservation
   compute_ld_from_standardized()  # Compute LD from standardized genotypes
   
   # Projection matrix functions (NEW)
   from ldcov.compute.projection import compute_projection_matrix, save_projection_matrix, load_projection_matrix
   
   # BCOR I/O (can be imported separately)
   from ldcov.io import BcorReader, BcorWriter, save_bcor
   ```

4. **Added Projection Matrix Support** (NEW):
   - Pre-compute QR decomposition once, reuse across analyses
   - Workflow 1: `ldcov --precompute-projection -c covs.txt --out proj` then `ldcov --bgen data.bgen --projection-matrix proj.proj.npz --compute-ld --out results`
   - Workflow 2: `ldcov --bgen data.bgen -c covs.txt --compute-ld --save-projection --out results`
   - Enables efficient distributed computing for large-scale analyses

5. **Added Sample Filtering During Loading** (NEW):
   - BGEN files are filtered to only load samples present in covariate/projection files
   - Filtering happens variant-by-variant during loading, not after
   - Example: 500K sample BGEN + 10K sample covariates = only 10K samples loaded into memory
   - Reduces memory usage by up to 98% for biobank-scale analyses

6. **Added BCOR Support**:
   - Implemented pure Python BCOR reader/writer (no ldstore dependency)
   - Extended BCOR format for adjusted LD matrices with non-unit diagonal
   - Optimized with memory mapping, bulk I/O, and vectorized operations
   - Modules can be imported separately for external use

7. **Dependencies**:
   - Core: numpy, pandas, bgen, gcsfs, tqdm
   - Dev: pytest, pytest-cov, black, flake8
   - Note: gcsfs enables gs:// paths for covariate files only (not BGEN)

8. **Testing**:
   - All tests pass including new BCOR tests and projection matrix tests
   - Tests updated to use new modular functions
   - Added comprehensive BCOR format tests including extended format
   - pytest-asyncio disabled (not needed)

## Important Notes

- BGEN files must be local (no gs:// support)
- Covariate files can be from Google Cloud Storage (gs://)
- Z-files filter and order variants based on external lists
- Correlation-preserving transformation maintains LD structure when saving adjusted genotypes
- BCOR format automatically selects standard or extended format based on diagonal values
- Extended BCOR format preserves adjusted LD matrices with non-unit diagonal for use with tools like susie_rss
- Package directory has been renamed from `pyldbm/` to `ldcov/` to match the new package name
- Projection matrices (.proj.npz files) store Q matrix, sample IDs, and metadata for validation
- Pre-computed projection matrices must have all samples present in the genotype file
- Sample filtering: When BGEN has more samples than covariates/projection, only the intersection is loaded
- The bgen Python library does not support selective sample loading, but ldcov filters samples during the variant-by-variant loading process for memory efficiency

## Performance Optimization Plan (January 2025)

### Target Use Case: 1000 samples × 10,000 variants

Based on performance analysis, the main bottlenecks are IO operations and Python native for-loops, not NumPy operations. The optimization strategy focuses on pure Python/NumPy improvements without introducing new dependencies.

### Identified Bottlenecks

1. **BGEN Reading**:
   - List append operations without pre-allocation (10K operations)
   - Full data copy with `np.column_stack()` at the end
   - Column-wise access patterns

2. **BCOR Operations**:
   - 70,000+ individual read operations for metadata
   - Nested loops for correlation value conversion (50M iterations)
   - No bulk I/O utilization

3. **String I/O**:
   - Creating 100M string objects for matrix output
   - Individual string formatting in nested loops

4. **Memory Patterns**:
   - Unnecessary data copies and temporary arrays
   - Cache-unfriendly access patterns
   - No exploitation of memory locality

### Optimization Strategy

1. **Pre-allocation**:
   - Pre-allocate numpy arrays based on known dimensions
   - Eliminate list append operations in favor of direct array assignment
   - Example: `dosages = np.empty((n_samples, n_variants))` then fill by column

2. **Bulk I/O**:
   - Read BCOR metadata in one operation instead of 70K+ reads
   - Use numpy's fromfile/frombuffer for bulk binary operations
   - Buffer output operations to reduce system calls

3. **Vectorization**:
   - Replace string formatting loops with numpy's savetxt
   - Use vectorized operations for value conversions
   - Batch process correlation values instead of individual operations

4. **Memory Optimization**:
   - Use in-place operations consistently
   - Avoid unnecessary transposes and copies
   - Process data in cache-friendly blocks

5. **Specific Optimizations**:
   - `load_all_variants_and_dosages()`: Pre-allocate dosage array
   - `_read_meta()`: Implement bulk metadata reading
   - `save_correlation_matrix()`: Use numpy.savetxt for matrix output
   - `_convert_float_to_int()`: Vectorize correlation value conversion
   - Add chunked processing for very large matrices

### Expected Performance Gains

- BGEN loading: 2-3x speedup from pre-allocation and reduced copying
- BCOR reading: 5-10x speedup from bulk I/O operations
- String I/O: 10x+ speedup from vectorized formatting
- Overall: 2-5x speedup for typical 1000×10,000 use case

These optimizations maintain compatibility and require no new dependencies, focusing on better algorithmic choices and NumPy best practices.

### Implemented Optimizations (January 2025)

The following performance optimizations have been implemented:

1. **BGEN Reader Optimization**:
   - Pre-allocation of dosage arrays based on known variant count
   - Direct array assignment instead of list append operations
   - Fallback to original implementation if pre-counting fails
   - Result: 2-3x speedup for loading 10K variants

2. **BCOR I/O Optimization**:
   - Bulk metadata reading: reduced from 70K+ reads to single read operation
   - Vectorized float-to-int conversion for correlation values
   - Memory-mapped file access for large files
   - Result: 5-10x speedup for BCOR operations

3. **String I/O Optimization**:
   - Using numpy.savetxt for matrix format output
   - Buffered writing for long format (10K line chunks)
   - Pre-extracted variant info arrays for faster access
   - Result: 10x+ speedup for writing correlation matrices

4. **Correlation Computation**:
   - Uses direct matrix multiplication (numpy.dot) for simplicity and reliability
   - Leverages optimized BLAS libraries for best performance
   - Benchmarking showed blocked computation could be up to 79% faster for specific cases
     (few samples with many variants), but added complexity not justified for typical use cases
   - Result: Simple, maintainable code with excellent performance for typical genomics workflows

5. **Memory Efficiency**:
   - Consistent use of in-place operations
   - Reduced temporary array creation
   - Optimized data access patterns

### Performance Impact

For the target use case (1000 samples × 10,000 variants):
- BGEN loading: ~2-3x faster
- BCOR reading: ~5-10x faster
- Correlation matrix output: ~10x faster
- Overall workflow: ~2-5x faster depending on operations

All optimizations maintain backward compatibility and pass existing tests.

## Blocked Correlation Computation (Archived)

During performance optimization, we explored blocked computation for correlation matrices. While benchmarking showed significant speedups for specific cases (few samples with many variants), we decided to keep the production code simple with direct computation only. The blocked implementation and benchmark results are preserved here for future reference.

### Benchmark Results

Cases with >25% speedup from blocked computation:
- 50 samples × 15,000 variants: **79% faster**
- 10 samples × 10,000 variants: **74% faster**
- 100 samples × 15,000 variants: **59% faster**
- 25 samples × 10,000 variants: **57% faster**
- 50 samples × 10,000 variants: **50% faster**
- 200 samples × 10,000 variants: **45% faster**

The pattern: blocked computation excels with ≤200 samples and ≥5,000 variants.

### Blocked Implementation (Reference)

```python
def compute_correlation_matrix_blocked(standardized_genotypes: np.ndarray, block_size: Optional[int] = None) -> np.ndarray:
    """
    Compute correlation matrix using blocked algorithm for better cache efficiency.
    This implementation showed benefits for few samples with many variants.
    """
    n_samples, n_variants = standardized_genotypes.shape
    
    # Determine block size if not provided
    if block_size is None:
        # Heuristic: aim for blocks that fit in L3 cache (~8MB)
        # Each block needs memory for: block_size * n_samples * 8 bytes
        target_memory_mb = 8
        block_size = min(1000, int((target_memory_mb * 1024 * 1024) / (n_samples * 8)))
        block_size = max(100, block_size)  # Minimum block size
    
    # Initialize output matrix
    corr_matrix = np.zeros((n_variants, n_variants), dtype=np.float64)
    
    # Compute correlation in blocks for better cache efficiency
    for i in range(0, n_variants, block_size):
        i_end = min(i + block_size, n_variants)
        block_i = standardized_genotypes[:, i:i_end]
        
        # Compute diagonal block
        corr_matrix[i:i_end, i:i_end] = np.dot(block_i.T, block_i)
        
        # Compute off-diagonal blocks
        for j in range(i + block_size, n_variants, block_size):
            j_end = min(j + block_size, n_variants)
            block_j = standardized_genotypes[:, j:j_end]
            
            # Compute correlation between blocks
            corr_block = np.dot(block_i.T, block_j)
            corr_matrix[i:i_end, j:j_end] = corr_block
            corr_matrix[j:j_end, i:i_end] = corr_block.T  # Symmetric
    
    return corr_matrix
```

### Optimal Thresholds (If Re-implementing)

Based on extensive benchmarking with 1,000 samples:
- Use blocked if: n_samples ≤ 100 and n_variants ≥ 5,000
- Use blocked if: n_samples ≤ 500 and n_variants ≥ 10,000
- Use direct computation for all other cases

The blocked approach has overhead that makes it slower for typical use cases (≥1,000 samples) or when both dimensions are large.

## BCOR and BGEN I/O Optimizations (January 2025)

Following the correlation computation optimizations, we further optimized the I/O operations:

### BCOR Format Optimizations

1. **Vectorized Read/Write Operations**:
   - BCOR writer: Uses `np.triu_indices` to extract upper triangle values efficiently
   - BCOR reader: Vectorized matrix filling using the same indexing approach
   - Removed unnecessary helper methods in favor of inline vectorized operations
   - Result: Cleaner code with better performance

2. **Consistent Implementation**:
   - Both reader and writer use upper triangle indexing
   - Efficient type conversions using precomputed shift factors
   - Memory-mapped file access in reader for large files

### BGEN Format Optimizations

1. **Sample Filtering**:
   - `get_sample_indices`: Replaced loop-based filtering with numpy's `np.isin` and array operations
   - Preserves order of requested samples while using vectorized operations
   - Result: Much faster for large sample lists

2. **Correlation-Preserving Transform**:
   - Vectorized min/max computation using `np.nanmin` and `np.nanmax`
   - Pre-compute scales and shifts for all variants at once
   - Only loop through variants with valid ranges
   - Result: Significant speedup for transforming adjusted genotypes

### Performance Impact

- BCOR operations: Fully vectorized for optimal performance
- BGEN sample filtering: ~10x faster for large sample lists
- Correlation-preserving transform: ~3-5x faster through vectorization
- All optimizations maintain backward compatibility and pass tests

## Test Suite Reorganization (January 2025)

The test suite was reorganized for better maintainability and clarity:

1. **Created `test_covariate.py`**: Consolidated all covariate-related tests
   - Moved from `test_compute.py`: standardization, adjustment, and loading tests
   - Moved from `test_io.py`: covariate file loading tests
   - Better separation of concerns

2. **Consolidated `test_io.py`**: Merged sample filtering tests
   - Integrated tests from `test_sample_filtering.py`
   - All I/O operations now in one place
   - Removed redundant test file

3. **Improved test coverage**: 
   - Fixed edge case tests for categorical encoding
   - Updated tests to match current API behavior
   - Maintained 81% overall coverage with all tests passing

This reorganization makes it easier to:
- Find and run specific test categories
- Maintain test code with clear module boundaries
- Add new tests in the appropriate location

## CI/CD and Packaging (January 2025)

### GitHub Actions CI
- **Python versions**: Tests run on Python 3.8, 3.9, 3.10, and 3.11
- **Workflow**: Located in `.github/workflows/ci.yml`
- **Checks performed**:
  - Code linting with flake8 (max-line-length: 100)
  - Code formatting check with black
  - Full test suite with pytest and coverage reporting
  - Package building and validation with twine
- **Branch support**: CI runs on both `main` and `master` branches

### Package Configuration
- **Build system**: Uses setuptools with pyproject.toml configuration
- **Package discovery**: Automatic discovery with `[tool.setuptools.packages.find]`
- **License format**: Uses `{text = "MIT License"}` table format for compatibility with older setuptools
- **Python requirement**: `>=3.8` (dropped 3.6/3.7 support)
- **Version management**: Dynamic versioning with setuptools_scm

### Known Issues and Workarounds
- **License deprecation warning**: Newer setuptools versions show a deprecation warning for the table-style license format, but this is necessary for Python 3.8 compatibility
- **MANIFEST.in warning**: "no files found matching '*.py' under directory 'examples'" is expected as examples directory contains only data files

### Development Dependencies
- Core: numpy>=1.19.0, pandas>=1.0.0, bgen, gcsfs>=0.7.0, tqdm>=4.50.0
- Dev: pytest>=6.0.0, pytest-cov>=2.10.0, black>=20.8b1, flake8>=3.8.0

## BGI-Optimized BGEN Reader (January 2025)

### Overview
ldcov now **requires** BGEN index (BGI) files for all BGEN operations. This eliminates the inefficient double-scanning pattern and provides significant performance improvements.

### BGI Requirements
- All BGEN files must have a corresponding `.bgi` index file
- Create BGI files using: `bgenix -g your_file.bgen`
- BGI files are SQLite databases containing variant metadata

### Implementation Details

1. **Custom BGI Reader** (`ldcov/io/bgi_reader.py`):
   - Direct SQLite queries for variant metadata
   - Returns numpy-friendly data structures
   - Supports querying by region, position/alleles, or all variants
   - No dependency on bgen library's index module

2. **Simplified BGEN Reader**:
   - Single unified loading method reduces code duplication
   - Direct seeking to file offsets from BGI
   - Pre-allocated arrays for memory efficiency
   - No variant scanning - all metadata from BGI

3. **Performance Improvements**:
   - **All variants**: ~50% faster (no initial scan)
   - **Filtered variants**: ~70-90% faster (no full file scan)
   - **Region queries**: Cleaner implementation
   - **Memory usage**: Lower peak memory

### Key Optimizations

1. **Efficiency**:
   - Pre-allocated numpy arrays based on BGI metadata
   - Batch SQLite queries with row factory for named access
   - Direct file offset seeking for dosage loading
   - Minimal data structure conversions

2. **Minimal Redundancy**:
   - Single `_load_variants` method handles all scenarios
   - Unified variant metadata format throughout
   - Reuse of existing dosage processing code
   - No duplicate scanning or metadata collection

3. **Clean Architecture**:
   - BGI reader handles all SQLite operations
   - BGEN reader focuses on dosage extraction
   - Clear separation of metadata and genotype data

### Migration Guide

For users with existing BGEN files:
```bash
# Create BGI index for your BGEN files
bgenix -g your_file.bgen

# ldcov will now use the BGI for efficient loading
ldcov --bgen your_file.bgen --compute-ld --out results
```

### Error Handling
- Missing BGI files raise clear error with instructions
- Invalid BGI files detected via table structure check
- Helpful messages guide users to create indices

## BGEN Library Dependency (January 2025)

### Custom BGEN Fork
ldcov uses a custom fork of the bgen library with critical memory initialization fixes:
- GitHub: https://github.com/mkanai/bgen
- Specific commit: 99839781e932be6ed0b4cb3ff948b75eec2fc663
- Fixes transient NaN errors caused by uninitialized memory in the original bgen library

### Installation
The custom bgen version is specified in `pyproject.toml` and will be automatically installed:
```bash
# Install ldcov with the fixed bgen library
pip install -e .

# Or install from GitHub
pip install git+https://github.com/mkanai/ldcov.git
```

### Memory Initialization Fixes
The custom bgen fork includes:
1. Replaced `np.empty()` with `np.zeros()` in Python interface
2. Zero-initialized C++ arrays with `new char[size]()`
3. Proper string initialization with `resize(len, '\0')`
4. Bounds checking for array operations

These fixes prevent transient errors where uninitialized memory could contain extreme values causing numeric overflow and NaN propagation in LD calculations.