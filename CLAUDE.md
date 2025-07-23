# CLAUDE.md

This file provides guidance to Claude Code when working with ldcov.

## Project Overview

ldcov is a Python package for efficient linkage disequilibrium (LD) calculation with covariate adjustment. It focuses exclusively on BGEN format genetic data and implements Frisch-Waugh-Lovell (FWL) projection to remove confounding effects from covariates.

**Key Features**:
- Pre-computed projection matrix support for efficient large-scale analyses
- Sample filtering during BGEN loading for memory efficiency
- Progress bars for variant loading operations (use `--progress` to enable)
- Flexible NaN handling with `--nan-action` (error/mean/omit)
- Custom Cython-based BGEN reader with BGI index requirement
- High-performance C++ batch decompressor with graceful Python fallback
- **NEW**: Direct Google Cloud Storage (GCS) support for BGEN files

## Development Commands

**IMPORTANT**: ALWAYS use `uv` for package management and Python operations.

```bash
# Install for development
uv pip install -e . --no-build-isolation

# Run tests
uv run pytest tests/
uv run pytest tests/test_specific.py::test_name  # Run specific test

# Code quality
make lint      # Check code style
make format    # Auto-format code
make coverage  # Run tests with coverage

# Build
make dist      # Build distribution packages
make clean     # Clean build artifacts

# Benchmarking
python benchmarks/benchmark_ldcov.py  # Run comprehensive benchmarks
python benchmarks/benchmark_ldcov_v2.py --test python-vs-v2  # Compare readers
python benchmarks/benchmark_phase1_optimizations.py  # Test GCS optimizations
# Docker benchmarking (recommended for reproducible results)
docker build -t ldcov:benchmark .
docker run --rm -v $(pwd)/benchmarks:/app/benchmarks ldcov:benchmark python benchmarks/benchmark_corrected_final.py

# Verify optimizations
python verify_optimizations.py  # Check that all optimizations are in place
```

## Example Workflows

```bash
# Pre-compute projection matrix
ldcov --precompute-projection -c covariates.txt --sample data.sample --out myproject

# Use pre-computed projection for LD computation  
ldcov --bgen data.bgen --projection-matrix myproject.proj.npz --compute-ld --out results

# Compute LD with covariate adjustment
ldcov --bgen data.bgen -c covariates.txt --compute-ld --out results

# Handle NaN values with mean imputation
ldcov --bgen data.bgen --compute-ld --nan-action mean --out results

# Read BGEN from Google Cloud Storage
ldcov --bgen gs://bucket/data.bgen --compute-ld --out results

# GCS with covariate adjustment
ldcov --bgen gs://bucket/data.bgen -c gs://bucket/covariates.txt --compute-ld --out results
```

## Architecture

- **cli/**: Command-line interface
- **compute/**: Core computation logic
  - `correlation.py`: LD computation functions
  - `covariate.py`: FWL projection and standardization
  - `projection.py`: Pre-computation and I/O for projection matrices
- **io/**: File format handlers
  - `bgen/`: High-performance Cython BGEN reader (BGI index required)
    - `reader.pyx`: Main BGEN reader implementation (V2, optimized)
    - `decompress/`: Modern decompressor architecture
      - `buffer_manager.h/.cpp`: Thread-safe buffer pool
      - `sequential_decompressor.h/.cpp`: Optimized sequential access with read-ahead
      - `parallel_decompressor.h/.cpp`: Multi-threaded parallel decompression
      - `memory_pool.h/.cpp`: Memory pool for reduced allocation overhead
      - `simd_utils.h/.cpp`: SIMD-optimized probability conversion
      - `thread_pool.h/.cpp`: Reusable thread pool for parallel operations
      - `compression_utils.h/.cpp`: Unified compression library interface
      - `decompressor_factory.h/.cpp`: Factory functions for creating decompressors
    - `format/`: BGEN format parsers (header, variant, genotype)
    - `index/`: BGI index reader with SQLite3 and LRU caching
    - `io/`: I/O abstraction layer with memory-mapped file support
    - `file_reader_wrapper.h`: Header-only Python file object wrapper
  - `bcor_reader.py`, `bcor_writer.py`: BCOR format I/O
  - `covariate_loader.py`: Covariate loading with categorical encoding
  - `correlation_io.py`: LD matrix output formats
- **utils/**: Supporting utilities
- **benchmarks/**: Performance testing and documentation
  - `phase2_docs/`: Phase 2 optimization results and analysis

## Key Implementation Notes

1. **Genotype Standardization**: L2 norm (not standard deviation)
2. **Covariate Adjustment**: QR decomposition for numerical stability
3. **Memory Efficiency**: Sample filtering during BGEN loading
4. **BGI Requirement**: All BGEN files must have `.bgi` index files
5. **BCOR Format**: Supports standard and extended formats
6. **C++ Decompressor**: Automatic backend selection based on file size and availability
   - ✅ **Phase 2 Optimizations Complete** (June 2025):
     - Removed redundant memcpy operations
     - SSD-aware prefetching optimization
     - Memory pool for reduced allocation overhead
     - SIMD-accelerated probability conversion
     - Dynamic batch sizing based on variant count
   - Falls back to Python implementation gracefully  
   - Supports zlib and zstd compression with vendored libraries (zlib-ng 2.2.4, zstd 1.5.7)

## Testing

- Test coverage: 81% overall
- Organized into focused test modules (test_cli.py, test_compute.py, etc.)
- Mock data generation for tests without file dependencies

## Code Style

- Black formatter (100-char line length)
- Type hints throughout
- Comprehensive logging

## Dependencies

- Core: numpy, pandas, gcsfs, tqdm
- Dev: pytest, pytest-cov, black, flake8, Cython
- Build: CMake ≥ 3.12 (for vendored compression libraries)
- Runtime: C++11 compatible compiler for optimal performance

## Reference Tools

- **bgenix**: Located at `reference_only/bgen_v1.2.0-osx-arm64/bgenix`
- **qctool**: Located at `reference_only/qctool_v2.2.1-osx/qctool`

### Docker Build with Vendored Libraries

To build Docker images with vendored libraries (recommended for consistency):

```bash
# Use the provided script that properly handles git submodules
./scripts/docker_build_with_vendored.sh

# Or manually prepare the build context:
./scripts/prepare_vendored_libs.sh
docker build -f Dockerfile -t ldcov .docker-build-context/
```

### Performance Benchmarks

**Phase 2 Optimization Results** (June 2025):
- Full file loading: 53-63% speedup over baseline
- Small files (1K×1K): 51% improvement (22.8 → 34.3 MB/s)
- Medium files (5K×5K): 63% improvement (58.2 → 94.7 MB/s)  
- Large files (10K×10K): 53% improvement (39.3 → 60.3 MB/s)
- Z-file filtering: 15-30% improvement depending on access pattern

**GCS Phase 1 Results** (June 2025):
- Sequential reads: 15-50% improvement depending on file size
- BGI cache: 7.6x speedup for 10-file workflows
- Reliability: 90% reduction in failures
- GWAS use case: 65.8x faster than downloading full file

**Key Performance Insights**:
- Decompression overhead eliminated (0% of runtime)
- Bottlenecks shift with file size: Python overhead → Memory bandwidth
- Peak performance at medium file sizes (cache-friendly)
- Consecutive variant access benefits most from optimizations
- GCS particularly effective for partial file reads

## Important Notes

- **BGI files are mandatory** - create with `bgenix -g file.bgen`
- BGEN files can be local or on Google Cloud Storage (gs://)
- Covariate files can be from Google Cloud Storage
- GCS BGI files are automatically downloaded to current directory (like bcftools)
- Package renamed from `pyldbm` to `ldcov` in January 2025
- Custom Cython BGEN reader with optimized performance
- No BGEN writing functionality (read-only)
- **C++ Decompressor**: Automatically enabled when successfully compiled
  - Significant performance improvement for large BGEN files
  - Uses vendored zlib-ng and zstd for consistent compression support
  - Graceful fallback to Python implementation if compilation fails
- **BGEN Reader V2**: High-performance implementation with Phase 1 & 2 optimizations complete

## BGEN Reader Architecture (January 2025)

ldcov uses a high-performance BGEN reader (V2) with modern C++ architecture:

### Key Features:
- **Performance**: 53-63% faster than baseline (Phase 2 optimizations)
- **Adaptive Decompressor**: Automatically selects optimal strategy (sequential/parallel)
- **SIMD Optimizations**: Hardware-accelerated probability conversion
- **Memory Efficiency**: Thread-safe buffer pooling and memory pool
- **Zero-copy**: Eliminated redundant memory copies
- **SSD-aware**: Optimized prefetching based on storage type

### Architecture:
- **C++ Core**: Modern C++11 implementation with move semantics
- **Decompressor Types**:
  - `adaptive`: Automatically switches based on access pattern (default)
  - `sequential`: Optimized for sequential access with read-ahead
  - `parallel`: Multi-threaded decompression for random access
- **Buffer Management**: Proper memory management prevents corruption
- **Index Integration**: Efficient BGI index usage with LRU caching

### Usage:
```python
# Standard usage
from ldcov.io import load_bgen
dosages, info, samples = load_bgen("file.bgen")

# Direct reader usage with decompressor control
from ldcov.io.bgen import BgenReader
with BgenReader("file.bgen", decompressor_type='parallel') as reader:
    dosages, info = reader.load_variants()
```

### Implementation Notes:
- No backward compatibility parameters - clean, performance-focused design
- Vendored compression libraries (zlib-ng, zstd) for consistency

## GCS BGEN Reader (June 2025)

ldcov supports reading BGEN files directly from Google Cloud Storage:

### Architecture:
- **GCSFileReader**: Implements FileReader interface using Python's gcsfs
- **Simple BGI Download**: Downloads BGI files to current directory (like bcftools)
- **Optimized Buffering**: 10MB read-ahead buffer for improved GCS performance
- **Range Requests**: Efficient partial file reading

### Phase 1 Optimizations (Completed June 2025):
1. **10MB Buffer Size**: 10x larger buffer reduces GCS API calls
   - 20-30% improvement for sequential reads
   - Most effective for files >10MB
   
2. **Retry Logic**: Exponential backoff with 3 retries
   - 90% reduction in transient failures
   - Automatic recovery from network issues
   
3. **BGI Memory Cache**: In-memory cache for BGI paths
   - 87.5% reduction in BGI lookup overhead
   - 100-400ms saved per operation

### Performance Results:
- **Sequential reads**: 15-50% improvement depending on file size
- **Reliability**: 90% fewer failures (1% → 0.1%)
- **Real-world scenarios**:
  - GWAS 100 variants from 100GB: 65.8x speedup vs downloading
  - Interactive analysis: 6.2x speedup with BGI cache
  - Batch processing: 17% faster with retry logic

### Key Components:
1. **File Reader Factory**: Automatically selects appropriate reader based on path
   - `gs://` paths → GCSFileReader
   - Local paths → MMapFileReader or RegularFileReader

2. **BGI Handling**:
   - BGI files downloaded to current directory when needed
   - Memory cache prevents redundant checks
   - Retry logic for download reliability
   - Simple and predictable behavior (similar to bcftools)

3. **Python/C++ Integration**:
   - Uses Python C API to call gcsfs from C++
   - Maintains GIL safety for thread compatibility
   - Graceful error handling with retry wrapper

### Usage:
```python
# Automatic GCS detection
from ldcov.io import load_bgen
dosages, info, samples = load_bgen("gs://bucket/data.bgen")

# BGI file will be downloaded to ./data.bgen.bgi if not present
# Subsequent accesses use memory cache for speed
# To re-download, simply delete the local .bgi file
```

### Future Optimizations (Phase 2):
- GCS-aware prefetching (2-3x additional improvement)
- GIL optimization for batch operations (30-50% improvement)
- Parallel region queries for random access

## Development Status (June 2025)

### Recently Completed:
- ✅ GCS BGEN reader implementation (feature/gcs-bgen-reader branch)
- ✅ Phase 1 GCS optimizations (10MB buffer, retry logic, BGI cache)
- ✅ Comprehensive benchmarking showing 15-65x speedup for various use cases
- ✅ Production-ready with zero API changes

### Current Focus:
- Testing and validation of GCS reader in real-world scenarios
- Documentation updates for GCS functionality
- Preparing for Phase 2 optimizations

### Upcoming Work:
- Phase 2 GCS optimizations (prefetching, GIL optimization)
- S3 and Azure blob storage support
- Streaming LD computation for very large matrices
- GPU acceleration for LD computation