"""
Setup configuration for ldcov package with Cython extensions.
"""

import os
import sys
import platform
import subprocess
import shutil
from pathlib import Path
import numpy as np
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
from distutils.ccompiler import new_compiler
from setuptools.command.build_ext import build_ext

# Determine platform-specific compile args
EXTRA_COMPILE_ARGS = ['-O3', '-funroll-loops', '-ffast-math']
EXTRA_LINK_ARGS = []

# Add SIMD optimizations for x86_64 architectures
if platform.machine() in ['x86_64', 'AMD64']:
    if sys.platform != 'win32':
        EXTRA_COMPILE_ARGS += ['-mavx', '-mavx2', '-mfma']

if sys.platform == 'darwin':
    EXTRA_COMPILE_ARGS += ['-stdlib=libc++', '-std=c++14']
elif sys.platform == 'linux':
    EXTRA_COMPILE_ARGS += ['-std=c++14']
elif sys.platform == 'win32':
    EXTRA_COMPILE_ARGS = ['/O2', '/std:c++14', '/arch:AVX2']

# Base directory for BGEN module
BGEN_DIR = Path("ldcov/io/bgen")
BUILD_DIR = Path("build")

def build_zlib_ng():
    """Build zlib-ng from submodule."""
    zlib_dir = BGEN_DIR / "zlib-ng"
    zlib_build_dir = BUILD_DIR / "zlib-ng"
    
    if not zlib_dir.exists():
        raise RuntimeError(
            f"zlib-ng submodule not found at {zlib_dir}\n\n"
            "The vendored compression libraries are missing. Please run:\n"
            "  git submodule update --init --recursive\n\n"
            "Or clone with submodules:\n"
            "  git clone --recursive https://github.com/mkanai/ldcov.git\n"
        )
    
    # Create build directory
    zlib_build_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure with CMake (use absolute paths)
    cmake_args = [
        "cmake",
        "-S", str(zlib_dir.absolute()),
        "-B", str(zlib_build_dir.absolute()),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DZLIB_COMPAT=ON",  # Enable zlib compatibility mode
        "-DBUILD_SHARED_LIBS=OFF",  # Build static library
        "-DZLIB_ENABLE_TESTS=OFF",  # Disable tests to avoid GTest dependency
        "-DWITH_GTEST=OFF",  # Disable GTest
        "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",  # Enable -fPIC for static libs
    ]
    
    if sys.platform == "darwin":
        # macOS specific flags
        cmake_args.extend([
            "-DCMAKE_OSX_DEPLOYMENT_TARGET=10.9",
        ])
    
    try:
        subprocess.check_call(cmake_args)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"CMake configuration failed with error: {e}\n\n"
            "Please ensure CMake is installed:\n"
            "  Ubuntu/Debian: sudo apt-get install cmake\n"
            "  macOS: brew install cmake\n"
            "  pip: pip install cmake\n"
        )
    
    # Build
    subprocess.check_call(["cmake", "--build", str(zlib_build_dir), "--config", "Release"])
    
    # Find the built library
    if sys.platform == "win32":
        lib_path = zlib_build_dir / "Release" / "zlibstatic.lib"
    else:
        lib_path = zlib_build_dir / "libz.a"
    
    if not lib_path.exists():
        raise RuntimeError(f"Failed to find built zlib-ng library at {lib_path}")
    
    # Return the build directory for headers since that's where zlib.h is generated
    return str(lib_path), str(zlib_build_dir)

def build_zstd():
    """Build zstd from submodule."""
    zstd_dir = BGEN_DIR / "zstd"
    zstd_lib_dir = zstd_dir / "lib"
    zstd_build_dir = BUILD_DIR / "zstd"
    
    if not zstd_dir.exists():
        raise RuntimeError(
            f"zstd submodule not found at {zstd_dir}\n\n"
            "The vendored compression libraries are missing. Please run:\n"
            "  git submodule update --init --recursive\n\n"
            "Or clone with submodules:\n"
            "  git clone --recursive https://github.com/mkanai/ldcov.git\n"
        )
    
    # Create build directory
    zstd_build_dir.mkdir(parents=True, exist_ok=True)
    
    # Build using make (simpler than CMake for zstd)
    if sys.platform == "win32":
        # On Windows, use CMake
        cmake_args = [
            "cmake",
            "-S", str(zstd_dir / "build" / "cmake"),
            "-B", str(zstd_build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DZSTD_BUILD_PROGRAMS=OFF",
            "-DZSTD_BUILD_SHARED=OFF",
            "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",  # Enable -fPIC for static libs
        ]
        try:
            subprocess.check_call(cmake_args)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"CMake configuration failed with error: {e}\n\n"
                "Please ensure CMake is installed:\n"
                "  Ubuntu/Debian: sudo apt-get install cmake\n"
                "  macOS: brew install cmake\n"
                "  pip: pip install cmake\n"
            )
        subprocess.check_call(["cmake", "--build", str(zstd_build_dir), "--config", "Release"])
        lib_path = zstd_build_dir / "lib" / "Release" / "zstd_static.lib"
    else:
        # On Unix-like systems, compile directly
        # Compile all source files
        sources = [
            "common/debug.c",
            "common/entropy_common.c",
            "common/error_private.c",
            "common/fse_decompress.c",
            "common/pool.c",
            "common/threading.c",
            "common/xxhash.c",
            "common/zstd_common.c",
            "compress/fse_compress.c",
            "compress/hist.c",
            "compress/huf_compress.c",
            "compress/zstd_compress.c",
            "compress/zstd_compress_literals.c",
            "compress/zstd_compress_sequences.c",
            "compress/zstd_compress_superblock.c",
            "compress/zstd_double_fast.c",
            "compress/zstd_fast.c",
            "compress/zstd_lazy.c",
            "compress/zstd_ldm.c",
            "compress/zstd_opt.c",
            "compress/zstdmt_compress.c",
            "decompress/huf_decompress.c",
            "decompress/zstd_ddict.c",
            "decompress/zstd_decompress.c",
            "decompress/zstd_decompress_block.c",
            "dictBuilder/cover.c",
            "dictBuilder/divsufsort.c",
            "dictBuilder/fastcover.c",
            "dictBuilder/zdict.c",
        ]
        
        # Check if we need to include assembly files
        asm_sources = []
        if platform.machine() in ['x86_64', 'AMD64'] and sys.platform == 'linux':
            asm_sources.append("decompress/huf_decompress_amd64.S")
        
        # Compile each source file
        compiler = new_compiler()
        if sys.platform == "darwin":
            compiler.compiler_so[0] = "clang"
            compiler.compiler[0] = "clang"
        
        objects = []
        for src in sources:
            src_path = zstd_lib_dir / src
            obj_name = src.replace("/", "_").replace(".c", ".o")
            
            compile_args = ["-O3", "-fPIC", "-I" + str(zstd_lib_dir), "-I" + str(zstd_lib_dir / "common")]
            if sys.platform == "darwin":
                compile_args.extend(["-stdlib=libc++"])
            
            # Compile and get the actual object file path
            obj_files = compiler.compile([str(src_path)], output_dir=str(zstd_build_dir), 
                                       extra_preargs=compile_args)
            if obj_files:
                objects.extend(obj_files)
        
        # Compile assembly files if any
        for asm_src in asm_sources:
            asm_path = zstd_lib_dir / asm_src
            asm_obj_path = zstd_build_dir / (asm_src.replace("/", "_").replace(".S", ".o"))
            
            # Use gcc/as to compile assembly
            asm_compile_cmd = ["gcc", "-c", "-fPIC", str(asm_path), "-o", str(asm_obj_path)]
            subprocess.check_call(asm_compile_cmd)
            objects.append(str(asm_obj_path))
        
        # Create static library
        lib_path = zstd_build_dir / "libzstd.a"
        if sys.platform == "darwin":
            subprocess.check_call(["ar", "rcs", str(lib_path)] + objects)
        else:
            subprocess.check_call(["ar", "rcs", str(lib_path)] + objects)
    
    if not lib_path.exists():
        raise RuntimeError(f"Failed to find built zstd library at {lib_path}")
    
    # For zstd, the headers are in the source lib directory
    return str(lib_path), str(zstd_lib_dir)

class CustomBuildExt(build_ext):
    """Custom build extension to build vendored libraries first."""
    
    def _write_build_config(self, backend_type):
        """Write build configuration to a Python module."""
        config_content = f'''# Auto-generated during build - DO NOT EDIT
# This file records which compression backend was used during compilation

COMPRESSION_BACKEND = "{backend_type}"


def get_build_info():
    """Get build-time configuration."""
    if COMPRESSION_BACKEND == "vendored":
        return {{
            "type": "vendored",
            "zlib": "zlib-ng 2.2.4 (zlib-compatible, optimized)",
            "zstd": "zstd 1.5.7",
            "note": "Using vendored high-performance compression libraries",
        }}
    else:
        return {{
            "type": "system",
            "zlib": "System zlib",
            "zstd": "System zstd",
            "note": "Using system compression libraries",
        }}
'''
        
        # Write to the bgen module directory
        config_path = BGEN_DIR / "_build_config.py"
        with open(config_path, 'w') as f:
            f.write(config_content)
        print(f"Wrote build configuration to {config_path}")
    
    def run(self):
        # Check for environment variable to control behavior
        use_system_libs = os.environ.get('LDCOV_USE_SYSTEM_LIBS', '').lower() in ('1', 'true', 'yes')
        
        if use_system_libs:
            print("Using system libraries as requested via LDCOV_USE_SYSTEM_LIBS")
            # Use system libraries
            for ext in self.extensions:
                if "bgen" in ext.name:
                    ext.libraries.extend(["z", "zstd"])
                    # Add decompress.cpp to sources (but not if already included)
                    decompress_path = str(BGEN_DIR / "decompress.cpp")
                    if "_bgen" in ext.name and decompress_path not in ext.sources:
                        ext.sources.append(decompress_path)
            
            # Write build configuration
            self._write_build_config("system")
        else:
            # Build vendored libraries (default behavior)
            try:
                print("Building vendored compression libraries...")
                zlib_lib, zlib_include = build_zlib_ng()
                zstd_lib, zstd_include = build_zstd()
                
                # Update extensions with library paths
                for ext in self.extensions:
                    if "bgen" in ext.name:
                        # Remove the hardcoded source paths and replace with build paths
                        # Remove existing vendored paths
                        ext.include_dirs = [d for d in ext.include_dirs 
                                          if "zlib-ng" not in d and "zstd/lib" not in d]
                        
                        # Add the correct paths: build dir for zlib-ng (has generated headers)
                        # and source dir for zstd
                        ext.include_dirs.extend([zlib_include, zstd_include])
                        
                        # Add library objects directly
                        ext.extra_objects.extend([zlib_lib, zstd_lib])
                        
                        # Add decompress.cpp to sources (but not if already included)
                        decompress_path = str(BGEN_DIR / "decompress.cpp")
                        if "_bgen" in ext.name and decompress_path not in ext.sources:
                            ext.sources.append(decompress_path)
                
                print("Successfully built vendored compression libraries")
                
                # Write build configuration
                self._write_build_config("vendored")
                
            except Exception as e:
                error_msg = f"""
Failed to build vendored compression libraries: {e}

The ldcov package requires building zlib-ng and zstd from source for optimal
performance and consistency. The build failed with the above error.

Possible solutions:
1. Ensure you have CMake installed: pip install cmake
2. Ensure you have a C++ compiler installed
3. Check the error message above for specific issues

If you want to use system libraries instead (not recommended), you can:
  LDCOV_USE_SYSTEM_LIBS=1 pip install ldcov

Note: Using system libraries may result in:
- Different behavior between systems
- Slower performance (zlib vs zlib-ng)
- Potential version incompatibilities
"""
                raise RuntimeError(error_msg)
        
        # Continue with normal build
        build_ext.run(self)

# Common define macros for NumPy compatibility
NUMPY_MACROS = [
    ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
    ("NPY_1_7_API_VERSION", "0x00000007")
]

# Define Cython extensions
extensions = [
    Extension(
        "ldcov.io.bgen.header",
        ["ldcov/io/bgen/header.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=EXTRA_COMPILE_ARGS,
        extra_link_args=EXTRA_LINK_ARGS,
        define_macros=NUMPY_MACROS,
        language="c++",
    ),
    Extension(
        "ldcov.io.bgen._bgen",
        ["ldcov/io/bgen/_bgen.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=EXTRA_COMPILE_ARGS,
        extra_link_args=EXTRA_LINK_ARGS,
        define_macros=NUMPY_MACROS,
        language="c++",
    ),
    Extension(
        "ldcov.io.bgen.variant",
        ["ldcov/io/bgen/variant.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=EXTRA_COMPILE_ARGS,
        extra_link_args=EXTRA_LINK_ARGS,
        define_macros=NUMPY_MACROS,
        language="c++",
    ),
    # High-performance BGEN reader (V2)
    Extension(
        "ldcov.io.bgen.reader",
        [
            "ldcov/io/bgen/reader.pyx",
            # Core implementation
            "ldcov/io/bgen/bgen_reader_impl.cpp",
            # BGI index
            "ldcov/io/bgen/index/bgi_reader.cpp",
            # "ldcov/io/bgen/index/simple_bgi_cache.cpp",  # BGI download now handled by Python
            # Format parsers
            "ldcov/io/bgen/format/bgen_header.cpp",
            "ldcov/io/bgen/format/variant_parser.cpp",
            "ldcov/io/bgen/format/genotype_parser.cpp",
            "ldcov/io/bgen/format/genotype_parser_simd.cpp",
            # IO
            "ldcov/io/bgen/io/mmap_reader.cpp",
            "ldcov/io/bgen/io/gcs_file_reader.cpp",
            # Decompression architecture
            "ldcov/io/bgen/decompress/decompressor_factory.cpp",
            "ldcov/io/bgen/decompress/compression_utils.cpp",
            "ldcov/io/bgen/decompress/buffer_manager.cpp",
            "ldcov/io/bgen/decompress/sequential_decompressor.cpp",
            "ldcov/io/bgen/decompress/sequential_fast_path.cpp",
            "ldcov/io/bgen/decompress/parallel_decompressor.cpp",
            "ldcov/io/bgen/decompress/memory_pool.cpp",
            "ldcov/io/bgen/decompress/simd_utils.cpp",
            "ldcov/io/bgen/decompress/thread_pool.cpp",
        ],
        include_dirs=[
            np.get_include(),
            "ldcov/io/bgen",  # Base directory
            "ldcov/io/bgen/io",  # IO headers including reader_interface.h
            "ldcov/io/bgen/index",  # BGI reader headers
            "ldcov/io/bgen/format",  # Format headers
            "ldcov/io/bgen/decompress",  # Decompressor headers
            "ldcov/io/bgen/zlib-ng",  # zlib-ng headers
            "ldcov/io/bgen/zstd/lib",  # zstd headers
        ],
        libraries=["sqlite3"],
        extra_compile_args=EXTRA_COMPILE_ARGS,
        extra_link_args=EXTRA_LINK_ARGS,
        define_macros=NUMPY_MACROS,
        language="c++",
    ),
    # V1 reader kept only for benchmarking/reference (not part of main API)
    Extension(
        "ldcov.io.bgen.reader_v1",
        ["ldcov/io/bgen/reader_v1.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=EXTRA_COMPILE_ARGS,
        extra_link_args=EXTRA_LINK_ARGS,
        define_macros=NUMPY_MACROS,
        language="c++",
    ),
    # Legacy decompressor for V1 (only for benchmarking)
    Extension(
        "ldcov.io.bgen._decompressor",
        [
            "ldcov/io/bgen/_decompressor.pyx",
            "ldcov/io/bgen/legacy/batch_decompressor.cpp",
            "ldcov/io/bgen/legacy/sequential_decompressor.cpp",
            "ldcov/io/bgen/legacy/buffer_pool.cpp",
            "ldcov/io/bgen/decompress.cpp"
        ],
        include_dirs=[np.get_include(), "ldcov/io/bgen/legacy"],
        extra_compile_args=EXTRA_COMPILE_ARGS,
        extra_link_args=EXTRA_LINK_ARGS,
        define_macros=NUMPY_MACROS,
        language="c++",
    ),
]

# Build extensions
ext_modules = cythonize(
    extensions,
    compiler_directives={
        'language_level': '3',
        'boundscheck': False,
        'wraparound': False,
        'nonecheck': False,
        'cdivision': True,
    },
    annotate=True,  # Generate HTML annotations
)

# Setup configuration is handled by pyproject.toml
# This file only handles the Cython extension building
if __name__ == "__main__":
    setup(
        ext_modules=ext_modules,
        cmdclass={"build_ext": CustomBuildExt},
        zip_safe=False,
    )
