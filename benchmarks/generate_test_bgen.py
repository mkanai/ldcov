#!/usr/bin/env python3
"""
Generate test BGEN files with varying sample and variant sizes for benchmarking.

This script creates BGEN files using the external bgen library and generates
accompanying BGI index files using bgenix.

NOTE: The generated test_data/ directory is excluded from git.
Run this script locally to generate test files for benchmarking.
"""

import os
import sys
import time
import subprocess
import numpy as np
from pathlib import Path
import tempfile

try:
    # Try to import the installed bgen library
    from bgen import BgenWriter
except ImportError:
    # If not installed, try to add the source path
    sys.path.insert(0, str(Path(__file__).parent.parent / "reference_only" / "bgen" / "src"))
    try:
        from bgen import BgenWriter
    except ImportError:
        print("Error: Could not import bgen library")
        print("Please install it with: pip install git+https://github.com/mkanai/bgen.git")
        sys.exit(1)


def generate_random_genotypes(n_samples, n_alleles=2, ploidy=2):
    """Generate random genotype probabilities that sum to 1."""
    n_genotypes = 3 if n_alleles == 2 and ploidy == 2 else n_alleles ** ploidy
    
    # Generate random probabilities that sum to 1 for each sample
    probs = np.random.dirichlet(np.ones(n_genotypes), size=n_samples)
    
    return probs


def create_bgen_file(output_path, n_samples, n_variants, compression='zlib', bit_depth=8, 
                     use_zstd=False, layout=2):
    """
    Create a BGEN file with specified parameters.
    
    Args:
        output_path: Path to output BGEN file
        n_samples: Number of samples
        n_variants: Number of variants
        compression: 'zlib', 'zstd', or None (no compression)
        bit_depth: Bit depth for storing probabilities (8, 16, 32)
        use_zstd: Use zstd compression (overrides compression parameter)
        layout: BGEN layout version (2 for v1.2)
    """
    print(f"Creating BGEN file: {output_path}")
    print(f"  Samples: {n_samples:,}, Variants: {n_variants:,}")
    print(f"  Compression: {'zstd' if use_zstd else compression if compression else 'none'}")
    print(f"  Bit depth: {bit_depth}")
    
    # Generate sample IDs
    samples = [f"SAMPLE_{i:06d}" for i in range(n_samples)]
    
    # Create temporary sample file for bgenix
    sample_file = output_path.with_suffix('.sample')
    with open(sample_file, 'w') as f:
        f.write("ID_1 ID_2 missing\n")
        f.write("0 0 0\n")
        for sample in samples:
            f.write(f"{sample} {sample} 0\n")
    
    # Set compression type
    if use_zstd:
        compression = 'zstd'
    
    # Create BGEN writer
    start_time = time.time()
    writer = BgenWriter(
        str(output_path),
        n_samples=n_samples,
        compression=compression,
        layout=layout,
        samples=samples
    )
    
    # Generate chromosome names (use only chr1)
    chromosomes = ["chr1"] * n_variants
    
    # Write variants
    for i in range(n_variants):
        if i % 1000 == 0:
            print(f"  Writing variant {i:,}/{n_variants:,}...", end='\r')
        
        # Generate variant metadata
        chrom = chromosomes[i]
        pos = (i + 1) * 1000  # Space variants 1kb apart, starting at 1000
        rsid = f"rs{i+1000000}"
        varid = f"VAR_{chrom}_{pos}"
        alleles = ["A", "G"]
        
        # Generate random genotype probabilities
        genotypes = generate_random_genotypes(n_samples)
        
        # Add variant with genotype data (based on test examples)
        writer.add_variant(varid, rsid, chrom, pos, alleles, genotypes,
                         phased=False, bit_depth=bit_depth)
    
    writer.close()
    
    write_time = time.time() - start_time
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    
    print(f"\n  Write time: {write_time:.1f}s")
    print(f"  File size: {file_size:.1f} MB")
    print(f"  Speed: {n_variants / write_time:.0f} variants/sec")
    
    return output_path


def create_bgi_index(bgen_path):
    """Create BGI index using bgenix."""
    # Check if BGI already exists (created by the external bgen library)
    bgi_path = Path(str(bgen_path) + ".bgi")
    if bgi_path.exists():
        bgi_size = os.path.getsize(bgi_path) / (1024 * 1024)  # MB
        print(f"  BGI file already exists: {bgi_size:.1f} MB")
        return
    
    # Use system bgenix
    bgenix_path = "bgenix"
    
    print(f"Creating BGI index for {bgen_path}...")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [str(bgenix_path), "-g", str(bgen_path), "-index"],
            capture_output=True,
            text=True,
            check=True
        )
        index_time = time.time() - start_time
        print(f"  Index creation time: {index_time:.1f}s")
        
        # Check that BGI file was created
        if bgi_path.exists():
            bgi_size = os.path.getsize(bgi_path) / (1024 * 1024)  # MB
            print(f"  BGI file size: {bgi_size:.1f} MB")
        else:
            print("  Warning: BGI file not created")
            
    except subprocess.CalledProcessError as e:
        print(f"Error creating BGI index: {e}")
        if "stderr" in str(e):
            print(f"stderr: {e.stderr}")
    except FileNotFoundError:
        print("Error: bgenix not found. Please install bgenix or check the path.")


def main():
    """Generate test BGEN files with various configurations."""
    output_dir = Path(__file__).parent / "test_data"
    output_dir.mkdir(exist_ok=True)
    
    # Test configurations (using 8-bit as default)
    configs = [
        # Small files for quick testing
        {"n_samples": 1000, "n_variants": 1000, "compression": "zlib", "bit_depth": 8},
        {"n_samples": 1000, "n_variants": 5000, "compression": "zlib", "bit_depth": 8},
        
        # Medium files
        {"n_samples": 5000, "n_variants": 5000, "compression": "zlib", "bit_depth": 8},
        {"n_samples": 5000, "n_variants": 10000, "compression": "zlib", "bit_depth": 8},
        
        # Large files for performance testing
        {"n_samples": 10000, "n_variants": 10000, "compression": "zlib", "bit_depth": 8},
        {"n_samples": 10000, "n_variants": 20000, "compression": "zlib", "bit_depth": 8},
        
        # Very large file (50K samples)
        {"n_samples": 50000, "n_variants": 10000, "compression": "zlib", "bit_depth": 8},
        
        # Different compression types (all 8-bit)
        {"n_samples": 5000, "n_variants": 5000, "compression": None, "bit_depth": 8},  # No compression
        
        # Different bit depths for comparison
        {"n_samples": 5000, "n_variants": 5000, "compression": "zlib", "bit_depth": 16},  # Higher precision
        {"n_samples": 5000, "n_variants": 5000, "compression": "zlib", "bit_depth": 32},  # Highest precision
    ]
    
    # Add uncompressed tests with different sample sizes for thorough testing
    configs.extend([
        {"n_samples": 1000, "n_variants": 5000, "compression": None, "bit_depth": 8},   # Small uncompressed
        {"n_samples": 2500, "n_variants": 5000, "compression": None, "bit_depth": 8},   # Small-Medium uncompressed
        {"n_samples": 10000, "n_variants": 5000, "compression": None, "bit_depth": 8},  # Large uncompressed
        {"n_samples": 15000, "n_variants": 5000, "compression": None, "bit_depth": 8},  # Very Large uncompressed
        # Additional size test for break-even analysis
        {"n_samples": 7500, "n_variants": 5000, "compression": None, "bit_depth": 8},   # Medium-Large uncompressed
    ])
    
    # Add zstd compression tests with different sample sizes if supported
    try:
        import zstandard
        # Test zstd with various sample sizes to see how it scales
        configs.extend([
            {"n_samples": 1000, "n_variants": 5000, "use_zstd": True, "bit_depth": 8},   # Small
            {"n_samples": 5000, "n_variants": 5000, "use_zstd": True, "bit_depth": 8},   # Medium
            {"n_samples": 10000, "n_variants": 5000, "use_zstd": True, "bit_depth": 8},  # Large
            {"n_samples": 25000, "n_variants": 5000, "use_zstd": True, "bit_depth": 8},  # Very Large
            {"n_samples": 50000, "n_variants": 5000, "use_zstd": True, "bit_depth": 8},  # Extra Large
        ])
    except ImportError:
        print("Note: zstandard not available, skipping zstd compression tests")
    
    print(f"Generating {len(configs)} test BGEN files in {output_dir}\n")
    
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] Configuration:")
        
        # Generate filename
        n_samples = config["n_samples"]
        n_variants = config["n_variants"]
        compression = config.get("compression", "zlib")
        bit_depth = config.get("bit_depth", 16)
        use_zstd = config.get("use_zstd", False)
        
        if use_zstd:
            comp_str = "zstd"
        elif compression is None:
            comp_str = "nocomp"
        else:
            comp_str = compression
            
        filename = f"test_{n_samples}s_{n_variants}v_{comp_str}_{bit_depth}bit.bgen"
        output_path = output_dir / filename
        
        # Create BGEN file
        try:
            create_bgen_file(output_path, **config)
            
            # Create BGI index
            create_bgi_index(output_path)
            
        except Exception as e:
            print(f"Error creating {filename}: {e}")
            continue
        
        print()  # Empty line between files
    
    print("\nBenchmark data generation complete!")
    print(f"Files created in: {output_dir}")
    
    # Print summary
    print("\nSummary of generated files:")
    for bgen_file in sorted(output_dir.glob("*.bgen")):
        bgi_file = Path(str(bgen_file) + ".bgi")
        bgen_size = os.path.getsize(bgen_file) / (1024 * 1024)
        bgi_exists = "✓" if bgi_file.exists() else "✗"
        print(f"  {bgen_file.name:<50} {bgen_size:>8.1f} MB  BGI: {bgi_exists}")


if __name__ == "__main__":
    main()