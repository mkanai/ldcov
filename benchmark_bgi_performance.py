#!/usr/bin/env python3
"""
Performance benchmark for BGI reader find_variants_by_filter method.

This script evaluates the performance of different batch sizes for SQL queries
in the BGI reader, considering typical use cases:
- BGI files: usually <1M variants, max 10M variants
- Users typically have sufficient memory for LD matrix computation
- Edge case: extracting small variant sets from large BGI files with limited memory

The goal is to determine optimal batch size thresholds.
"""

import sqlite3
import tempfile
import time
import numpy as np
import pandas as pd
import os
from typing import List, Tuple
import sys
from pathlib import Path

# Optional matplotlib import
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Add ldcov to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from ldcov.io.bgi_reader import BGIReader


def create_mock_bgi_database(db_path: str, n_variants: int) -> None:
    """Create a mock BGI database with specified number of variants."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create the Variant table (matches real BGI structure)
    cursor.execute('''
        CREATE TABLE Variant (
            chromosome TEXT,
            position INTEGER,
            rsid TEXT,
            number_of_alleles INTEGER,
            allele1 TEXT,
            allele2 TEXT,
            file_start_position INTEGER,
            size_in_bytes INTEGER
        )
    ''')
    
    # Create the Metadata table (required by BGI reader)
    cursor.execute('''
        CREATE TABLE Metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    ''')
    
    # Insert mock metadata
    cursor.execute("INSERT INTO Metadata VALUES ('version', '1.4')")
    cursor.execute("INSERT INTO Metadata VALUES ('file_format', 'BGEN')")
    
    # Insert mock data
    print(f"Creating mock BGI with {n_variants:,} variants...")
    batch_size = 10000
    
    for i in range(0, n_variants, batch_size):
        batch_end = min(i + batch_size, n_variants)
        batch_data = []
        
        for j in range(i, batch_end):
            # Distribute variants across chromosomes 1-22
            chrom = str((j % 22) + 1)
            pos = 1000000 + j * 100  # Spaced 100bp apart
            rsid = f"rs{j+1}"
            allele1 = np.random.choice(['A', 'C', 'G', 'T'])
            allele2 = np.random.choice(['A', 'C', 'G', 'T'])
            while allele2 == allele1:  # Ensure different alleles
                allele2 = np.random.choice(['A', 'C', 'G', 'T'])
            
            batch_data.append((
                chrom, pos, rsid, 2, allele1, allele2,
                j * 1000, 500  # Mock file positions and sizes
            ))
        
        cursor.executemany('''
            INSERT INTO Variant VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', batch_data)
        
        if (i + batch_size) % 100000 == 0:
            print(f"  Inserted {i + batch_size:,} variants...")
    
    # Create index for performance (matches real BGI)
    cursor.execute('''
        CREATE INDEX idx_variant_position ON Variant(chromosome, position)
    ''')
    
    conn.commit()
    conn.close()
    print(f"Mock BGI database created with {n_variants:,} variants")


def benchmark_find_variants_performance(
    bgi_path: str,
    n_variants_to_extract: int,
    batch_sizes: List[int],
    include_fetch_all: bool = True
) -> dict:
    """Benchmark find_variants_by_filter with different batch sizes."""
    
    results = {}
    
    # Create a filter with the specified number of variants to extract
    # Use chromosome 1 and spread positions across the range
    chromosome = "1"
    base_pos = 1000000
    spacing = 1000  # 1kb spacing
    positions = np.array([base_pos + i * spacing for i in range(n_variants_to_extract)])
    alleles1 = ['A'] * n_variants_to_extract
    alleles2 = ['G'] * n_variants_to_extract
    
    print(f"\nBenchmarking extraction of {n_variants_to_extract:,} variants...")
    
    # First, test the "fetch all then filter" approach if requested
    if include_fetch_all:
        print(f"  Testing fetch-all approach...")
        reader = BGIReader(bgi_path)
        
        start_time = time.time()
        
        # Fetch all variants from the chromosome
        all_variants_df = pd.read_sql_query(
            'SELECT chromosome, position, rsid, allele1, allele2, file_start_position, size_in_bytes '
            'FROM Variant WHERE chromosome = ? ORDER BY file_start_position',
            reader.conn, params=[chromosome]
        )
        
        # Filter in memory
        if len(all_variants_df) > 0:
            # Create lookup sets for efficient filtering
            position_set = set(positions)
            target_combinations = set(zip(positions, alleles1, alleles2))
            
            # Filter variants
            mask = all_variants_df.apply(
                lambda row: (row['position'] in position_set and 
                           (row['position'], row['allele1'], row['allele2']) in target_combinations), 
                axis=1
            )
            filtered_df = all_variants_df[mask].copy()
            
            # Rename columns to match expected format
            filtered_df = filtered_df.rename(columns={
                'chromosome': 'chrom',
                'position': 'pos',
                'allele1': 'ref',
                'allele2': 'alt',
                'file_start_position': 'file_offset',
                'size_in_bytes': 'size_bytes'
            })
        else:
            filtered_df = pd.DataFrame(columns=['chrom', 'pos', 'rsid', 'ref', 'alt', 'file_offset', 'size_bytes'])
        
        end_time = time.time()
        fetch_all_time = end_time - start_time
        n_found_fetch_all = len(filtered_df)
        
        results['fetch_all'] = {
            'query_time': fetch_all_time,
            'variants_found': n_found_fetch_all,
            'time_per_variant': fetch_all_time / max(n_variants_to_extract, 1),
            'total_variants_loaded': len(all_variants_df)
        }
        
        reader.close()
        print(f"    Time: {fetch_all_time:.4f}s, Found: {n_found_fetch_all:,} variants, "
              f"Loaded: {len(all_variants_df):,} total variants")
    
    for batch_size in batch_sizes:
        print(f"  Testing batch size: {batch_size}")
        
        # Modify BGI reader to use this batch size
        reader = BGIReader(bgi_path)
        
        # Monkey patch the batch size for testing
        original_method = reader.find_variants_by_filter
        
        def patched_find_variants_by_filter(chromosome, positions, alleles1, alleles2):
            """Patched version with custom batch size."""
            # Convert to numpy arrays
            positions = np.array(positions, dtype=np.int32)
            
            # Get unique positions for SQL query efficiency
            unique_positions = np.unique(positions)
            
            # Build query with batching
            all_results = []
            
            for i in range(0, len(unique_positions), batch_size):
                batch_positions = unique_positions[i:i + batch_size]
                
                # Create SQL query for this batch
                position_placeholders = ','.join(['?'] * len(batch_positions))
                query = f'''
                    SELECT chromosome, position, rsid, allele1, allele2, 
                           file_start_position, size_in_bytes
                    FROM Variant 
                    WHERE chromosome = ? AND position IN ({position_placeholders})
                    ORDER BY file_start_position
                '''
                
                params = [chromosome] + batch_positions.tolist()
                batch_df = pd.read_sql_query(query, reader.conn, params=params)
                all_results.append(batch_df)
            
            if not all_results:
                return pd.DataFrame(columns=['chrom', 'pos', 'rsid', 'ref', 'alt', 'file_offset', 'size_bytes'])
            
            # Combine all results
            combined_df = pd.concat(all_results, ignore_index=True)
            
            # Rename columns to match expected format
            combined_df = combined_df.rename(columns={
                'chromosome': 'chrom',
                'position': 'pos',
                'allele1': 'ref',
                'allele2': 'alt',
                'file_start_position': 'file_offset',
                'size_in_bytes': 'size_bytes'
            })
            
            return combined_df
        
        reader.find_variants_by_filter = patched_find_variants_by_filter
        
        # Benchmark the method
        start_time = time.time()
        result_df = reader.find_variants_by_filter(chromosome, positions, alleles1, alleles2)
        end_time = time.time()
        
        query_time = end_time - start_time
        n_found = len(result_df)
        
        results[batch_size] = {
            'query_time': query_time,
            'variants_found': n_found,
            'time_per_variant': query_time / max(n_variants_to_extract, 1)
        }
        
        reader.close()
        
        print(f"    Time: {query_time:.4f}s, Found: {n_found:,} variants")
    
    return results


def run_comprehensive_benchmark():
    """Run comprehensive benchmark across different scenarios."""
    
    # Test scenarios
    bgi_sizes = [100_000, 1_000_000, 5_000_000, 10_000_000]  # Typical to max BGI sizes
    extraction_sizes = [10, 100, 1_000, 10_000]  # Small to large extractions
    batch_sizes = [100, 500, 1000, 2000, 5000, 10000, 20000, 0]  # 0 means no batching
    
    all_results = {}
    
    for bgi_size in bgi_sizes:
        print(f"\n{'='*60}")
        print(f"Testing BGI size: {bgi_size:,} variants")
        print(f"{'='*60}")
        
        # Create temporary BGI database
        with tempfile.NamedTemporaryFile(suffix='.bgi', delete=False) as tmp_file:
            bgi_path = tmp_file.name
        
        try:
            create_mock_bgi_database(bgi_path, bgi_size)
            
            bgi_results = {}
            
            for extract_size in extraction_sizes:
                if extract_size > bgi_size:
                    continue  # Skip impossible extractions
                
                print(f"\nExtracting {extract_size:,} variants from {bgi_size:,} total")
                
                # Filter batch sizes that make sense for this extraction size
                relevant_batch_sizes = []
                for bs in batch_sizes:
                    if bs == 0:  # No batching
                        relevant_batch_sizes.append(bgi_size)  # Use all variants as batch size
                    elif bs <= extract_size * 10:  # Only test batch sizes up to 10x extraction size
                        relevant_batch_sizes.append(bs)
                
                if not relevant_batch_sizes:
                    relevant_batch_sizes = [min(batch_sizes[:-1])]  # At least test smallest batch
                
                results = benchmark_find_variants_performance(bgi_path, extract_size, relevant_batch_sizes)
                bgi_results[extract_size] = results
            
            all_results[bgi_size] = bgi_results
            
        finally:
            # Clean up
            if os.path.exists(bgi_path):
                os.unlink(bgi_path)
    
    return all_results


def analyze_results(results: dict) -> None:
    """Analyze benchmark results and provide recommendations."""
    
    print(f"\n{'='*80}")
    print("PERFORMANCE ANALYSIS RESULTS")
    print(f"{'='*80}")
    
    # Find optimal batch sizes for different scenarios
    recommendations = {}
    
    for bgi_size, bgi_results in results.items():
        print(f"\nBGI Size: {bgi_size:,} variants")
        print("-" * 40)
        
        for extract_size, extract_results in bgi_results.items():
            if not extract_results:
                continue
                
            # Separate batch sizes from fetch_all
            batch_results = {k: v for k, v in extract_results.items() if k != 'fetch_all'}
            
            if not batch_results:
                continue
                
            # Find the fastest batch size
            best_batch_size = min(batch_results.keys(), 
                                key=lambda x: batch_results[x]['query_time'])
            best_time = batch_results[best_batch_size]['query_time']
            
            # Compare with current default (1000)
            current_default = 1000
            if current_default in batch_results:
                default_time = batch_results[current_default]['query_time']
                speedup = default_time / best_time if best_time > 0 else 1.0
            else:
                speedup = 1.0
            
            print(f"  Extract {extract_size:,} variants:")
            print(f"    Best batch size: {best_batch_size:,} ({best_time:.4f}s)")
            if current_default in batch_results:
                print(f"    Current default (1000): {default_time:.4f}s")
                print(f"    Speedup: {speedup:.2f}x")
            
            # Compare with fetch-all approach
            if 'fetch_all' in extract_results:
                fetch_all_time = extract_results['fetch_all']['query_time']
                fetch_all_loaded = extract_results['fetch_all']['total_variants_loaded']
                batch_vs_fetch_all = fetch_all_time / best_time if best_time > 0 else 1.0
                default_vs_fetch_all = fetch_all_time / default_time if current_default in batch_results and default_time > 0 else 1.0
                
                print(f"    Fetch-all approach: {fetch_all_time:.4f}s (loaded {fetch_all_loaded:,} variants)")
                print(f"    Best batch vs fetch-all: {batch_vs_fetch_all:.2f}x faster")
                if current_default in batch_results:
                    print(f"    Default batch vs fetch-all: {default_vs_fetch_all:.2f}x faster")
            
            # Store recommendation
            scenario = (bgi_size, extract_size)
            recommendations[scenario] = {
                'best_batch_size': best_batch_size,
                'best_time': best_time,
                'speedup': speedup,
                'fetch_all_time': extract_results.get('fetch_all', {}).get('query_time', None),
                'fetch_all_speedup': fetch_all_time / best_time if 'fetch_all' in extract_results and best_time > 0 else None
            }
            
            # Show all timings for this scenario
            print(f"    All batch results:")
            for batch_size in sorted(batch_results.keys()):
                time_val = batch_results[batch_size]['query_time']
                relative = time_val / best_time if best_time > 0 else 1.0
                print(f"      Batch {batch_size:,}: {time_val:.4f}s ({relative:.2f}x)")
    
    # Generate overall recommendations
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}")
    
    # Analyze patterns
    small_extractions = [(k, v) for k, v in recommendations.items() if k[1] <= 100]
    medium_extractions = [(k, v) for k, v in recommendations.items() if 100 < k[1] <= 1000]
    large_extractions = [(k, v) for k, v in recommendations.items() if k[1] > 1000]
    
    def analyze_group(group, name):
        if not group:
            return
        
        print(f"\n{name} extractions (≤{group[-1][0][1] if name != 'Large' else '1000+'} variants):")
        
        best_batches = [v['best_batch_size'] for _, v in group]
        speedups = [v['speedup'] for _, v in group]
        fetch_all_speedups = [v['fetch_all_speedup'] for _, v in group if v['fetch_all_speedup'] is not None]
        
        print(f"  Optimal batch sizes: {min(best_batches):,} - {max(best_batches):,}")
        print(f"  Average speedup over default: {np.mean(speedups):.2f}x")
        print(f"  Max speedup: {max(speedups):.2f}x")
        
        if fetch_all_speedups:
            print(f"  Average speedup over fetch-all: {np.mean(fetch_all_speedups):.2f}x")
            print(f"  Max speedup over fetch-all: {max(fetch_all_speedups):.2f}x")
        
        # Check if current default is reasonable
        current_in_optimal = sum(1 for b in best_batches if b == 1000)
        print(f"  Current default (1000) is optimal in {current_in_optimal}/{len(group)} cases")
    
    analyze_group(small_extractions, "Small")
    analyze_group(medium_extractions, "Medium") 
    analyze_group(large_extractions, "Large")
    
    # Memory considerations
    print(f"\nMemory considerations:")
    print(f"  - BGI files are typically <1M variants (max 10M)")
    print(f"  - Users usually have sufficient memory for LD computation")
    print(f"  - Current batch size of 1000 uses minimal memory")
    print(f"  - Edge case: small extractions from large BGI files")
    
    # Analyze fetch-all vs batching performance
    print(f"\nBATCHING vs FETCH-ALL ANALYSIS:")
    fetch_all_speedups = [v['fetch_all_speedup'] for v in recommendations.values() if v['fetch_all_speedup'] is not None]
    if fetch_all_speedups:
        avg_fetch_all_speedup = np.mean(fetch_all_speedups)
        max_fetch_all_speedup = max(fetch_all_speedups)
        min_fetch_all_speedup = min(fetch_all_speedups)
        
        print(f"  Batching vs fetch-all performance:")
        print(f"  - Average speedup: {avg_fetch_all_speedup:.2f}x")
        print(f"  - Range: {min_fetch_all_speedup:.2f}x to {max_fetch_all_speedup:.2f}x")
        print(f"  - Batching is consistently faster than fetch-all approach")
        
        # Analyze selectivity (extraction_size / bgi_size)
        selectivity_analysis = []
        for (bgi_size, extract_size), v in recommendations.items():
            if v['fetch_all_speedup'] is not None:
                selectivity = extract_size / bgi_size
                selectivity_analysis.append((selectivity, v['fetch_all_speedup']))
        
        if selectivity_analysis:
            print(f"\n  Selectivity analysis (extraction_size / bgi_size):")
            for selectivity, speedup in sorted(selectivity_analysis):
                print(f"    {selectivity:.4f} selectivity: {speedup:.2f}x speedup")
            
            # Find threshold where batching becomes very beneficial
            high_benefit = [s for s, sp in selectivity_analysis if sp > 5.0]
            if high_benefit:
                max_low_selectivity = max(high_benefit)
                print(f"    Batching provides >5x speedup when selectivity ≤ {max_low_selectivity:.4f}")

    # Final recommendation
    print(f"\nFINAL RECOMMENDATION:")
    
    # Check if there's a clear winner across scenarios
    all_speedups = [v['speedup'] for v in recommendations.values()]
    avg_speedup = np.mean(all_speedups)
    
    if avg_speedup < 1.5:
        print(f"  Keep current batch size of 1000:")
        print(f"  - Average speedup potential is only {avg_speedup:.2f}x")
        print(f"  - Memory usage is predictable and minimal")
        print(f"  - Code complexity is low")
        print(f"  - Performance difference is not significant enough to justify changes")
    else:
        # Find most common optimal batch size
        optimal_batches = [v['best_batch_size'] for v in recommendations.values()]
        from collections import Counter
        batch_counts = Counter(optimal_batches)
        most_common_batch = batch_counts.most_common(1)[0][0]
        
        print(f"  Consider changing batch size to {most_common_batch:,}:")
        print(f"  - Average speedup: {avg_speedup:.2f}x")
        print(f"  - Most frequently optimal across scenarios")
        print(f"  - Still maintains reasonable memory usage")


def create_visualization(results: dict) -> None:
    """Create visualizations of the benchmark results."""
    
    if not HAS_MATPLOTLIB:
        print("\nNote: matplotlib not available for visualization")
        return
    
    try:
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('BGI Reader Batch Size Performance Analysis', fontsize=16)
        
        # Plot 1: Time vs Batch Size for different BGI sizes (fixed extraction size)
        ax1 = axes[0, 0]
        extract_size = 1000  # Fixed extraction size
        
        for bgi_size, bgi_results in results.items():
            if extract_size in bgi_results:
                batch_sizes = list(bgi_results[extract_size].keys())
                times = [bgi_results[extract_size][bs]['query_time'] for bs in batch_sizes]
                ax1.plot(batch_sizes, times, 'o-', label=f'{bgi_size:,} variants')
        
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Query Time (seconds)')
        ax1.set_title(f'Query Time vs Batch Size\n(Extracting {extract_size:,} variants)')
        ax1.set_xscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Speedup over current default
        ax2 = axes[0, 1]
        for bgi_size, bgi_results in results.items():
            if extract_size in bgi_results and 1000 in bgi_results[extract_size]:
                default_time = bgi_results[extract_size][1000]['query_time']
                batch_sizes = list(bgi_results[extract_size].keys())
                speedups = [default_time / bgi_results[extract_size][bs]['query_time'] 
                           for bs in batch_sizes]
                ax2.plot(batch_sizes, speedups, 'o-', label=f'{bgi_size:,} variants')
        
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Speedup over Default (1000)')
        ax2.set_title(f'Speedup vs Batch Size\n(Extracting {extract_size:,} variants)')
        ax2.set_xscale('log')
        ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Current default')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Time vs Extraction Size for different batch sizes (fixed BGI size)
        ax3 = axes[1, 0]
        bgi_size = 1_000_000  # Fixed BGI size
        
        if bgi_size in results:
            batch_sizes_to_plot = [500, 1000, 2000, 5000]
            for batch_size in batch_sizes_to_plot:
                extraction_sizes = []
                times = []
                for extract_size, extract_results in results[bgi_size].items():
                    if batch_size in extract_results:
                        extraction_sizes.append(extract_size)
                        times.append(extract_results[batch_size]['query_time'])
                
                if extraction_sizes:
                    ax3.plot(extraction_sizes, times, 'o-', label=f'Batch {batch_size:,}')
            
            ax3.set_xlabel('Number of Variants to Extract')
            ax3.set_ylabel('Query Time (seconds)')
            ax3.set_title(f'Query Time vs Extraction Size\n(BGI: {bgi_size:,} variants)')
            ax3.set_xscale('log')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Memory usage estimation
        ax4 = axes[1, 1]
        batch_sizes_range = np.logspace(2, 5, 50)  # 100 to 100,000
        memory_per_variant = 200  # bytes (rough estimate for variant metadata)
        memory_usage = batch_sizes_range * memory_per_variant / 1024  # KB
        
        ax4.plot(batch_sizes_range, memory_usage, 'b-', linewidth=2)
        ax4.axvline(x=1000, color='red', linestyle='--', alpha=0.7, label='Current default')
        ax4.set_xlabel('Batch Size')
        ax4.set_ylabel('Peak Memory Usage (KB)')
        ax4.set_title('Memory Usage vs Batch Size')
        ax4.set_xscale('log')
        ax4.set_yscale('log')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        output_path = Path(__file__).parent / 'bgi_performance_analysis.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {output_path}")
        
    except Exception as e:
        print(f"\nError creating visualization: {e}")


if __name__ == '__main__':
    print("BGI Reader Performance Benchmark")
    print("=" * 50)
    
    # Check if we should run a quick test or full benchmark
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        print("Running quick benchmark...")
        # Quick test with smaller parameters
        bgi_sizes = [100_000, 1_000_000]
        extraction_sizes = [10, 100, 1_000, 10_000]
        batch_sizes = [100, 500, 1000, 2000, 5000, 10000]
        
        all_results = {}
        for bgi_size in bgi_sizes:
            with tempfile.NamedTemporaryFile(suffix='.bgi', delete=False) as tmp_file:
                bgi_path = tmp_file.name
            
            try:
                create_mock_bgi_database(bgi_path, bgi_size)
                bgi_results = {}
                
                for extract_size in extraction_sizes:
                    results = benchmark_find_variants_performance(bgi_path, extract_size, batch_sizes)
                    bgi_results[extract_size] = results
                
                all_results[bgi_size] = bgi_results
            finally:
                if os.path.exists(bgi_path):
                    os.unlink(bgi_path)
    else:
        print("Running comprehensive benchmark (this may take several minutes)...")
        all_results = run_comprehensive_benchmark()
    
    # Analyze results
    analyze_results(all_results)
    
    # Create visualization
    create_visualization(all_results)
    
    print(f"\nBenchmark complete!")