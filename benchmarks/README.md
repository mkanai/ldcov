# ldcov Benchmarking System

This directory contains a comprehensive benchmarking system for comparing ldcov performance across different git commits.

## Components

1. **Dockerfile.benchmark**: Multi-stage Docker image builder that supports building ldcov from any commit
2. **run_benchmark.py**: Python script that runs the standard ldcov workflow and captures metrics
3. **benchmark_commits.sh**: Bash script that automates building and benchmarking multiple commits
4. **compare_results.py**: Python script that analyzes and compares results from different commits

## Quick Start

### Benchmark Multiple Commits

```bash
# Benchmark specific commits
./benchmark_commits.sh abc1234 def5678 master

# Compare the results
./compare_results.py benchmark_results/benchmark_*.json

# View the summary
cat comparison_results/benchmark_summary.md
```

### Standard Workflow

The benchmark runs the standard ldcov workflow:
```bash
ldcov \
    --bgen <bgen_file> \
    --bgi <bgi_file> \
    --sample <sample_file> \
    --projection <projection_matrix> \
    --z <variant_filter_file> \
    --compute-ld \
    --output-format bcor \
    --nan-action mean \
    --out <output_prefix>
```

## Test Data

The benchmarks use test data files in `test_data/` with various configurations:
- Sample sizes: 1000, 2500, 5000, 7500, 10000, 15000, 50000, 100000
- Variant counts: 1000, 5000, 10000, 20000
- Compression: zlib, no compression
- Bit depth: 8-bit, 16-bit, 32-bit

## Metrics Captured

- **Execution time**: Total, loading, computation phases
- **Memory usage**: Peak and average
- **Throughput**: Variants per second, MB per second
- **System info**: CPU, memory, platform details

## Output Files

### Benchmark Results
- `benchmark_results/benchmark_<commit>_<timestamp>.json`: Detailed results
- `benchmark_results/benchmark_<commit>_latest.json`: Latest results for each commit
- `benchmark_results/benchmark_summary_<timestamp>.txt`: Summary of benchmark run

### Comparison Results
- `comparison_results/benchmark_summary.md`: Markdown summary with tables
- `comparison_results/benchmark_details.csv`: Detailed CSV data
- `comparison_results/benchmark_comparison.json`: JSON comparison data
- `comparison_results/performance_timeline.png`: Performance plot
- `comparison_results/throughput_timeline.png`: Throughput plot
- `comparison_results/performance_heatmap.png`: Performance change heatmap

## Advanced Usage

### Custom Benchmark Options

```bash
# Use more runs for stability
./benchmark_commits.sh --num-runs 5 master

# Parallel builds
./benchmark_commits.sh --parallel 4 abc1234 def5678

# Custom output directory
./benchmark_commits.sh --output-dir my_results master
```

### Direct Docker Usage

```bash
# Build for specific commit
docker build --build-arg GIT_COMMIT=abc1234 -f benchmarks/Dockerfile.benchmark -t ldcov:benchmark-abc1234 .

# Run benchmark
docker run --rm \
    -v $(pwd)/benchmark_results:/results \
    -v $(pwd)/test_data:/data/test_data \
    ldcov:benchmark-abc1234 \
    python /app/benchmarks/run_benchmark.py
```

## Performance Regression Detection

The comparison script automatically detects performance regressions:
- Flags workflows that are >10% slower than baseline
- Highlights significant memory usage increases
- Identifies throughput degradations

## Tips

1. Run benchmarks on a quiet system for consistent results
2. Use `--num-runs 5` or higher for more stable measurements
3. Clear system caches between runs if possible (requires root)
4. Use the same hardware for comparing commits
5. Consider running benchmarks in Docker for isolation

## Troubleshooting

- **Build failures**: Check `benchmark_logs/build_<commit>.log`
- **Runtime errors**: Check `benchmark_logs/run_<commit>.log`
- **Missing test data**: Ensure `test_data/` directory is present
- **Permission issues**: Run with appropriate Docker permissions

## Requirements

- Docker
- Python 3.7+ (for comparison script)
- Optional: matplotlib, seaborn (for plots)
- Optional: GNU parallel (for faster builds)