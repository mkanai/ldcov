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

## Profiling

The benchmark system supports comprehensive profiling for both Python and C++ code to identify performance bottlenecks.

### Quick Start

```bash
# Profile a single commit
./benchmark_commits.sh --profile HEAD

# Profile multiple commits
./benchmark_commits.sh --profile commit1 commit2

# Analyze profiling results
./analyze_profiles.py benchmark_results/profiles/
```

### Important Notes

- **Performance Impact**: Profiling slows down execution significantly. Results are not comparable with regular benchmarks.
- **Single Run**: When profiling is enabled, only one run is performed (instead of the default 3).
- **Output Location**: Profile files are saved to `<output_dir>/profiles/`
- **Container Permissions**: The script automatically adds `--cap-add=SYS_ADMIN` for perf to work.

### Profile Output Files

For each profiled workflow, the following files are generated:

#### Python Profiling Files
- `<workflow>_<timestamp>_python.prof` - Raw cProfile data
- `<workflow>_<timestamp>_python_stats.txt` - Human-readable statistics (top 50 functions, callers)

#### C++ Profiling Files (if perf is available)
- `<workflow>_<timestamp>_perf.data` - Raw perf data
- `<workflow>_<timestamp>_perf_report.txt` - Perf report output
- `<workflow>_<timestamp>_flame.svg` - Flame graph (if flamegraph.pl is available)

### Analyzing Profile Results

```bash
# Analyze all profiles in a directory
./analyze_profiles.py benchmark_results/profiles/

# Generate only summary report
./analyze_profiles.py --summary-only benchmark_results/profiles/

# Manual analysis with snakeviz (interactive visualization)
snakeviz benchmark_results/profiles/*_python.prof

# Generate call graph
gprof2dot -f pstats profile.prof | dot -Tsvg -o callgraph.svg
```

### Identifying Optimization Opportunities

Look for:
1. Functions with high cumulative time (including subcalls)
2. Functions with high total time (excluding subcalls)
3. Functions called many times with small individual time
4. High CPU usage in specific C++ functions
5. Memory allocation hotspots

### Example Profiling Workflow

```bash
# 1. Profile the current implementation
./benchmark_commits.sh --profile HEAD

# 2. Analyze results
./analyze_profiles.py benchmark_results/profiles/

# 3. Identify top bottlenecks
cat benchmark_results/profiles/analysis/*_python_stats.txt | head -100

# 4. Implement optimizations
# ... make code changes ...

# 5. Profile the optimized version
./benchmark_commits.sh --profile HEAD

# 6. Compare results
diff benchmark_results/profiles/analysis/*_analysis.txt
```

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

### Profiling Issues

- **ImportError: No module named ldcov.__main__**: Fixed by using `python -m ldcov.cli.main` instead
- **Perf permission denied**: Ensure container runs with `--cap-add=SYS_ADMIN`
- **Missing flame graphs**: Install flamegraph.pl from brendangregg/FlameGraph
- **Empty profile files**: Check if command failed or execution time was too short

## Requirements

- Docker
- Python 3.7+ (for comparison script)
- Optional: matplotlib, seaborn (for plots)
- Optional: GNU parallel (for faster builds)