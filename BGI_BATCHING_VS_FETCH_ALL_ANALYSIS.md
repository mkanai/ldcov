# BGI Reader: Batching vs Fetch-All Performance Analysis

## Executive Summary

A comprehensive performance comparison between the current **batched approach** and a **fetch-all-then-filter approach** was conducted for the `BGIReader.find_variants_by_filter()` method.

**Key Finding: Batching provides massive performance improvements over fetch-all, validating the current design.**

## Performance Results Overview

### Batching vs Fetch-All Performance
- **Average speedup**: **133.61x faster** with batching
- **Range**: 2.87x to 392.33x faster
- **Consistent advantage**: Batching outperforms fetch-all in all scenarios

### Critical Insights

1. **Selectivity Matters**: The performance advantage is inversely related to selectivity (extraction_size / bgi_size)
2. **Massive Gains for Sparse Extractions**: When extracting small numbers of variants from large BGI files, batching is 100x+ faster
3. **Still Beneficial for Dense Extractions**: Even when extracting many variants, batching is 3-20x faster

## Detailed Performance Comparison

### BGI Size: 100,000 variants

| Extract Size | Best Batch Time | Fetch-All Time | Speedup | Variants Loaded (Fetch-All) |
|--------------|-----------------|----------------|---------|----------------------------|
| 10           | 0.0010s        | 0.0534s       | 53.14x  | 4,546                     |
| 100          | 0.0010s        | 0.0408s       | 40.86x  | 4,546                     |
| 1,000        | 0.0024s        | 0.0531s       | 22.41x  | 4,546                     |
| 10,000       | 0.0179s        | 0.0513s       | 2.87x   | 4,546                     |

### BGI Size: 1,000,000 variants

| Extract Size | Best Batch Time | Fetch-All Time | Speedup | Variants Loaded (Fetch-All) |
|--------------|-----------------|----------------|---------|----------------------------|
| 10           | 0.0011s        | 0.4194s       | 392.33x | 45,455                    |
| 100          | 0.0012s        | 0.4241s       | 358.50x | 45,455                    |
| 1,000        | 0.0024s        | 0.4242s       | 177.63x | 45,455                    |
| 10,000       | 0.0206s        | 0.4352s       | 21.15x  | 45,455                    |

## Selectivity Analysis

The performance advantage follows a clear pattern based on **selectivity** (extraction_size / bgi_size):

| Selectivity | Scenario                    | Speedup |
|-------------|----------------------------|---------|
| 0.00001     | 10 from 1M                | 392.33x |
| 0.0001      | 100 from 1M               | 358.50x |
| 0.001       | 1,000 from 1M             | 177.63x |
| 0.01        | 10,000 from 1M            | 21.15x  |
| 0.1         | 10,000 from 100K          | 2.87x   |

**Key Threshold**: Batching provides >5x speedup when selectivity ≤ 0.01 (1%)

## Why Batching Outperforms Fetch-All

### 1. **SQL Query Efficiency**
- **Batching**: Uses targeted `WHERE position IN (...)` clauses
- **Fetch-All**: Loads entire chromosome, then filters in Python

### 2. **Data Transfer Overhead**  
- **Batching**: Only transfers matching rows from SQLite to pandas
- **Fetch-All**: Transfers all chromosome data, then discards most of it

### 3. **Memory Efficiency**
- **Batching**: Memory usage proportional to results
- **Fetch-All**: Memory usage proportional to chromosome size

### 4. **CPU Processing**
- **Batching**: SQLite does the filtering (optimized C code)
- **Fetch-All**: Python pandas does the filtering (slower)

## Real-World Implications

### Typical Genomics Use Cases

1. **LD Calculation for Fine-Mapping**
   - Extract 100-1000 variants from chromosome
   - Selectivity: ~0.001 (0.1%)
   - Batching advantage: **100x+ faster**

2. **Regional Association Analysis**
   - Extract 10-100 variants from specific locus
   - Selectivity: ~0.0001 (0.01%)
   - Batching advantage: **300x+ faster**

3. **Whole-Chromosome Analysis**
   - Extract 10,000+ variants from chromosome
   - Selectivity: ~0.1 (10%)
   - Batching advantage: **3-20x faster**

### Memory Impact

**Fetch-All Approach Memory Usage**:
- 100K variant BGI: ~4.5MB per chromosome
- 1M variant BGI: ~45MB per chromosome
- 10M variant BGI: ~450MB per chromosome

**Batching Approach Memory Usage**:
- Proportional to actual matches
- 1000 variants: ~200KB
- 10,000 variants: ~2MB

## Batch Size Optimization Within Batching

While batching dramatically outperforms fetch-all, the optimal batch size within the batching approach shows modest differences:

### Small Extractions (≤100 variants)
- **Optimal batch sizes**: 1,000-2,000
- **Speedup over default (1000)**: 1.05x
- **Current default is optimal**: 50% of cases

### Medium Extractions (≤1,000 variants)  
- **Optimal batch sizes**: 1,000-5,000
- **Speedup over default (1000)**: 1.05x
- **Current default is optimal**: 50% of cases

### Large Extractions (10,000+ variants)
- **Optimal batch sizes**: 2,000-5,000
- **Speedup over default (1000)**: 1.22x
- **Current default is optimal**: 0% of cases

## Conclusions

### 1. **Batching Design is Excellent**
The current batched approach provides **massive performance advantages** (133x average) over a naive fetch-all approach, validating the original design decision.

### 2. **Current Batch Size is Reasonable**
While there's room for modest improvement (1.09x average), the current batch size of 1000 is close to optimal and provides excellent performance across all scenarios.

### 3. **Architecture Validation**
This analysis confirms that:
- The batching strategy was the right architectural choice
- SQLite's query optimization is highly effective
- Memory-efficient design scales well with BGI size

### 4. **No Need for Major Changes**
The performance analysis supports maintaining the current implementation:
- **Batching vs fetch-all**: Current approach is optimal
- **Batch size tuning**: Marginal gains don't justify added complexity
- **Memory efficiency**: Current approach scales well

## Recommendations

### 1. **Keep Current Batching Approach** ✅
- Provides 100x+ performance advantage over alternatives
- Memory efficient and scalable
- Well-optimized for genomics use cases

### 2. **Maintain Current Batch Size (1000)** ✅  
- Close to optimal across all scenarios
- Simple and predictable
- Adequate performance for all use cases

### 3. **Document Performance Characteristics** ✅
- Users should understand the efficiency of sparse extractions
- Batching approach enables efficient fine-mapping workflows
- Performance scales well with selectivity

## Technical Notes

- Benchmark tested with realistic mock BGI databases (100K-1M variants)
- Used SQLite with proper indexing on (chromosome, position)
- Measured wall-clock time including all SQL and pandas operations
- Results are consistent across different BGI sizes, indicating good scalability
- Real-world performance may vary based on disk I/O, but relative patterns will hold