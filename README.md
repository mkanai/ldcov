# ldcov: Linkage Disequilibrium with Covariate adjustment

A Python package for efficient linkage disequilibrium (LD) calculation with covariate adjustment for BGEN format genetic data.

## Key Features

- **BGEN format support**: Efficient reading and writing of BGEN v1.2/v1.3 files
- **Covariate adjustment**: Remove confounding effects using Frisch-Waugh-Lovell (FWL) projection
- **Pre-computed projection matrices**: Compute QR decomposition once, reuse across multiple analyses
- **Flexible computation modes**: Compute LD only, save adjusted genotypes only, or both
- **Z-file support**: Filter and order variants based on external variant lists
- **Correlation-preserving transformation**: Save adjusted genotypes while maintaining LD structure
- **Region filtering**: Process specific genomic regions
- **Cloud storage for covariates**: Load covariate files directly from Google Cloud Storage (gs://)
- **Optimized for large datasets**: Efficient memory usage and computation

## Installation

```bash
# Install from GitHub (recommended)
pip install git+https://github.com/mkanai/ldcov.git

# For development
git clone https://github.com/mkanai/ldcov.git
cd ldcov
pip install -e .
```

**Note**: ldcov uses a custom fork of the bgen library with memory initialization fixes. This will be automatically installed when you install ldcov.

## Usage

### Command-Line Interface

The CLI uses flexible flags to control what operations to perform:

```bash
# Compute LD only (no adjusted genotypes saved)
ldcov --bgen input.bgen --out output --compute-ld

# Compute LD and save adjusted genotypes
ldcov --bgen input.bgen --out output --compute-ld --export-adjusted-bgen -c covariates.txt

# Only save adjusted genotypes (no LD computation)
ldcov --bgen input.bgen --out output --export-adjusted-bgen -c covariates.txt

# With region filtering
ldcov --bgen input.bgen --out output --compute-ld --region 1:1000000-2000000

# With Z-file for variant filtering and ordering
ldcov --bgen input.bgen --out output --compute-ld --z variants.z

# Specify custom BGEN index file
ldcov --bgen input.bgen --bgi input.bgen.bgi --out output --compute-ld

# Use covariate file from Google Cloud Storage
ldcov --bgen input.bgen --out output --compute-ld --export-adjusted-bgen -c gs://bucket/covariates.txt
```

### Pre-computed Projection Matrices (New!)

For large-scale analyses, you can pre-compute the covariate projection matrix once and reuse it:

```bash
# Step 1: Pre-compute projection matrix from covariates
ldcov --precompute-projection -c covariates.txt --sample data.sample --out myproject

# Step 2: Use pre-computed projection for LD computation
ldcov --bgen chr1.bgen --projection-matrix myproject.proj.npz --compute-ld --out chr1_results
ldcov --bgen chr2.bgen --projection-matrix myproject.proj.npz --compute-ld --out chr2_results
# ... process all chromosomes with the same projection matrix

# Alternative: Compute LD and save projection matrix for future use
ldcov --bgen input.bgen -c covariates.txt --compute-ld --save-projection --out results
```

This is particularly useful for:
- Processing multiple genomic regions with the same covariates
- Distributed computing across a cluster
- Iterative analyses with different variant filters

#### Output Files

Based on the flags used, ldcov will create:

- `--compute-ld`: Creates `{out}.ld` (matrix format) or `{out}.ld.gz` (long format)
- `--export-adjusted-bgen`: Creates `{out}.adj.bgen` and `{out}.adj.metadata.csv.gz`
- `--precompute-projection` or `--save-projection`: Creates `{out}.proj.npz`

### Python API

The package provides modular functions for flexibility:

```python
import ldcov

# Load and adjust genotypes
standardized_genotypes, variant_info, sample_ids, means, norms = ldcov.load_and_adjust_genotypes(
    genotype_file="data.bgen",
    covariate_file="covariates.txt",  # Optional
    region="1:1000000-2000000",        # Optional
    z_file="variants.z"                # Optional
)

# Compute LD from standardized genotypes
ldcov.compute_ld_from_standardized(
    standardized_genotypes=standardized_genotypes,
    variant_info=variant_info,
    output_file="output.ld"
)

# Save adjusted genotypes to BGEN
ldcov.save_adjusted_genotypes(
    standardized_genotypes=standardized_genotypes,
    variant_info=variant_info,
    sample_ids=sample_ids,
    output_file="adjusted.bgen",
    means=means,
    norms=norms
)

# Lower-level functions for custom workflows
genotypes, variant_info, sample_ids = ldcov.load_bgen("data.bgen")
standardized, means, norms = ldcov.standardize_genotypes(genotypes)
adjusted = ldcov.regress_out_covariates(standardized, covariates)

# Pre-computed projection matrix workflow
from ldcov.compute.projection import compute_projection_matrix, save_projection_matrix, load_projection_matrix

# Pre-compute projection
projection_data = compute_projection_matrix(
    covariate_file="covariates.txt",
    sample_ids=sample_ids
)
save_projection_matrix(projection_data, "myproject.proj.npz")

# Later: Load and use projection
projection_data = load_projection_matrix("myproject.proj.npz")
adjusted = ldcov.regress_out_covariates(
    standardized_genotypes,
    projection_matrix_Q=projection_data.Q
)
```

## Covariate File Format

Covariates should be provided as a text file with:

- First column: Sample IDs (must match BGEN file)
- Additional columns: Covariate values (numeric or categorical)
- Header row with column names

Example:

```
ID      PC1     PC2     batch
SAMPLE1 0.032   -0.011  A
SAMPLE2 -0.021  0.043   B
```

Supported formats: CSV, TSV, or whitespace-delimited text files. Can be loaded from local filesystem or Google Cloud Storage (gs://).

## Z-file Format

Z-files specify variants to include and their order:

```
rsid        chromosome  position  allele1  allele2
rs123456    1          1000000   A        G
rs789012    1          1000100   C        T
```

## Technical Details

### Genotype Standardization

Genotypes are standardized using L2 normalization:

1. Center by subtracting the mean
2. Scale by dividing by the L2 norm

This ensures that the dot product of standardized genotypes equals the Pearson correlation coefficient.

### Covariate Adjustment

The package uses Frisch-Waugh-Lovell (FWL) projection to remove covariate effects:

1. Standardize genotypes
2. Compute QR decomposition of the covariate matrix (with intercept)
3. Project out covariates using the orthogonal projection matrix Q
4. The residuals represent genotypes adjusted for covariate effects

For efficiency, the QR decomposition can be pre-computed once and reused across multiple analyses, as the projection matrix Q depends only on the covariates, not the genotypes.

### Correlation-Preserving Transformation

When saving adjusted genotypes:

1. Adjusted standardized genotypes are transformed back to allelic scale (0-2 range)
2. The transformation preserves the correlation structure
3. Re-standardizing the saved genotypes yields the same LD matrix

See [docs/adjusted_genotypes.md](docs/adjusted_genotypes.md) for mathematical details.

## Requirements

- Python 3.8+
- numpy >= 1.19.0
- pandas >= 1.0.0
- bgen (custom fork with memory fixes, automatically installed)
- gcsfs >= 0.7.0
- tqdm >= 4.50.0

## License

MIT License
