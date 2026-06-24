# ldcov

[![CI](https://github.com/mkanai/ldcov/actions/workflows/ci.yml/badge.svg)](https://github.com/mkanai/ldcov/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/ldcov.svg)](https://pypi.org/project/ldcov/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Python package for efficient linkage disequilibrium (LD) calculation with covariate adjustment for BGEN format genetic data.

## Key Features

- **BGEN format support**: Efficient reading of BGEN v1.1/v1.2 files with mandatory BGI index, including streaming directly from Google Cloud Storage (gs://) without downloading
- **Covariate adjustment**: Remove confounding effects via Frisch-Waugh-Lovell (FWL) projection, with optional pre-computed projection matrices (compute the QR decomposition once, reuse across analyses)
- **Flexible LD computation**: With or without covariate adjustment, optionally filtered and ordered by a z-file or restricted to a genomic region
- **Multiple output formats**: tab-delimited matrix, gzipped long format, or binary `.bcor`
- **BCOR index (`.bcor.idx`)**: Auto-emitted alongside `.bcor` outputs for O(1) rsid lookups and partial reads (including over GCS) without scanning metadata
- **Hail BlockMatrix LD extraction (`--ld-bm`)**: Read a partial submatrix of a Hail `BlockMatrix` LD store (e.g. gnomAD on GCS, Pan-UKB on AWS S3) in pure Python (no Hail/Spark) and export `.bcor` / `.npz`, selected by region, z-file, or index range (see [below](#extracting-ld-from-a-hail-blockmatrix---ld-bm))

## Installation

### Requirements

- Python ≥ 3.9

[`lazybgen`](https://github.com/mkanai/lazybgen), the BGEN reader dependency, installs automatically from PyPI as a prebuilt binary wheel (Linux, macOS arm64), so no compiler is required.

### Standard Installation

```bash
# Install from PyPI
pip install ldcov

# Read BlockMatrix LD from AWS S3 (e.g. Pan-UKB) as well
pip install "ldcov[s3]"

# Latest development version from GitHub
pip install git+https://github.com/mkanai/ldcov

# For development
git clone https://github.com/mkanai/ldcov
cd ldcov
pip install -e ".[dev]"
```

### BGEN reading via lazybgen

ldcov reads BGEN files through [`lazybgen`](https://github.com/mkanai/lazybgen), a
standalone high-performance reader (formerly vendored inside ldcov). It statically
links **zlib-ng** (an optimized zlib replacement) and **zstd** for speed, and
supports partial reads directly from local files and cloud object stores (GCS
built-in; S3 via the `s3` extra).

`lazybgen` is installed automatically as a dependency. All BGEN files must have
accompanying BGI index files (create with `bgenix -g file.bgen`).

## Usage

### Cloud Storage (GCS) Support

ldcov can read BGEN files directly from Google Cloud Storage without downloading:

```bash
# Read BGEN from GCS
ldcov --bgen gs://bucket/data.bgen --compute-ld --out results

# With covariate adjustment (covariates can also be on GCS)
ldcov --bgen gs://bucket/data.bgen -c gs://bucket/covariates.txt --compute-ld --out results

# BGI index files are automatically downloaded to current directory
```

**Requirements**:

- gcsfs (installed as a dependency)
- BGI index files (`.bgen.bgi`) must exist alongside BGEN files on GCS
- Appropriate GCS credentials configured (via gcloud, service account, etc.)

**How it works**:

- BGEN files are streamed from GCS using efficient range requests
- BGI index files are downloaded to current directory (like bcftools)
- Smart buffering minimizes API calls and latency
- Compatible with all existing ldcov features

### Command-Line Interface

The CLI uses flexible flags to control what operations to perform:

```bash
# Compute LD only (no covariate adjustment)
ldcov --bgen input.bgen --out output --compute-ld

# Compute LD with covariate adjustment
ldcov --bgen input.bgen --out output --compute-ld -c covariates.txt

# Use specific columns as covariates
ldcov --bgen input.bgen --out output --compute-ld -c covariates.txt --covariate-cols PC1 PC2 PC3

# With region filtering
ldcov --bgen input.bgen --out output --compute-ld --region 1:1000000-2000000

# With Z-file for variant filtering and ordering
ldcov --bgen input.bgen --out output --compute-ld --z variants.z

# Specify custom BGEN index file
ldcov --bgen input.bgen --bgi input.bgen.bgi --out output --compute-ld

# Use covariate file from Google Cloud Storage
ldcov --bgen input.bgen --out output --compute-ld -c gs://bucket/covariates.txt
```

### Pre-computed Projection Matrices

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

- `--compute-ld --output-format matrix`: `{out}.ld` (default; tab-delimited matrix)
- `--compute-ld --output-format long`: `{out}.ld.gz` (gzipped long format)
- `--compute-ld --output-format bcor`: `{out}.bcor` (binary correlation format) and, by default, a `{out}.bcor.idx` index file (pass `--no-bcor-idx` to skip)
- `--precompute-projection` or `--save-projection`: `{out}.proj.npz`

### BCOR Index

When `--output-format bcor` is selected, ldcov also writes a small `.bcor.idx` index file that maps rsid to row and records per-variant byte offsets. This lets `BcorReader` resolve rsid-based queries without scanning the variable-length metadata block, which matters most when reading remote files:

```python
from ldcov.io import BcorReader

# Local or gs://, same API. The .bcor.idx auto-loads if present alongside the .bcor.
reader = BcorReader("gs://bucket/study.bcor")

# Partial read by rsid, fetching only the bytes needed (range-merged, parallelized for GCS).
subset, meta = reader.read_corr_by_rsid(["rs1234", "rs5678", "rs9012"])

# Two-list (asymmetric) query.
subset, meta = reader.read_corr_by_rsid(rsids_a, rsids2=rsids_b)
```

To generate an index for an existing `.bcor` file (e.g., LDstore output), run the
helper script from a clone of the ldcov repository (it is not installed with the package):

```bash
python scripts/make_bcor_idx.py path/to/file.bcor
```

The index binds to its parent `.bcor` via a header-level fingerprint, so stale or truncated pairs are detected at load time and the reader falls back gracefully.

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

## Extracting LD from a Hail BlockMatrix (`--ld-bm`)

Read a submatrix of a Hail `BlockMatrix` LD store (e.g. gnomAD) directly from cloud storage
(no Hail/Spark) and export it as `.bcor` (plus a `.variants.tsv` and optional `.npz`).

### One-time: build the variant index

The variant-to-matrix-index mapping lives in the matrix's companion `variant_indices.ht`. Convert it
once to a Parquet variant index on a machine with Hail installed, using the helper script from a
clone of the ldcov repository (it is not installed with the package):

```bash
python scripts/make_bm_variant_index.py \
    --ht gs://gcp-public-data--gnomad/release/2.1.1/ld/gnomad.genomes.r2.1.1.nfe.common.adj.ld.variant_indices.ht \
    --out gnomad_v2.nfe.b37.variant_index.parquet
```

The builder fails loudly on any multiallelic/monomorphic variant (LD matrices require split variants).

### Pre-computed variant indexes (gnomAD and Pan-UKB)

To skip the one-time Hail step, pre-computed variant indexes for the gnomAD and Pan-UKB LD matrices
are hosted at `gs://ldcov-requester-pays/`. The bucket is requester-pays, so reads are billed to your
own project (pass `--billing-project YOUR_PROJECT` to `gcloud storage`). Both individual Parquet files
and per-dataset `.tar.gz` bundles are available:

```bash
# List what's available
gcloud storage ls -r gs://ldcov-requester-pays/ --billing-project YOUR_PROJECT

# Download a single variant index (named <dataset>.<pop>.<build>.variant_index.parquet)
gcloud storage cp gs://ldcov-requester-pays/gnomad_v2.nfe.b37.variant_index.parquet . \
    --billing-project YOUR_PROJECT

# Or grab a whole dataset as a tar.gz bundle
gcloud storage cp gs://ldcov-requester-pays/bundles/gnomad_v2.b37.variant_index.tar.gz . \
    --billing-project YOUR_PROJECT
tar -xzf gnomad_v2.b37.variant_index.tar.gz
```

Per-dataset bundles:

- gnomAD GRCh37: `gs://ldcov-requester-pays/bundles/gnomad_v2.b37.variant_index.tar.gz`
- gnomAD GRCh38: `gs://ldcov-requester-pays/bundles/gnomad_v2.b38.variant_index.tar.gz`
- Pan-UKB GRCh37: `gs://ldcov-requester-pays/bundles/panukb.b37.variant_index.tar.gz`
- Pan-UKB GRCh38: `gs://ldcov-requester-pays/bundles/panukb.b38.variant_index.tar.gz`

Indexes come in `b37` (GRCh37) and `b38` (GRCh38) builds; pick the one matching your z-file / region
coordinates. gnomAD populations: `{afr, amr, asj, eas, est, fin, nfe, nwe, seu}`. Pan-UKB populations:
`{AFR, AMR, CSA, EAS, EUR, MID}`. Point `--variant-index` at the downloaded Parquet; no Hail install
required.

### Extract by region, z-file, or idx-range

```bash
# Genomic region
ldcov --ld-bm \
    --bm gs://gcp-public-data--gnomad/release/2.1.1/ld/gnomad.genomes.r2.1.1.nfe.common.adj.ld.bm \
    --variant-index gnomad_v2.nfe.b37.variant_index.parquet \
    --region 1:55000000-55100000 \
    --out region

# FINEMAP/SuSiE z-file (variants matched by locus+alleles; swapped alleles are sign-flipped).
ldcov --ld-bm \
    --bm gs://gcp-public-data--gnomad/release/2.1.1/ld/gnomad.genomes.r2.1.1.nfe.common.adj.ld.bm \
    --variant-index gnomad_v2.nfe.b37.variant_index.parquet \
    --z mystudy.z --out study --output-format both

# Explicit BlockMatrix index range
ldcov --ld-bm \
    --bm gs://gcp-public-data--gnomad/release/2.1.1/ld/gnomad.genomes.r2.1.1.nfe.common.adj.ld.bm \
    --variant-index gnomad_v2.nfe.b37.variant_index.parquet \
    --idx-range 5000:5500 --out slice
```

Outputs: `PREFIX.bcor` (+ `.bcor.idx`), `PREFIX.variants.tsv` (matrix-row order, with `flipped` /
matched columns), and `PREFIX.npz` when `--output-format npz|both`. Pairs outside the matrix's stored
band are filled with `NaN` (or `0` with `--fill zero`) and reported.

The needed blocks are fetched concurrently from cloud storage; tune with `--fetch-workers N`
(default 4) and `--block-cache N` (decoded-block LRU, default 4). For unmatched z-file variants,
`--on-missing {warn,error,drop}` controls the behavior. A ~10K-variant (3 Mb) region exports to
`.bcor` in a few seconds.

### Pan-UKB LD on AWS S3

The [Pan-UKB](https://pan.ukbb.broadinstitute.org/) LD matrices are public Hail BlockMatrices on S3
in the same format. Reading `s3://` requires the S3 extra; pair it with a pre-computed Pan-UKB variant
index (above) and extract as usual:

```bash
pip install "ldcov[s3]"

# Extract (s3:// is read anonymously by default)
ldcov --ld-bm \
    --bm s3://pan-ukb-us-east-1/ld_release/UKBB.EUR.ldadj.bm \
    --variant-index panukb.EUR.b37.variant_index.parquet \
    --region 1:55000000-55100000 \
    --out panukb_eur
```

Public-bucket reads are anonymous by default. To use credentials, a custom endpoint, or
requester-pays, pass `--storage-options` as a JSON dict, e.g.
`--storage-options '{}'` to force the normal AWS credential chain, or
`--storage-options '{"key": "AKIA...", "secret": "..."}'`.

### Python API (BlockMatrix)

```python
from ldcov.ld_bm import extract_ld

matrix, variants = extract_ld(
    bm_path="gs://gcp-public-data--gnomad/release/2.1.1/ld/gnomad.genomes.r2.1.1.nfe.common.adj.ld.bm",
    variant_index_path="gnomad_v2.nfe.b37.variant_index.parquet",
    region="1:55000000-55100000",
    out="region",
)
```

## Covariate File Format

Covariates should be provided as a text file with:

- A sample-ID column named `IID` by default (must match the BGEN sample IDs). Use a
  different column name via `--covariate-id-col`.
- Additional columns: Covariate values (numeric or categorical)
- Header row with column names

Example:

```
IID     PC1     PC2     batch
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

Allele convention: `allele1` = ref, `allele2` = alt. This applies throughout ldcov, including
`--ld-bm` extraction, where z-file variants are matched to the LD matrix by locus and alleles and
swapped alleles are sign-flipped.

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

## Dependencies

Installed automatically with the package:

- lazybgen >= 0.1 (BGEN reader; prebuilt binary wheel)
- numpy >= 1.19.0
- pandas >= 1.0.0
- gcsfs >= 0.7.0
- fsspec >= 2021.0.0
- lz4 >= 3.1.0
- pyarrow >= 6.0.0

Optional extra `ldcov[s3]` adds `s3fs` for reading BlockMatrix LD from AWS S3 (e.g. Pan-UKB).

## License

MIT License

## Citation

Kanai, M. et al. [Population-scale multiome immune cell atlas reveals complex disease drivers](https://doi.org/10.1101/2025.11.25.25340489). medRxiv (2025)

## Contact

Masahiro Kanai (<mkanai@broadinstitute.org>)
