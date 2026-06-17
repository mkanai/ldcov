# ldcov

A Python package for efficient linkage disequilibrium (LD) calculation with covariate adjustment for BGEN format genetic data.

## Key Features

- **BGEN format support**: Efficient reading of BGEN v1.1/v1.2 files with mandatory BGI index
- **Covariate adjustment**: Remove confounding effects using Frisch-Waugh-Lovell (FWL) projection
- **Pre-computed projection matrices**: Compute QR decomposition once, reuse across multiple analyses
- **Flexible LD computation**: With or without covariate adjustment
- **Z-file support**: Filter and order variants based on external variant lists
- **Region filtering**: Process specific genomic regions
- **Cloud storage support**: Load BGEN files and covariates directly from Google Cloud Storage (gs://)
- **Optimized for large datasets**: Efficient memory usage and computation
- **GCS BGEN reading**: Stream BGEN files from Google Cloud Storage without downloading
- **BCOR index (`.bcor.idx`)**: Auto-emitted alongside `.bcor` outputs to enable O(1) rsid lookups and partial reads (including over GCS) without scanning metadata
- **Hail BlockMatrix LD extraction (`--ld-bm`)**: Read a partial submatrix of a Hail `BlockMatrix` LD store (e.g. gnomAD on GCS, Pan-UKB on AWS S3) in pure Python — no Hail/Spark — and export `.bcor` / `.npz`, selected by region, z-file, or index range (see [below](#extracting-ld-from-a-hail-blockmatrix---ld-bm))

## Installation

### Requirements

- Python ≥ 3.8
- CMake ≥ 3.12 (for building compression libraries)
- C++ compiler (for building Cython extensions)

### Standard Installation

```bash
# Install from GitHub (recommended)
pip install git+https://github.com/mkanai/ldcov

# For development
git clone https://github.com/mkanai/ldcov
cd ldcov
git submodule update --init --recursive  # Get compression libraries
pip install -e .
```

### Compression Libraries

ldcov uses high-performance compression libraries for BGEN file reading:

- **zlib-ng**: An optimized zlib replacement (10-30% faster)
- **zstd**: Fast compression library

By default, ldcov builds these libraries from source for optimal performance and consistency. If the build fails, installation will stop with an error message.

#### Using System Libraries (Not Recommended)

If you cannot build the vendored libraries, you can use system libraries:

```bash
# Use system zlib and zstd libraries
LDCOV_USE_SYSTEM_LIBS=1 pip install git+https://github.com/mkanai/ldcov.git
```

**Warning**: Using system libraries may result in:

- Different behavior between systems
- Slower performance (standard zlib vs optimized zlib-ng)
- Potential version incompatibilities

#### Verifying Compression Backend

You can check which compression backend is being used:

```python
from ldcov.io.bgen._bgen import get_compression_backend
print(get_compression_backend())
# {'type': 'vendored', 'zlib': 'zlib-ng 2.2.4 (zlib-compatible, optimized)', ...}
```

**Note**: ldcov includes a custom Cython-based BGEN reader optimized for performance. All BGEN files must have accompanying BGI index files (create with `bgenix -g file.bgen`).

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

- Install gcsfs: `pip install gcsfs` (included in dependencies)
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

When `--output-format bcor` is selected, ldcov also writes a small `.bcor.idx` index file that maps rsid → row and records per-variant byte offsets. This lets `BcorReader` resolve rsid-based queries without scanning the variable-length metadata block, which matters most when reading remote files:

```python
from ldcov.io import BcorReader

# Local or gs:// — same API. The .bcor.idx auto-loads if present alongside the .bcor.
reader = BcorReader("gs://bucket/study.bcor")

# Partial read by rsid — fetches only the bytes needed (range-merged, parallelized for GCS).
subset, meta = reader.read_corr_by_rsid(["rs1234", "rs5678", "rs9012"])

# Two-list (asymmetric) query.
subset, meta = reader.read_corr_by_rsid(rsids_a, rsids2=rsids_b)
```

To generate an index for an existing `.bcor` file (e.g., LDstore output), run:

```bash
python scripts/make_bcor_idx.py path/to/file.bcor
```

The index binds to its parent `.bcor` via a header-level fingerprint, so stale or truncated pairs are detected at load time and the reader falls back gracefully.

## Extracting LD from a Hail BlockMatrix (`--ld-bm`)

Read a submatrix of a Hail `BlockMatrix` LD store (e.g. gnomAD) directly from cloud storage —
no Hail/Spark — and export it as `.bcor` (+ a `.variants.tsv` and optional `.npz`).

### One-time: build the variant index

The variant↔matrix-index mapping lives in the matrix's companion `variant_indices.ht`. Convert it
once to a Parquet variant index on a machine with Hail installed:

```bash
python scripts/make_bm_variant_index.py \
    --ht gs://gcp-public-data--gnomad/release/2.1.1/ld/gnomad.genomes.r2.1.1.nfe.common.adj.ld.variant_indices.ht \
    --out gnomad_nfe.variant_index.parquet
```

The builder fails loudly on any multiallelic/monomorphic variant (LD matrices require split variants).

### Extract by region, z-file, or idx-range

```bash
# Genomic region
ldcov --ld-bm \
    --bm gs://gcp-public-data--gnomad/release/2.1.1/ld/gnomad.genomes.r2.1.1.nfe.common.adj.ld.bm \
    --variant-index gnomad_nfe.variant_index.parquet \
    --region 1:55000000-55100000 \
    --out region

# FINEMAP/SuSiE z-file (variants matched by locus+alleles; swapped alleles are sign-flipped).
# z-file allele convention (same as the rest of ldcov): allele1 = ref, allele2 = alt.
ldcov --ld-bm --bm gs://.../...ld.bm --variant-index gnomad_nfe.variant_index.parquet \
    --z mystudy.z --out study --output-format both

# Explicit BlockMatrix index range
ldcov --ld-bm --bm gs://.../...ld.bm --variant-index gnomad_nfe.variant_index.parquet \
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
in the same format. Install the S3 extra, build a variant index from the population's .variant.ht, then
extract as usual:

```bash
pip install "ldcov[s3]"        # or: uv pip install -e ".[s3]"

# One-time variant index (needs Hail). GRCh37 index -> .variant.ht; GRCh38 (liftover) -> .variant.b38.ht.
python scripts/make_bm_variant_index.py \
    --ht s3://pan-ukb-us-east-1/ld_release/UKBB.EUR.ldadj.variant.ht \
    --out UKBB.EUR.variant_index.parquet

# Extract (s3:// is read anonymously by default)
ldcov --ld-bm \
    --bm s3://pan-ukb-us-east-1/ld_release/UKBB.EUR.ldadj.bm \
    --variant-index UKBB.EUR.variant_index.parquet \
    --region 1:55000000-55100000 \
    --out panukb_eur
```

> **Variant-index builder & S3:** `make_bm_variant_index.py` runs on Hail/Spark, and `ldcov`'s
> anonymous-S3 plumbing only covers the *runtime* read path — not Hail. To read the variant `.ht`
> directly from `s3://`, the Hail session must be configured for S3; for the public Pan-UKB bucket
> use anonymous access, e.g.
> `--ht s3a://...` with Spark conf
> `spark.hadoop.fs.s3a.aws.credentials.provider=org.apache.hadoop.fs.s3a.AnonymousAWSCredentialsProvider`.
> If your Hail environment isn't set up for S3, point `--ht` at a local copy of the `.ht` instead.
> Only this one-time step needs Hail/S3 config; the extract step below reads the BlockMatrix
> anonymously without Hail.

Populations: `{EUR, AFR, AMR, CSA, EAS, MID}` (substitute for `EUR` above). Choose the variant
index whose build matches your z-file / region coordinates: `.variant.ht` is GRCh37 (contigs like
`1`), `.variant.b38.ht` is GRCh38 (contigs like `chr1`).

Public-bucket reads are anonymous by default. To use credentials, a custom endpoint, or
requester-pays, pass `--storage-options` as a JSON dict, e.g.
`--storage-options '{}'` to force the normal AWS credential chain, or
`--storage-options '{"key": "AKIA...", "secret": "..."}'`.

### Python API (BlockMatrix)

```python
from ldcov.ld_bm import extract_ld

matrix, variants = extract_ld(
    bm_path="gs://.../...ld.bm",
    variant_index_path="gnomad_nfe.variant_index.parquet",
    region="1:55000000-55100000",
    out="region",
)
```

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

## Requirements

- Python 3.8+
- numpy >= 1.19.0
- pandas >= 1.0.0
- gcsfs >= 0.7.0
- tqdm >= 4.50.0

## License

MIT License

## Citation

Kanai, M. et al. [Population-scale multiome immune cell atlas reveals complex disease drivers](https://doi.org/10.1101/2025.11.25.25340489). medRxiv (2025)

## Contact

Masahiro Kanai (<mkanai@broadinstitute.org>)
