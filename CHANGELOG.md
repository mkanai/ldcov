# Changelog

All notable changes to **ldcov** are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) (0.x: minor versions may
make breaking changes while the API stabilizes).

Versions are derived from git tags via `setuptools_scm`; see the "Versioning" note
at the bottom.

## [0.6.0] - 2026-11-21

### Changed
- **Requires lazybgen 0.3.0 or later.** 0.3.0 reads 8-bit zlib BGEN files 1.1-1.2x
  faster than 0.2.0 at 5k samples and 1.4x at 500k, sizes its default worker pool to
  the CPUs the job may use (CPU affinity and cgroup quota), and fixes v1.1 and zlib
  decoding to follow the BGEN spec. The `load_bgen` API ldcov calls is unchanged.

### Removed
- **Python 3.9 is no longer supported**; the floor is now 3.10, which lazybgen already
  required.

## [0.5.0] - 2026-07-04

### Changed (breaking)
- **`VariantIndex.match()` no longer matches the ref/alt swap by default.** Matching is now
  exact-orientation only unless the caller opts in with `allow_allele_swap=True`. Previously
  `match()` always tried the `(alt, ref)` swap and returned `flip=True`; that implicit swap
  could silently sign-flip r for variants the caller intended to match exactly. A variant
  present only in swapped orientation now returns `(None, None)` by default.
- The z-file path of `extract_ld` / `ldcov --ld-bm` likewise defaults to exact orientation;
  the opt-out `--no-allele-swap` flag is replaced by the opt-in `--allow-allele-swap`.

### Added
- `allow_allele_swap` parameter (default `False`) on `VariantIndex.match()` and
  `extract_ld()`, and the `--allow-allele-swap` flag on `ldcov --ld-bm`, to re-enable
  ref/alt-swap matching (with r sign-flipped) when desired.
- `VariantIndex.match_variants(variants, allow_allele_swap=False)`: a vectorized batch matcher
  that issues one `query_region` per contig (instead of a pyarrow scan per variant) and
  returns a DataFrame of `idx` (−1 where unmatched) and `flip`, aligned to the input.
  Intended for locus-scale callers (e.g. SLALOM) that previously re-implemented this.

## [0.4.0] - 2026-06-24

First release on PyPI: `pip install ldcov`.

### Changed
- **BGEN reading now uses the standalone [`lazybgen`](https://pypi.org/project/lazybgen/)
  package** instead of a vendored C++/Cython reader. lazybgen ships prebuilt binary
  wheels (Linux manylinux/musllinux, macOS arm64, CPython 3.9–3.13), so ldcov is now
  pure Python and needs no C/C++ toolchain or CMake to install. The `load_bgen` API
  and behavior are unchanged.
- `__version__` is derived from the installed package metadata (was hardcoded).

### Fixed
- **Long-format LD files (`.ld` / `.ld.gz` / `.ld.bgz`) now round-trip correctly**
  (to the `%.6f` text precision of the writer). Previously `load_correlation_matrix`
  could not read back what `save_correlation_matrix` wrote: the `#`-prefixed header was
  discarded (raising `KeyError`), multiallelic variants sharing chrom/pos/rsid collapsed
  into one row, string metadata such as chromosome `"01"` was coerced to integers, and
  `.bgz` files failed to decompress. The writer also now stores the matrix diagonal, so
  non-unit diagonals (covariate-adjusted LD) and single-variant metadata are preserved.
- `standardize_genotypes(scale=False)` no longer raises `UnboundLocalError`.
- Covariate provenance metadata records the actual package version, not a literal.

### Removed
- The vendored BGEN reader (`ldcov/io/bgen/`, zlib-ng/zstd submodules, `setup.py`),
  now provided by lazybgen. The `tqdm` dependency (unused) was dropped.

## [0.3.0] - 2026-06-20

### Added
- **Hail BlockMatrix LD extraction (`--ld-bm`)**: read a partial submatrix of a Hail
  `BlockMatrix` LD store (e.g. gnomAD on GCS, Pan-UKB on AWS S3) in pure Python (no
  Hail/Spark) and export `.bcor` / `.npz`, selected by region, z-file, or index range.
- Pre-computed Parquet variant indexes for the supported LD stores, and the
  `make_bm_variant_index` builder.

### Changed
- Faster `.bcor` subset reads; vectorized mean imputation, FWL projection, and the
  long-format LD writer.

## [0.2.0] - 2026-04-26

### Added
- **`.bcor.idx` index format**, auto-emitted alongside `.bcor` outputs, for O(1) rsid
  lookups and partial reads (including over GCS) without scanning metadata.
- GCS-aware `BcorReader` with partial reads by rsid.
- `make_bcor_idx` tool to index existing `.bcor` files.

### Fixed
- Correctly detect string columns under pandas `future.infer_string` mode.

## [0.1.0] - 2025-12-05

### Added
- Initial release: efficient linkage-disequilibrium computation with Frisch–Waugh–Lovell
  covariate adjustment for BGEN data; `matrix` / `long` / `bcor` output formats;
  pre-computed projection matrices; and direct reads from Google Cloud Storage.

---

## Versioning

The package version is computed at build time from the latest git tag by
`setuptools_scm` (no version string is committed). Tagging `vX.Y.Z` and building
produces version `X.Y.Z`; commits after a tag get a development version like
`X.Y.(Z+1).devN+g<hash>`. `ldcov.__version__` reads this from the installed package
metadata (`importlib.metadata`), falling back to `0.0.0+unknown` when run from an
unbuilt source tree.

[0.6.0]: https://github.com/mkanai/ldcov/releases/tag/v0.6.0
[0.5.0]: https://github.com/mkanai/ldcov/releases/tag/v0.5.0
[0.4.0]: https://github.com/mkanai/ldcov/releases/tag/v0.4.0
[0.3.0]: https://github.com/mkanai/ldcov/releases/tag/v0.3.0
[0.2.0]: https://github.com/mkanai/ldcov/releases/tag/v0.2.0
[0.1.0]: https://github.com/mkanai/ldcov/releases/tag/v0.1.0
