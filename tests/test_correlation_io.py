"""
Tests for correlation matrix I/O functionality.

This module tests:
- Saving and loading correlation matrices in different formats (matrix, long, bcor)
- BCOR format standard and extended formats
- Different compression levels
- Matrix validation and error handling
"""

import numpy as np
import pandas as pd
import os
import pytest
import tempfile
from pathlib import Path

from ldcov.io.correlation_io import save_correlation_matrix, load_correlation_matrix, BcorReader
from ldcov.io.bcor_writer import BcorWriter, save_bcor
from ldcov.compute.correlation import load_and_adjust_genotypes, compute_correlation_matrix
from ldcov.io.bcor_index import BcorIndexReader


@pytest.fixture(scope="module")
def test_data():
    """Set up test data for correlation I/O tests."""
    examples_dir = Path(__file__).parents[1] / "examples"
    bgen_file = examples_dir / "data" / "data.bgen"
    sample_file = examples_dir / "data" / "data.sample"
    ref_bcor_file = examples_dir / "data" / "data.bcor"

    return {
        "examples_dir": examples_dir,
        "bgen_file": bgen_file,
        "sample_file": sample_file,
        "ref_bcor_file": ref_bcor_file,
    }


# ==================== Matrix Format I/O Tests ====================


@pytest.mark.parametrize(
    "output_format,file_extension,test_loading",
    [("matrix", ".ld", True), ("long", ".ld.gz", False), ("bcor", ".bcor", False)],
)
def test_save_correlation_matrix_formats(tmp_path, output_format, file_extension, test_loading):
    """Test saving correlation matrices in different formats."""
    # Create test correlation matrix
    n_vars = 10
    test_matrix = np.random.rand(n_vars, n_vars)
    test_matrix = (test_matrix + test_matrix.T) / 2  # Make symmetric
    np.fill_diagonal(test_matrix, 1.0)

    variant_info = pd.DataFrame(
        {
            "rsid": [f"var_{i}" for i in range(n_vars)],
            "chrom": ["01"] * n_vars,
            "pos": range(1000, 1000 + n_vars),
            "ref": ["A"] * n_vars,
            "alt": ["G"] * n_vars,
        }
    )

    # Test the specific format
    output_file = tmp_path / f"test{file_extension}"
    save_correlation_matrix(
        test_matrix, str(output_file), variant_info=variant_info, output_format=output_format
    )
    assert os.path.exists(output_file)

    # Test loading only for formats that support it
    if test_loading:
        loaded_matrix, loaded_variant_info = load_correlation_matrix(str(output_file))
        np.testing.assert_array_almost_equal(test_matrix, loaded_matrix)


# ==================== BCOR Format Tests ====================


def test_bcor_export_matches_reference(test_data, tmp_path):
    """Test that exported bcor matches reference bcor."""
    # Load data and compute LD
    standardized_genotypes, variant_info, sample_ids, means, norms = load_and_adjust_genotypes(
        genotype_file=str(test_data["bgen_file"]),
        sample_file=str(test_data["sample_file"]),
        covariate_file=None,
    )

    ld_matrix = compute_correlation_matrix(standardized_genotypes)

    # Load reference bcor
    ref_bcor = BcorReader(str(test_data["ref_bcor_file"]))
    ref_bcor_matrix = ref_bcor.read_corr([], [])
    ref_meta = ref_bcor.get_meta()

    # Export our LD as bcor
    output_file = tmp_path / "test_bcor_export.bcor"

    # Convert variant info to match bcor format
    variant_info_bcor = pd.DataFrame(
        {
            "rsid": variant_info["rsid"].tolist(),
            "chrom": variant_info["chrom"].tolist(),
            "pos": variant_info["pos"].tolist(),
            "ref": variant_info["ref"].tolist(),
            "alt": variant_info["alt"].tolist(),
        }
    )

    save_correlation_matrix(
        corr_matrix=ld_matrix,
        output_file=str(output_file),
        variant_info=variant_info_bcor,
        output_format="bcor",
        n_samples=ref_bcor.get_n_samples(),
        compression=1,
    )

    # Read back our bcor
    our_bcor = BcorReader(str(output_file))
    our_bcor_matrix = our_bcor.read_corr([], [])
    our_meta = our_bcor.get_meta()

    # Compare matrices
    assert our_bcor_matrix.shape == ref_bcor_matrix.shape

    matrix_diff = np.abs(our_bcor_matrix - ref_bcor_matrix)
    max_matrix_diff = np.max(matrix_diff)
    mean_matrix_diff = np.mean(matrix_diff)

    assert max_matrix_diff < 1e-6, f"BCOR matrix differs from reference: max_diff={max_matrix_diff}"
    assert (
        mean_matrix_diff < 1e-8
    ), f"BCOR matrix differs from reference: mean_diff={mean_matrix_diff}"

    # Compare metadata
    assert len(our_meta) == len(ref_meta), "Metadata length mismatch"
    assert our_bcor.get_n_samples() == ref_bcor.get_n_samples(), "Sample count mismatch"
    assert our_bcor.get_n_snps() == ref_bcor.get_n_snps(), "SNP count mismatch"


@pytest.mark.parametrize(
    "compression,tolerance",
    [
        (0, 1e-4),  # 2 bytes
        (1, 1e-6),  # 4 bytes
        (2, 1e-7),  # 8 bytes
        (3, 2e-2),  # 1 byte
    ],
)
def test_bcor_export_compression_levels(tmp_path, compression, tolerance):
    """Test bcor export with different compression levels."""
    # Create a smaller test matrix
    n_vars = 10
    test_matrix = np.random.rand(n_vars, n_vars)
    test_matrix = (test_matrix + test_matrix.T) / 2  # Make symmetric
    np.fill_diagonal(test_matrix, 1.0)
    # Ensure values are in [-1, 1] range
    test_matrix = np.clip(test_matrix * 2 - 1, -1, 1)
    np.fill_diagonal(test_matrix, 1.0)

    variant_info = pd.DataFrame(
        {
            "rsid": [f"rs{i}" for i in range(n_vars)],
            "chrom": ["01"] * n_vars,
            "pos": list(range(1, n_vars + 1)),
            "ref": ["A"] * n_vars,
            "alt": ["G"] * n_vars,
        }
    )

    output_file = tmp_path / f"test_compression_{compression}.bcor"

    save_correlation_matrix(
        corr_matrix=test_matrix,
        output_file=str(output_file),
        variant_info=variant_info,
        output_format="bcor",
        n_samples=1000,
        compression=compression,
    )

    # Read back and verify
    reader = BcorReader(str(output_file))
    read_matrix = reader.read_corr([], [])

    assert read_matrix.shape == test_matrix.shape

    # Check that values are reasonably close (precision depends on compression)
    diff = np.abs(read_matrix - test_matrix)
    max_diff = np.max(diff)

    assert max_diff < tolerance, f"Compression {compression}: max_diff={max_diff} > {tolerance}"


def test_bcor_export_without_ldstore(tmp_path):
    """Test that bcor export works even without ldstore for reading back."""
    # Create small test matrix
    n_vars = 5
    test_matrix = np.eye(n_vars)

    variant_info = pd.DataFrame(
        {
            "rsid": [f"var_{i}" for i in range(n_vars)],
            "chrom": ["1"] * n_vars,
            "pos": list(range(100, 100 + n_vars)),
            "ref": ["A"] * n_vars,
            "alt": ["T"] * n_vars,
        }
    )

    output_file = tmp_path / "test_no_ldstore.bcor"

    # Should not raise any errors
    save_correlation_matrix(
        corr_matrix=test_matrix,
        output_file=str(output_file),
        variant_info=variant_info,
        output_format="bcor",
        n_samples=1000,
        compression=1,
    )

    # File should exist and have reasonable size
    assert os.path.exists(output_file)
    file_size = os.path.getsize(output_file)
    assert file_size > 100  # Should be at least a few hundred bytes


# ==================== Extended BCOR Format Tests ====================


def test_standard_bcor_with_unit_diagonal(tmp_path):
    """Test that standard bcor is written when diagonal is all 1s."""
    n_vars = 10
    # Create correlation matrix with unit diagonal
    corr_matrix = np.random.rand(n_vars, n_vars) * 0.8
    corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Make symmetric
    np.fill_diagonal(corr_matrix, 1.0)  # Unit diagonal

    variant_info = pd.DataFrame(
        {
            "rsid": [f"rs{i}" for i in range(n_vars)],
            "chrom": ["1"] * n_vars,
            "pos": list(range(1000, 1000 + n_vars * 100, 100)),
            "ref": ["A"] * n_vars,
            "alt": ["G"] * n_vars,
        }
    )

    output_file = tmp_path / "standard.bcor"
    save_bcor(corr_matrix, str(output_file), variant_info, n_samples=100, compression=1)

    # Read back and verify
    reader = BcorReader(str(output_file))
    assert not reader.is_extended, "Should be standard format"

    # Check magic string
    with open(output_file, "rb") as f:
        magic = f.read(7)
        assert magic == b"bcor1.1", "Should have standard magic string"

    # Verify matrix
    loaded = reader.read_corr()
    np.testing.assert_array_almost_equal(corr_matrix, loaded, decimal=5)


def test_extended_bcor_with_non_unit_diagonal(tmp_path):
    """Test that extended bcor is written when diagonal is not all 1s."""
    n_vars = 10
    # Create correlation matrix with non-unit diagonal (adjusted LD)
    corr_matrix = np.random.rand(n_vars, n_vars) * 0.8
    corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Make symmetric
    # Set non-unit diagonal values
    diagonal_values = np.array([0.9, 1.0, 0.85, 1.0, 0.95, 1.0, 0.88, 1.0, 0.92, 1.0])
    np.fill_diagonal(corr_matrix, diagonal_values)

    variant_info = pd.DataFrame(
        {
            "rsid": [f"rs{i}" for i in range(n_vars)],
            "chrom": ["1"] * n_vars,
            "pos": list(range(1000, 1000 + n_vars * 100, 100)),
            "ref": ["A"] * n_vars,
            "alt": ["G"] * n_vars,
        }
    )

    output_file = tmp_path / "extended.bcor"
    save_bcor(corr_matrix, str(output_file), variant_info, n_samples=100, compression=1)

    # Read back and verify
    reader = BcorReader(str(output_file))
    assert reader.is_extended, "Should be extended format"

    # Check magic string
    with open(output_file, "rb") as f:
        magic = f.read(7)
        assert magic == b"bcor1.x", "Should have extended magic string"

    # Verify matrix including diagonal
    loaded = reader.read_corr()
    np.testing.assert_array_almost_equal(corr_matrix, loaded, decimal=5)

    # Specifically check diagonal values
    np.testing.assert_array_almost_equal(np.diag(loaded), diagonal_values, decimal=5)


@pytest.mark.parametrize(
    "compression,tolerance",
    [
        (0, 1e-4),  # 2 bytes
        (1, 1e-6),  # 4 bytes
        (2, 1e-7),  # 8 bytes
        (3, 0.02),  # 1 byte
    ],
)
def test_extended_bcor_compression_levels(tmp_path, compression, tolerance):
    """Test extended bcor with different compression levels."""
    n_vars = 8
    # Create correlation matrix with non-unit diagonal
    corr_matrix = np.eye(n_vars)
    # Add some off-diagonal correlations
    corr_matrix[0, 1] = corr_matrix[1, 0] = 0.8
    corr_matrix[2, 3] = corr_matrix[3, 2] = 0.6
    # Non-unit diagonal
    diagonal_values = np.array([0.95, 0.90, 1.0, 0.85, 1.0, 0.92, 0.88, 1.0])
    np.fill_diagonal(corr_matrix, diagonal_values)

    variant_info = pd.DataFrame(
        {
            "rsid": [f"var{i}" for i in range(n_vars)],
            "chrom": ["1"] * n_vars,
            "pos": list(range(100, 100 + n_vars)),
            "ref": ["A"] * n_vars,
            "alt": ["T"] * n_vars,
        }
    )

    output_file = tmp_path / f"extended_comp{compression}.bcor"

    writer = BcorWriter(str(output_file), n_samples=100, compression=compression)
    writer.write(corr_matrix, variant_info)

    # Read back
    reader = BcorReader(str(output_file))
    assert reader.is_extended
    loaded = reader.read_corr()

    # Check values with appropriate tolerance
    diff = np.abs(loaded - corr_matrix)
    max_diff = np.max(diff)
    assert max_diff < tolerance, f"Compression {compression}: max_diff={max_diff} > {tolerance}"


def test_extended_bcor_subset_reads(tmp_path):
    """Test reading subsets from extended bcor files."""
    n_vars = 20
    # Create test matrix with non-unit diagonal
    corr_matrix = np.random.rand(n_vars, n_vars) * 0.5 + 0.3
    corr_matrix = (corr_matrix + corr_matrix.T) / 2
    # Variable diagonal values
    diagonal_values = 0.8 + 0.2 * np.random.rand(n_vars)
    np.fill_diagonal(corr_matrix, diagonal_values)

    output_file = tmp_path / "extended_subset.bcor"
    save_bcor(corr_matrix, str(output_file), n_samples=100)

    reader = BcorReader(str(output_file))
    assert reader.is_extended

    # Test reading specific rows
    rows = [0, 5, 10, 15]
    subset = reader.read_corr(rows)
    expected = corr_matrix[:, rows]
    np.testing.assert_array_almost_equal(subset, expected, decimal=5)

    # Test reading specific pairs
    rows = [1, 3, 5]
    cols = [2, 4, 6, 8]
    subset = reader.read_corr(rows, cols)
    expected = corr_matrix[np.ix_(rows, cols)]
    np.testing.assert_array_almost_equal(subset, expected, decimal=5)

    # Test diagonal element reads
    for i in range(n_vars):
        diag_val = reader.read_corr([i], [i])[0, 0]
        assert abs(diag_val - diagonal_values[i]) < 1e-5


def test_bcor_backward_compatibility(tmp_path):
    """Test that standard bcor files can still be read correctly."""
    n_vars = 5
    # Create standard correlation matrix (unit diagonal)
    corr_matrix = np.eye(n_vars)
    corr_matrix[0, 1] = corr_matrix[1, 0] = 0.5

    output_file = tmp_path / "standard_compat.bcor"
    save_bcor(corr_matrix, str(output_file), n_samples=100)

    reader = BcorReader(str(output_file))
    assert not reader.is_extended

    # Should read correctly with diagonal as 1.0
    loaded = reader.read_corr()
    np.testing.assert_array_almost_equal(corr_matrix, loaded, decimal=6)
    np.testing.assert_array_equal(np.diag(loaded), np.ones(n_vars))


# ==================== Error Handling Tests ====================


def test_save_correlation_matrix_invalid_format(tmp_path):
    """Test error handling for invalid output format."""
    test_matrix = np.eye(3)
    output_file = tmp_path / "test.invalid"

    with pytest.raises(ValueError, match="Unsupported output format"):
        save_correlation_matrix(test_matrix, str(output_file), output_format="invalid")


def test_save_correlation_matrix_invalid_compression(tmp_path):
    """Test error handling for invalid compression level."""
    test_matrix = np.eye(3)
    output_file = tmp_path / "test.bcor"

    with pytest.raises(ValueError, match="Invalid compression level"):
        save_correlation_matrix(test_matrix, str(output_file), output_format="bcor", compression=5)


def test_bcor_reader_invalid_file(tmp_path):
    """Test error handling for invalid BCOR file."""
    # Create invalid file
    invalid_file = tmp_path / "invalid.bcor"
    with open(invalid_file, "wb") as f:
        f.write(b"not a bcor file")

    with pytest.raises(ValueError, match="is not a valid bcor file"):
        BcorReader(str(invalid_file))


def test_bcor_reader_nonexistent_file():
    """Test error handling for non-existent BCOR file."""
    with pytest.raises(FileNotFoundError):
        BcorReader("/path/to/nonexistent/file.bcor")


def test_save_bcor_emits_index_sidecar_by_default(tmp_path):
    n = 8
    corr = np.eye(n) + 0.1
    np.fill_diagonal(corr, 1.0)
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1.0)

    variant_info = pd.DataFrame({
        "rsid": [f"rs{1000 - i}" for i in range(n)],  # rsids descend; positions still ascend
        "chrom": ["1"] * n,
        "pos": list(range(1, n + 1)),
        "ref": ["A"] * n,
        "alt": ["G"] * n,
    })

    out = tmp_path / "with_idx.bcor"
    save_bcor(corr, str(out), variant_info=variant_info, n_samples=100)

    idx_path = str(out) + ".idx"
    assert os.path.exists(idx_path)

    with open(idx_path, "rb") as fh:
        idx = BcorIndexReader.from_stream(fh, size=os.path.getsize(idx_path))

    assert idx.n_snps == n
    for i, rsid in enumerate(variant_info["rsid"]):
        assert idx.rsid_to_row(rsid) == i


def test_save_bcor_skips_index_when_write_index_false(tmp_path):
    n = 3
    corr = np.eye(n)
    out = tmp_path / "no_idx.bcor"
    save_bcor(corr, str(out), n_samples=10, write_index=False)
    assert os.path.exists(out)
    assert not os.path.exists(str(out) + ".idx")


def test_bcor_idx_meta_offsets_point_to_real_records(tmp_path):
    """Byte ranges from the sidecar should slice out valid meta records in the .bcor."""
    n = 5
    corr = np.eye(n)
    variant_info = pd.DataFrame({
        "rsid": [f"variant_{i}" for i in range(n)],
        "chrom": [str((i % 22) + 1) for i in range(n)],
        "pos": list(range(100, 100 + n)),
        "ref": ["A"] * n,
        "alt": ["T"] * n,
    })

    out = tmp_path / "check.bcor"
    save_bcor(corr, str(out), variant_info=variant_info, n_samples=42)

    with open(str(out) + ".idx", "rb") as fh:
        idx = BcorIndexReader.from_stream(fh)

    with open(out, "rb") as fh:
        for i in range(n):
            start, length = idx.meta_byte_range(i)
            fh.seek(start)
            rec = fh.read(length)
            # First 4 bytes = L_buffer; next 4 = index; next 2 = L_rsid; next L_rsid = rsid.
            L_buffer = int.from_bytes(rec[0:4], "little")
            # L_buffer does not include the 4 bytes of L_buffer itself.
            assert len(rec) == L_buffer + 4
            rsid_len = int.from_bytes(rec[8:10], "little")
            rsid_bytes = rec[10 : 10 + rsid_len]
            assert rsid_bytes.decode("utf-8") == variant_info["rsid"].iloc[i]


# ==================== Example data sidecar regression test ====================


def test_example_data_sidecar_round_trip(test_data):
    """The example .bcor + .bcor.idx pair under examples/data/ must load and partial-read
    correctly. Catches regressions in the sidecar format or generator script."""
    bcor_path = test_data["ref_bcor_file"]
    idx_path = Path(str(bcor_path) + ".idx")
    if not idx_path.exists():
        pytest.skip(
            "examples/data/data.bcor.idx not present; "
            "regenerate with: python scripts/make_bcor_idx.py examples/data/data.bcor"
        )

    reader = BcorReader(str(bcor_path))
    assert reader.has_index, "sidecar must auto-load"
    assert reader.index.n_snps == reader.n_snps

    # Round-trip: partial read by rsid must match the full-matrix read.
    full = reader.read_corr()
    rsids = ["rs1", "rs5", "rs25", "rs55"]
    subset, subset_meta = reader.read_corr_by_rsid(rsids)
    assert list(subset_meta["rsid"]) == rsids

    meta = reader.get_meta()
    rows = [int(meta.index[meta["rsid"] == r][0]) for r in rsids]
    expected = full[np.ix_(rows, rows)]
    np.testing.assert_array_equal(subset, expected)
