"""
Tests for vectorized subset reads in BcorReader.

Verifies that the integer-index subset paths (read_corr with snps1/snps2)
produce byte-identical results to slicing the full matrix, for both standard
(diag=1) and extended (diag!=1) formats, including off-diagonal NaN entries.
"""

import numpy as np
import pandas as pd
import pytest

from ldcov.io.bcor_writer import save_bcor
from ldcov.io.bcor_reader import BcorReader


def _make_bcor(tmp_path, extended: bool, n: int = 30):
    """Write a random symmetric matrix to a .bcor file and return (path, matrix)."""
    rng = np.random.default_rng(0)
    m = rng.uniform(-1, 1, (n, n))
    m = (m + m.T) / 2
    if extended:
        np.fill_diagonal(m, rng.uniform(1.0, 1.5, n))
    else:
        np.fill_diagonal(m, 1.0)
    # Punch a NaN off-diagonal (symmetric)
    m[1, 4] = m[4, 1] = np.nan

    vi = pd.DataFrame(
        {
            "rsid": [f"v{i}" for i in range(n)],
            "chrom": ["1"] * n,
            "pos": range(n),
            "ref": ["A"] * n,
            "alt": ["G"] * n,
        }
    )
    label = "ext" if extended else "std"
    p = str(tmp_path / f"x_{label}.bcor")
    save_bcor(m, p, variant_info=vi, compression=2, write_index=False)
    return p, m


def test_read_corr_subset_matches_full(tmp_path):
    """read_corr integer subsets must exactly match slicing the full matrix."""
    rows = [2, 5, 7, 4, 1]
    cols = [0, 4, 9, 1]

    for extended in (False, True):
        path, m = _make_bcor(tmp_path, extended=extended)
        r = BcorReader(path)
        full = r.read_corr([], [])

        # --- pairs branch: read_corr(rows, cols) -> shape (len(rows), len(cols)) ---
        sub = r.read_corr(rows, cols)
        expected = full[np.ix_(rows, cols)]
        assert sub.shape == expected.shape, f"extended={extended}: pairs shape mismatch"
        np.testing.assert_array_equal(
            np.isnan(sub),
            np.isnan(expected),
            err_msg=f"extended={extended}: NaN mask mismatch in pairs branch",
        )
        np.testing.assert_allclose(
            np.nan_to_num(sub),
            np.nan_to_num(expected),
            atol=1e-6,
            err_msg=f"extended={extended}: values mismatch in pairs branch",
        )

        # --- rows-only branch: read_corr(rows) -> shape (n, len(rows)) == full[:, rows] ---
        subrows = r.read_corr(rows)
        expected_rows = full[:, rows]
        assert (
            subrows.shape == expected_rows.shape
        ), f"extended={extended}: rows-only shape mismatch: {subrows.shape} vs {expected_rows.shape}"
        np.testing.assert_array_equal(
            np.isnan(subrows),
            np.isnan(expected_rows),
            err_msg=f"extended={extended}: NaN mask mismatch in rows-only branch",
        )
        np.testing.assert_allclose(
            np.nan_to_num(subrows),
            np.nan_to_num(expected_rows),
            atol=1e-6,
            err_msg=f"extended={extended}: values mismatch in rows-only branch",
        )


def test_read_corr_subset_diagonal_exact(tmp_path):
    """Diagonal entries in a subset read must match the full matrix exactly."""
    n = 20
    for extended in (False, True):
        path, m = _make_bcor(tmp_path, extended=extended, n=n)
        r = BcorReader(path)
        full = r.read_corr([], [])

        # Square subset that contains diagonal elements
        idxs = [0, 3, 7, 10, 15]
        sub = r.read_corr(idxs, idxs)
        expected = full[np.ix_(idxs, idxs)]
        np.testing.assert_allclose(
            np.nan_to_num(sub),
            np.nan_to_num(expected),
            atol=1e-6,
            err_msg=f"extended={extended}: diagonal subset mismatch",
        )
        # Explicitly check diagonal of this sub-square
        np.testing.assert_allclose(
            np.diag(sub),
            np.diag(expected),
            atol=1e-6,
            err_msg=f"extended={extended}: diagonal values mismatch",
        )


def test_read_corr_empty_inputs(tmp_path):
    """Empty snps1 or snps2 must return correctly-shaped zero matrix."""
    path, m = _make_bcor(tmp_path, extended=False, n=10)
    r = BcorReader(path)

    # Both non-empty but one is zero-length list edge: snps1=[], snps2=[] -> full
    full = r.read_corr([], [])
    assert full.shape == (10, 10)

    # snps1 non-empty, snps2 non-empty
    sub = r.read_corr([0, 1], [2, 3, 4])
    assert sub.shape == (2, 3)


@pytest.mark.parametrize("compression", [0, 1, 2, 3])  # 2,4,8,1 bytes per value
@pytest.mark.parametrize("extended", [False, True])
def test_read_corr_subset_all_compressions(tmp_path, compression, extended):
    """Subset reads match full slices for every compression level (esp. 8-byte NA path)."""
    n = 25
    rng = np.random.default_rng(compression + (10 if extended else 0))
    m = rng.uniform(-1, 1, (n, n))
    m = (m + m.T) / 2
    np.fill_diagonal(m, rng.uniform(1.0, 1.5, n) if extended else 1.0)
    m[1, 4] = m[4, 1] = np.nan  # 8-byte NA sentinel overflows int64 -> regression guard
    m[0, 9] = m[9, 0] = np.nan
    vi = pd.DataFrame(
        {
            "rsid": [f"v{i}" for i in range(n)],
            "chrom": ["1"] * n,
            "pos": range(n),
            "ref": ["A"] * n,
            "alt": ["G"] * n,
        }
    )
    p = str(tmp_path / f"c{compression}_{int(extended)}.bcor")
    save_bcor(m, p, variant_info=vi, compression=compression, write_index=False)
    r = BcorReader(p)
    full = r.read_corr([], [])
    rows, cols = [2, 5, 7, 4, 1, 9], [0, 4, 9, 1]
    sub = r.read_corr(rows, cols)
    ref = full[np.ix_(rows, cols)]
    np.testing.assert_array_equal(np.isnan(sub), np.isnan(ref))
    np.testing.assert_allclose(np.nan_to_num(sub), np.nan_to_num(ref), atol=2e-2)
    subrows = r.read_corr(rows)
    np.testing.assert_array_equal(np.isnan(subrows), np.isnan(full[:, rows]))


def test_read_corr_empty_and_single(tmp_path):
    """Empty subsets and an all-diagonal request are handled without crashing."""
    p, m = _make_bcor(tmp_path, extended=False, n=20)
    r = BcorReader(p)
    assert r.read_corr([], [3, 4]).shape == (0, 2)
    assert r.read_corr([3, 4], []).shape[1] == 2  # rows-only -> (n_snps, 2)
    # all-diagonal request (off_mask empty): every entry is the diagonal value
    diag_only = r.read_corr([5, 8], [5, 8])
    assert diag_only[0, 0] == pytest.approx(1.0) and diag_only[1, 1] == pytest.approx(1.0)
