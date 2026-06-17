import numpy as np
import pandas as pd
import pytest
from ldcov.io.blockmatrix.reader import HailBlockMatrixReader
from ldcov.ld_bm.extract import _assemble, _apply_flips
from ldcov.ld_bm.extract import extract_ld
from ldcov.io.bcor_reader import BcorReader
from tests.helpers import make_symmetric_bm, write_synthetic_bm


def test_assemble_contiguous(tmp_path):
    reader, full = make_symmetric_bm(tmp_path)
    out = _assemble(reader, [0, 1, 2, 3, 4, 5])
    np.testing.assert_allclose(out, full)


def test_assemble_subset_and_symmetry(tmp_path):
    reader, full = make_symmetric_bm(tmp_path)
    idxs = [1, 4, 5]
    out = _assemble(reader, idxs)
    expected = full[np.ix_(idxs, idxs)]
    np.testing.assert_allclose(out, expected)
    np.testing.assert_allclose(out, out.T)


def test_assemble_offband_is_nan(tmp_path):
    # 6x6, block_size=3, but DROP block (0,1): cross-block pairs are off-band -> NaN.
    rng = np.random.default_rng(1)
    full = rng.standard_normal((6, 6))
    full = (full + full.T) / 2.0
    blocks = {(0, 0): np.triu(full[0:3, 0:3]).copy(), (1, 1): np.triu(full[3:6, 3:6]).copy()}
    write_synthetic_bm(str(tmp_path / "m.bm"), blocks, 6, 6, 3)
    reader = HailBlockMatrixReader(str(tmp_path / "m.bm"))
    out = _assemble(reader, [0, 5])
    assert np.isnan(out[0, 1]) and np.isnan(out[1, 0])
    assert out[0, 0] == pytest.approx(full[0, 0])


def test_apply_flips():
    m = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]])
    out = _apply_flips(m.copy(), [1])  # flip variant at position 1
    assert out[0, 1] == pytest.approx(-0.5)
    assert out[1, 2] == pytest.approx(-0.3)
    assert out[0, 2] == pytest.approx(0.2)  # neither flipped
    assert out[1, 1] == pytest.approx(1.0)  # double-flip on diagonal cancels


def _make_variant_index(tmp_path, n=6):
    df = pd.DataFrame(
        {
            "contig": ["1"] * n,
            "position": [100 + 10 * i for i in range(n)],
            "ref": ["A"] * n,
            "alt": ["G"] * n,
            "idx": list(range(n)),
        }
    )
    p = str(tmp_path / "variant_index.parquet")
    df.to_parquet(p, index=False)
    return p


def test_extract_ld_region_bcor_roundtrip(tmp_path):
    reader, full = make_symmetric_bm(tmp_path)
    variant_index = _make_variant_index(tmp_path, n=6)
    out_prefix = str(tmp_path / "out")
    matrix, variants = extract_ld(
        bm_path=str(tmp_path / "m.bm"),
        variant_index_path=variant_index,
        region="1:100-150",  # positions 100..150 -> idx 0..5
        out=out_prefix,
        output_format="bcor",
    )
    assert matrix.shape == (6, 6)
    assert list(variants["idx"]) == [0, 1, 2, 3, 4, 5]
    # bcor round-trip
    r = BcorReader(out_prefix + ".bcor")
    back = r.read_corr()
    np.testing.assert_allclose(back, full, atol=1e-4)


def test_extract_ld_zfile_order_and_flip(tmp_path):
    reader, full = make_symmetric_bm(tmp_path)
    variant_index = _make_variant_index(tmp_path, n=6)
    zpath = str(tmp_path / "in.z")
    # z-file: select idx 2 and 0; variant at idx0 has alleles swapped (ref=G, alt=A) -> flip.
    with open(zpath, "w") as fh:
        fh.write("rsid chromosome position allele1 allele2\n")
        fh.write("v2 1 120 A G\n")
        fh.write("v0 1 100 G A\n")
    matrix, variants = extract_ld(
        bm_path=str(tmp_path / "m.bm"),
        variant_index_path=variant_index,
        z=zpath,
        out=str(tmp_path / "z_out"),
        output_format="npz",
    )
    assert list(variants["idx"]) == [2, 0]  # preserves z order
    assert list(variants["flipped"]) == [False, True]
    # out[0,1] corresponds to (idx2, idx0) with idx0 flipped -> -full[2,0]
    assert matrix[0, 1] == pytest.approx(-full[2, 0], abs=1e-9)


def test_extract_ld_requires_one_selector(tmp_path):
    variant_index = _make_variant_index(tmp_path)
    with pytest.raises(ValueError):
        extract_ld(bm_path="x", variant_index_path=variant_index, out="o")


def test_extract_ld_idx_range_without_variant_index(tmp_path):
    reader, full = make_symmetric_bm(tmp_path)
    matrix, variants = extract_ld(
        bm_path=str(tmp_path / "m.bm"),
        variant_index_path=None,  # no variant_index -> placeholder metadata
        idx_range=(0, 3),
        out=str(tmp_path / "slice"),
        output_format="npz",
    )
    assert matrix.shape == (3, 3)
    assert list(variants["idx"]) == [0, 1, 2]
    assert list(variants["rsid"]) == ["idx_0", "idx_1", "idx_2"]
    np.testing.assert_allclose(matrix, full[0:3, 0:3])


def test_extract_ld_zfile_on_missing_drop(tmp_path):
    reader, full = make_symmetric_bm(tmp_path)
    variant_index = _make_variant_index(tmp_path, n=6)
    zpath = str(tmp_path / "nomatch.z")
    with open(zpath, "w") as fh:
        fh.write("rsid chromosome position allele1 allele2\n")
        fh.write("vX 1 999999 A G\n")  # position not in variant_index -> unmatched
    matrix, variants = extract_ld(
        bm_path=str(tmp_path / "m.bm"),
        variant_index_path=variant_index,
        z=zpath,
        out=str(tmp_path / "drop_out"),
        output_format="npz",
        on_missing="drop",
    )
    assert matrix.shape == (0, 0)
    assert len(variants) == 0


def test_extract_ld_zfile_duplicate_raises(tmp_path):
    reader, full = make_symmetric_bm(tmp_path)
    variant_index = _make_variant_index(tmp_path, n=6)
    zpath = str(tmp_path / "dup.z")
    with open(zpath, "w") as fh:
        fh.write("rsid chromosome position allele1 allele2\n")
        fh.write("v0 1 100 A G\n")
        fh.write("v0dup 1 100 A G\n")  # same locus+alleles -> same idx -> duplicate
    with pytest.raises(ValueError, match="duplicate"):
        extract_ld(
            bm_path=str(tmp_path / "m.bm"),
            variant_index_path=variant_index,
            z=zpath,
            out=str(tmp_path / "dup_out"),
            output_format="npz",
        )


def test_assemble_parallel_matches_serial(tmp_path):
    reader, full = make_symmetric_bm(tmp_path)
    serial = _assemble(reader, [0, 1, 2, 3, 4, 5], max_workers=1)
    parallel = _assemble(reader, [0, 1, 2, 3, 4, 5], max_workers=4)
    np.testing.assert_allclose(serial, full)
    np.testing.assert_allclose(parallel, full)
    np.testing.assert_array_equal(serial, parallel)


def test_as_slice_helper():
    from ldcov.ld_bm.extract import _as_slice

    assert _as_slice([3, 4, 5, 6]) == slice(3, 7)
    assert _as_slice([0]) == slice(0, 1)
    assert _as_slice([1, 4, 5]) is None  # gap -> not contiguous
    assert _as_slice([5, 4, 3]) is None  # descending -> not contiguous
    assert _as_slice([]) is None


def test_assemble_contiguous_and_shuffled_paths_agree(tmp_path):
    # The fast (contiguous slice) path and the general (np.ix_) path must agree.
    reader, full = make_symmetric_bm(tmp_path)
    contiguous = _assemble(reader, [0, 1, 2, 3, 4, 5])  # fast path
    order = [5, 0, 3, 1, 4, 2]  # non-contiguous -> general path
    shuffled = _assemble(reader, order)
    # Undo the permutation and compare to the contiguous assembly.
    inv = np.argsort(order)
    np.testing.assert_allclose(shuffled[np.ix_(inv, inv)], contiguous)
    np.testing.assert_allclose(contiguous, full)
