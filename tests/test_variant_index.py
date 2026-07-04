import pandas as pd
import pytest
from ldcov.io.variant_index import VariantIndex


@pytest.fixture
def variant_index(tmp_path):
    df = pd.DataFrame(
        {
            "contig": ["1", "1", "1", "2"],
            "position": [100, 200, 300, 100],
            "ref": ["A", "C", "G", "T"],
            "alt": ["G", "T", "A", "C"],
            "idx": [0, 1, 2, 3],
            "AF": [0.1, 0.2, 0.3, 0.4],
        }
    )
    p = str(tmp_path / "variant_index.parquet")
    df.to_parquet(p, index=False)
    return p


def test_query_region(variant_index):
    vi = VariantIndex(variant_index)
    out = vi.query_region("1", 150, 300)
    assert list(out["idx"]) == [1, 2]
    assert list(out["position"]) == [200, 300]


def test_match_exact(variant_index):
    vi = VariantIndex(variant_index)
    idx, flip = vi.match("1", 100, "A", "G")
    assert idx == 0 and flip is False


def test_match_swapped(variant_index):
    vi = VariantIndex(variant_index)
    # Swap matching is opt-in.
    idx, flip = vi.match("1", 100, "G", "A", allow_allele_swap=True)  # swapped vs variant_index
    assert idx == 0 and flip is True
    # Off by default: a swapped-only variant is not found.
    assert vi.match("1", 100, "G", "A") == (None, None)


def test_match_missing(variant_index):
    vi = VariantIndex(variant_index)
    assert vi.match("1", 100, "A", "T") == (None, None)


def test_by_idx_range(variant_index):
    vi = VariantIndex(variant_index)
    out = vi.by_idx_range(1, 3)
    assert list(out["idx"]) == [1, 2]


def test_match_no_swap(variant_index):
    vi = VariantIndex(variant_index)
    # Exact orientation still matches.
    assert vi.match("1", 100, "A", "G", allow_allele_swap=False) == (0, False)
    # Swapped-only variant is not found when allow_allele_swap is disabled.
    assert vi.match("1", 100, "G", "A", allow_allele_swap=False) == (None, None)


def test_match_variants_batch(variant_index):
    vi = VariantIndex(variant_index)
    query = pd.DataFrame(
        {
            "contig": ["1", "1", "1", "2", "1"],
            "position": [100, 200, 300, 100, 100],
            "ref": [
                "A",
                "T",
                "G",
                "T",
                "A",
            ],  # row1 exact, row2 swapped, row3 exact, row4 exact, row5 absent
            "alt": ["G", "C", "A", "C", "T"],
        }
    )
    out = vi.match_variants(query, allow_allele_swap=True)
    assert list(out["idx"]) == [0, 1, 2, 3, -1]
    assert list(out["flip"]) == [False, True, False, False, False]
    # Alignment with the input index is preserved.
    assert list(out.index) == list(query.index)


def test_match_variants_no_swap(variant_index):
    vi = VariantIndex(variant_index)
    query = pd.DataFrame(
        {
            "contig": ["1", "1"],
            "position": [100, 200],
            "ref": ["A", "T"],  # exact, swapped
            "alt": ["G", "C"],
        }
    )
    out = vi.match_variants(query, allow_allele_swap=False)
    assert list(out["idx"]) == [0, -1]  # swapped row not matched
    assert not out["flip"].any()


def test_match_variants_requires_columns(variant_index):
    vi = VariantIndex(variant_index)
    with pytest.raises(ValueError, match="requires columns"):
        vi.match_variants(pd.DataFrame({"contig": ["1"], "position": [100]}))
