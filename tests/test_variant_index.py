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
    idx, flip = vi.match("1", 100, "G", "A")  # ref/alt swapped vs variant_index
    assert idx == 0 and flip is True


def test_match_missing(variant_index):
    vi = VariantIndex(variant_index)
    assert vi.match("1", 100, "A", "T") == (None, None)


def test_by_idx_range(variant_index):
    vi = VariantIndex(variant_index)
    out = vi.by_idx_range(1, 3)
    assert list(out["idx"]) == [1, 2]
