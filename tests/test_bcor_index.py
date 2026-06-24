import os

import numpy as np
import pandas as pd
import pytest

from ldcov.io.bcor_index import (
    BCOR_IDX_MAGIC,
    BCOR_IDX_VERSION,
    BCOR_IDX_HEADER_SIZE,
    encode_idx_header,
    decode_idx_header,
    BcorIndexWriter,
    BcorIndexReader,
)


def test_magic_and_version():
    assert BCOR_IDX_MAGIC == b"bcoridx1"
    assert BCOR_IDX_VERSION == 1
    # 8 magic + 4 version + 4 n_snps + 4 flags + 4 bcor_meta_start + 8 bcor_file_size +
    # 8 bcor_corr_block_offset = 40
    assert BCOR_IDX_HEADER_SIZE == 40


def test_header_roundtrip():
    payload = encode_idx_header(
        n_snps=1234,
        bcor_meta_start=32,
        bcor_file_size=10_000_000,
        bcor_corr_block_offset=4096,
    )
    assert len(payload) == BCOR_IDX_HEADER_SIZE
    decoded = decode_idx_header(payload)
    assert decoded["n_snps"] == 1234
    assert decoded["flags"] == 0
    assert decoded["bcor_meta_start"] == 32
    assert decoded["bcor_file_size"] == 10_000_000
    assert decoded["bcor_corr_block_offset"] == 4096


def test_header_bad_magic():
    payload = b"notidx!!" + b"\x00" * 32
    with pytest.raises(ValueError, match="not a valid bcor index"):
        decode_idx_header(payload)


def test_header_bad_version():
    payload = (
        BCOR_IDX_MAGIC
        + (99).to_bytes(4, "little")
        + (0).to_bytes(4, "little")  # n_snps
        + (0).to_bytes(4, "little")  # flags
        + (0).to_bytes(4, "little")  # bcor_meta_start
        + (0).to_bytes(8, "little")  # bcor_file_size
        + (0).to_bytes(8, "little")  # bcor_corr_block_offset
    )
    with pytest.raises(ValueError, match="unsupported bcor index version"):
        decode_idx_header(payload)


# ---------------------------------------------------------------------------
# Task 2: BcorIndexWriter tests
# ---------------------------------------------------------------------------


def _make_variant_info(n: int) -> pd.DataFrame:
    # Unique rsids, in position-sorted (row) order.
    rsids = [f"rs{(i * 17) % 9973}" for i in range(n)]
    assert len(set(rsids)) == n, "test fixture must produce unique rsids"
    return pd.DataFrame(
        {
            "rsid": rsids,
            "chrom": ["1"] * n,
            "pos": list(range(1000, 1000 + n)),  # already sorted
            "ref": ["A"] * n,
            "alt": ["G"] * n,
        }
    )


def _write_args(meta_offsets, bcor_meta_start=32):
    """Build self-consistent default args from meta_offsets endpoints."""
    return {
        "meta_record_offsets": meta_offsets,
        "bcor_meta_start": bcor_meta_start,
        "bcor_corr_block_offset": int(meta_offsets[-1]),
        "bcor_file_size": int(meta_offsets[-1]) + 1000,  # arbitrary > corr_block_offset
    }


def test_index_writer_produces_expected_layout(tmp_path):
    n = 10
    variant_info = _make_variant_info(n)
    offsets = np.arange(32, 32 + 20 * (n + 1), 20, dtype=np.uint64)
    assert offsets.shape == (n + 1,)

    out = tmp_path / "foo.bcor.idx"
    writer = BcorIndexWriter(str(out))
    writer.write(variant_info, **_write_args(offsets))

    assert out.exists()

    buf = out.read_bytes()
    assert buf[:8] == b"bcoridx1"
    version = int.from_bytes(buf[8:12], "little")
    n_snps_hdr = int.from_bytes(buf[12:16], "little")
    assert version == 1
    assert n_snps_hdr == n

    # Header(40) + rsid_offsets(4·(n+1)) + meta_offsets(8·(n+1)) + rsid_block.
    expected_min = 40 + 4 * (n + 1) + 8 * (n + 1)
    assert len(buf) == expected_min + sum(len(r) for r in variant_info["rsid"])


def test_index_writer_rejects_bad_offsets(tmp_path):
    variant_info = _make_variant_info(5)
    writer = BcorIndexWriter(str(tmp_path / "x.bcor.idx"))

    bad = np.zeros(5, dtype=np.uint64)  # should be n+1 = 6
    with pytest.raises(ValueError, match="meta_record_offsets must have length n\\+1"):
        writer.write(variant_info, **_write_args(bad))

    bad2 = np.array([32, 50, 45, 60, 70, 80], dtype=np.uint64)
    with pytest.raises(
        ValueError, match="meta_record_offsets must be monotonically non-decreasing"
    ):
        writer.write(variant_info, **_write_args(bad2))


def test_index_writer_rejects_meta_offset_endpoints(tmp_path):
    """meta_record_offsets[0] must equal bcor_meta_start; [-1] must equal bcor_corr_block_offset."""
    n = 4
    variant_info = _make_variant_info(n)
    writer = BcorIndexWriter(str(tmp_path / "e.bcor.idx"))

    # Bad first offset.
    bad_first = np.array([0, 50, 70, 90, 110], dtype=np.uint64)
    with pytest.raises(ValueError, match="meta_record_offsets\\[0\\] must equal bcor_meta_start"):
        writer.write(variant_info, **_write_args(bad_first))

    # Bad last offset (mismatch between meta_record_offsets[-1] and bcor_corr_block_offset).
    base_offsets = np.array([32 + i * 20 for i in range(n + 1)], dtype=np.uint64)
    base_offsets[-1] = base_offsets[-1] + 1
    args = _write_args(base_offsets)
    args["bcor_corr_block_offset"] = int(base_offsets[-1]) - 1  # disagrees with offsets[-1]
    with pytest.raises(
        ValueError, match="meta_record_offsets\\[-1\\] must equal bcor_corr_block_offset"
    ):
        writer.write(variant_info, **args)


def test_index_writer_rejects_duplicate_rsids(tmp_path):
    """Partial read by rsid requires unique rsid → row mapping. Reject duplicates."""
    n = 4
    variant_info = pd.DataFrame(
        {
            "rsid": ["rsA", "rsB", "rsA", "rsC"],  # rsA duplicated
            "chrom": ["1"] * n,
            "pos": list(range(1, n + 1)),
            "ref": ["A"] * n,
            "alt": ["G"] * n,
        }
    )
    offsets = np.arange(32, 32 + 20 * (n + 1), 20, dtype=np.uint64)

    writer = BcorIndexWriter(str(tmp_path / "dup.bcor.idx"))
    with pytest.raises(ValueError, match="duplicate rsids"):
        writer.write(variant_info, **_write_args(offsets))


def test_index_writer_empty_input_requires_consistent_endpoints(tmp_path):
    """For n=0, meta_record_offsets has one slot that must equal both bcor_meta_start
    AND bcor_corr_block_offset. The writer must reject inconsistent endpoints."""
    empty_vi = pd.DataFrame({"rsid": [], "chrom": [], "pos": [], "ref": [], "alt": []})

    # Inconsistent: bcor_meta_start (32) != bcor_corr_block_offset (40).
    bad_offsets = np.array([32], dtype=np.uint64)
    writer = BcorIndexWriter(str(tmp_path / "bad_empty.bcor.idx"))
    with pytest.raises(
        ValueError, match="meta_record_offsets\\[-1\\] must equal bcor_corr_block_offset"
    ):
        writer.write(
            empty_vi,
            meta_record_offsets=bad_offsets,
            bcor_meta_start=32,
            bcor_file_size=32,
            bcor_corr_block_offset=40,  # disagrees with offsets[-1]
        )

    # Consistent: empty bcor where meta_start == corr_block_offset (no meta records).
    good_offsets = np.array([32], dtype=np.uint64)
    BcorIndexWriter(str(tmp_path / "ok_empty.bcor.idx")).write(
        empty_vi,
        meta_record_offsets=good_offsets,
        bcor_meta_start=32,
        bcor_file_size=32,
        bcor_corr_block_offset=32,
    )
    assert (tmp_path / "ok_empty.bcor.idx").exists()


def test_index_writer_keeps_row_order(tmp_path):
    """rsids must be stored in row order; rsid_offsets reflects per-row lengths."""
    n = 3
    variant_info = pd.DataFrame(
        {
            "rsid": ["ccc", "a", "bb"],  # row order, NOT sorted
            "chrom": ["1"] * n,
            "pos": list(range(1, n + 1)),
            "ref": ["A"] * n,
            "alt": ["G"] * n,
        }
    )
    offsets = np.arange(32, 32 + 20 * (n + 1), 20, dtype=np.uint64)
    writer = BcorIndexWriter(str(tmp_path / "row.bcor.idx"))
    writer.write(variant_info, **_write_args(offsets))

    buf = (tmp_path / "row.bcor.idx").read_bytes()
    rsid_off = np.frombuffer(buf[40 : 40 + 4 * (n + 1)], dtype=np.uint32).copy()
    assert rsid_off[0] == 0
    # In row order: "ccc" (3), "a" (1), "bb" (2).
    assert list(np.diff(rsid_off)) == [3, 1, 2]
    rsid_block_start = 40 + 4 * (n + 1) + 8 * (n + 1)
    assert buf[rsid_block_start : rsid_block_start + rsid_off[-1]] == b"ccca" + b"bb"


def test_index_writer_requires_rsid_column(tmp_path):
    """variant_info missing the 'rsid' column must be rejected with a clear error."""
    no_rsid = pd.DataFrame(
        {
            "chrom": ["1", "1"],
            "pos": [100, 200],
            "ref": ["A", "A"],
            "alt": ["G", "T"],
        }
    )
    offsets = np.array([32, 52, 72], dtype=np.uint64)
    writer = BcorIndexWriter(str(tmp_path / "no_rsid.bcor.idx"))
    with pytest.raises(ValueError, match="must have an 'rsid' column"):
        writer.write(no_rsid, **_write_args(offsets))


# ---------------------------------------------------------------------------
# Task 3: BcorIndexReader tests
# ---------------------------------------------------------------------------


def test_index_reader_roundtrip(tmp_path):
    n = 20
    variant_info = _make_variant_info(n)
    offsets = np.arange(32, 32 + 20 * (n + 1), 20, dtype=np.uint64)

    path = tmp_path / "foo.bcor.idx"
    BcorIndexWriter(str(path)).write(
        variant_info,
        offsets,
        bcor_meta_start=32,
        bcor_file_size=int(offsets[-1]) + 5000,
        bcor_corr_block_offset=int(offsets[-1]),
    )

    with open(path, "rb") as fh:
        reader = BcorIndexReader.from_stream(fh, size=os.path.getsize(path))

    assert reader.n_snps == n
    assert reader.bcor_meta_start == 32
    assert reader.bcor_corr_block_offset == int(offsets[-1])
    assert reader.bcor_file_size == int(offsets[-1]) + 5000

    # Every rsid in the original order should resolve to its original row.
    for i, rsid in enumerate(variant_info["rsid"]):
        assert reader.rsid_to_row(rsid) == i

    # Missing rsid returns None.
    assert reader.rsid_to_row("rs_nonexistent_zzz") is None

    # Vectorized lookup.
    query = ["rs_bogus"] + list(variant_info["rsid"])
    rows = reader.rsids_to_rows(query)
    assert rows[0] == -1
    np.testing.assert_array_equal(rows[1:], np.arange(n, dtype=np.int64))

    # Meta byte range for row i = (offsets[i], offsets[i+1] - offsets[i]).
    for i in range(n):
        start, length = reader.meta_byte_range(i)
        assert start == int(offsets[i])
        assert length == int(offsets[i + 1] - offsets[i])


def _emit_valid_sidecar(tmp_path, name="ok.bcor.idx", n=5):
    variant_info = _make_variant_info(n)
    offsets = np.arange(32, 32 + 20 * (n + 1), 20, dtype=np.uint64)
    path = tmp_path / name
    BcorIndexWriter(str(path)).write(
        variant_info,
        offsets,
        bcor_meta_start=32,
        bcor_file_size=int(offsets[-1]) + 1000,
        bcor_corr_block_offset=int(offsets[-1]),
    )
    return path, n


def test_index_reader_rejects_malformed_sidecar(tmp_path):
    """Reject obviously malformed sidecars at load time."""
    path, n = _emit_valid_sidecar(tmp_path)
    raw = path.read_bytes()

    # 1. Truncated rsid block.
    bad_path = tmp_path / "trunc_bad.bcor.idx"
    bad_path.write_bytes(raw[: len(raw) - 1])
    with open(bad_path, "rb") as fh:
        with pytest.raises(ValueError, match="truncated"):
            BcorIndexReader.from_stream(fh, size=os.path.getsize(bad_path))

    # 2. rsid_offsets[0] != 0.
    bad2 = bytearray(raw)
    rsid_offsets_off = 40
    bad2[rsid_offsets_off : rsid_offsets_off + 4] = (1).to_bytes(4, "little")
    bad2_path = tmp_path / "shifted.bcor.idx"
    bad2_path.write_bytes(bytes(bad2))
    with open(bad2_path, "rb") as fh:
        with pytest.raises(ValueError, match="rsid_offsets\\[0\\] must be 0"):
            BcorIndexReader.from_stream(fh)

    # 3. meta_record_offsets[0] != bcor_meta_start.
    bad3 = bytearray(raw)
    meta_offsets_off = 40 + 4 * (n + 1)
    cur = int.from_bytes(raw[meta_offsets_off : meta_offsets_off + 8], "little")
    bad3[meta_offsets_off : meta_offsets_off + 8] = (cur + 1).to_bytes(8, "little")
    bad3_path = tmp_path / "metahead.bcor.idx"
    bad3_path.write_bytes(bytes(bad3))
    with open(bad3_path, "rb") as fh:
        with pytest.raises(ValueError, match="meta_record_offsets\\[0\\] .* bcor_meta_start"):
            BcorIndexReader.from_stream(fh)


def test_index_reader_rejects_invalid_utf8_in_rsid_block(tmp_path):
    """rsid block with bytes that aren't valid UTF-8 must fail load with ValueError."""
    path, n = _emit_valid_sidecar(tmp_path, name="badutf.bcor.idx")
    raw = bytearray(path.read_bytes())
    # rsid block start: 40 + 4*(n+1) + 8*(n+1).
    rsid_block_start = 40 + 4 * (n + 1) + 8 * (n + 1)
    raw[rsid_block_start] = 0xFF  # invalid UTF-8 leading byte standalone
    bad_path = tmp_path / "badutf_corrupt.bcor.idx"
    bad_path.write_bytes(bytes(raw))
    with open(bad_path, "rb") as fh:
        with pytest.raises(ValueError, match="invalid UTF-8"):
            BcorIndexReader.from_stream(fh)


def test_index_reader_empty_index(tmp_path):
    """n_snps == 0 must round-trip and return -1 / None for all lookups."""
    empty_vi = pd.DataFrame({"rsid": [], "chrom": [], "pos": [], "ref": [], "alt": []})
    offsets = np.array([32], dtype=np.uint64)  # n+1 = 1
    path = tmp_path / "empty.bcor.idx"
    BcorIndexWriter(str(path)).write(
        empty_vi,
        offsets,
        bcor_meta_start=32,
        bcor_file_size=32,
        bcor_corr_block_offset=32,
    )

    with open(path, "rb") as fh:
        reader = BcorIndexReader.from_stream(fh)
    assert reader.n_snps == 0
    assert reader.rsid_to_row("anything") is None

    rows = reader.rsids_to_rows(["a", "b", "c"])
    np.testing.assert_array_equal(rows, np.array([-1, -1, -1], dtype=np.int64))


def test_index_reader_rejects_duplicate_rsid_in_block(tmp_path):
    """Duplicate rsids in the loaded block must be rejected by the reader (defense in depth)."""
    n = 4
    variant_info = pd.DataFrame(
        {
            "rsid": ["rs1", "rs2", "rs3", "rs4"],
            "chrom": ["1"] * n,
            "pos": list(range(1, n + 1)),
            "ref": ["A"] * n,
            "alt": ["G"] * n,
        }
    )
    offsets = np.arange(32, 32 + 20 * (n + 1), 20, dtype=np.uint64)
    path = tmp_path / "dup.bcor.idx"
    BcorIndexWriter(str(path)).write(
        variant_info,
        offsets,
        bcor_meta_start=32,
        bcor_file_size=int(offsets[-1]) + 1000,
        bcor_corr_block_offset=int(offsets[-1]),
    )

    raw = bytearray(path.read_bytes())
    # rsid_block sits at 40 + 4*(n+1) + 8*(n+1). It's "rs1rs2rs3rs4".
    rsid_block_start = 40 + 4 * (n + 1) + 8 * (n + 1)
    # Overwrite "rs2" → "rs1" to inject a duplicate.
    raw[rsid_block_start + 3 : rsid_block_start + 6] = b"rs1"
    bad_path = tmp_path / "dup_bad.bcor.idx"
    bad_path.write_bytes(bytes(raw))
    with open(bad_path, "rb") as fh:
        with pytest.raises(ValueError, match="duplicate rsid"):
            BcorIndexReader.from_stream(fh)


def test_index_reader_rejects_unsupported_version(tmp_path):
    path, _ = _emit_valid_sidecar(tmp_path, name="future.bcor.idx", n=3)
    raw = bytearray(path.read_bytes())
    raw[8:12] = (99).to_bytes(4, "little")
    bad = tmp_path / "future_bad.bcor.idx"
    bad.write_bytes(bytes(raw))
    with open(bad, "rb") as fh:
        with pytest.raises(ValueError, match="unsupported bcor index version"):
            BcorIndexReader.from_stream(fh)
