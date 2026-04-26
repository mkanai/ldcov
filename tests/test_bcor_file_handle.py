import numpy as np
import pytest
from ldcov.io.bcor_file_handle import BcorFileHandle


def test_local_handle_basic_io(tmp_path):
    data = bytes(range(256)) * 10  # 2560 bytes
    p = tmp_path / "blob.bin"
    p.write_bytes(data)

    with BcorFileHandle(str(p)) as h:
        assert not h.is_remote()
        assert h.size == len(data)
        h.seek(10)
        assert h.read(5) == data[10:15]
        assert h.tell() == 15

        # Ranged read returns a buffer-protocol object equal to the slice.
        out = h.read_range(100, 20)
        assert bytes(out) == data[100:120]

        # Batched ranged reads
        ranges = [(0, 8), (100, 16), (2000, 32)]
        results = h.read_ranges(ranges)
        assert [bytes(r) for r in results] == [data[0:8], data[100:116], data[2000:2032]]


def test_local_handle_mmap_path_returns_zero_copy_view(tmp_path):
    """For mmap-backed local files, read_range must avoid copying — return a memoryview."""
    p = tmp_path / "mmap.bin"
    # Force mmap by passing use_mmap=True regardless of size.
    p.write_bytes(b"x" * 4096)

    with BcorFileHandle(str(p), use_mmap=True) as h:
        assert h.mmap is not None
        out = h.read_range(0, 16)
        # Must be a memoryview / buffer protocol object whose memory is the mmap, not a copy.
        assert isinstance(out, memoryview)
        # NumPy can wrap it without copying.
        arr = np.frombuffer(out, dtype=np.uint8)
        assert arr.shape == (16,)


def test_local_handle_rejects_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        BcorFileHandle(str(tmp_path / "nope.bin")).__enter__()


def test_gcs_handle_path_detection():
    h = BcorFileHandle("gs://bucket/path/foo.bcor")
    assert h.is_remote()
    # Don't open; we're just checking path parsing here.
