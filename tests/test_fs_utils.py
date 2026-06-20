import pytest

import ldcov.io.fs_utils
from ldcov.io.fs_utils import resolve_filesystem


def test_s3_defaults_to_anon(monkeypatch):
    seen = {}

    def fake_url_to_fs(path, **kw):
        seen["path"] = path
        seen["kw"] = kw
        return ("FS", path.split("://", 1)[-1])

    monkeypatch.setattr(ldcov.io.fs_utils, "url_to_fs", fake_url_to_fs)
    fs, inner = resolve_filesystem("s3://bucket/obj")
    assert fs == "FS"
    assert inner == "bucket/obj"
    assert seen["kw"] == {"anon": True}


def test_s3_explicit_empty_options_uses_default_chain(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        ldcov.io.fs_utils,
        "url_to_fs",
        lambda path, **kw: seen.update(kw) or ("FS", path),
    )
    resolve_filesystem("s3://bucket/obj", storage_options={})
    assert seen == {}  # no anon injected -> normal AWS credential chain


def test_s3_explicit_options_passed_verbatim(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        ldcov.io.fs_utils,
        "url_to_fs",
        lambda path, **kw: seen.update(kw) or ("FS", path),
    )
    resolve_filesystem("s3://bucket/obj", storage_options={"key": "k", "secret": "s"})
    assert seen == {"key": "k", "secret": "s"}


def test_gs_and_local_not_touched(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        ldcov.io.fs_utils,
        "url_to_fs",
        lambda path, **kw: seen.update({path: kw}) or ("FS", path),
    )
    resolve_filesystem("gs://bucket/obj")
    resolve_filesystem("/local/path")
    assert seen["gs://bucket/obj"] == {}
    assert seen["/local/path"] == {}


def test_reader_uses_resolve_filesystem(monkeypatch):
    import ldcov.io.blockmatrix.reader as rmod

    calls = {}

    def fake_resolve(path, storage_options=None):
        calls["path"] = path
        calls["opts"] = storage_options
        raise RuntimeError("stop after fs resolution")  # short-circuit metadata read

    monkeypatch.setattr(rmod, "resolve_filesystem", fake_resolve)
    with pytest.raises(RuntimeError, match="stop after fs resolution"):
        rmod.HailBlockMatrixReader("s3://bucket/x.bm", storage_options=None)
    assert calls["path"] == "s3://bucket/x.bm"
    assert calls["opts"] is None


def test_variant_index_uses_resolve_filesystem(monkeypatch):
    import ldcov.io.variant_index as vmod

    calls = {}

    def fake_resolve(path, storage_options=None):
        calls["path"] = path
        calls["opts"] = storage_options
        raise RuntimeError("stop after fs resolution")

    monkeypatch.setattr(vmod, "resolve_filesystem", fake_resolve)
    with pytest.raises(RuntimeError, match="stop after fs resolution"):
        vmod.VariantIndex("s3://bucket/s.parquet")
    assert calls["path"] == "s3://bucket/s.parquet"
    assert calls["opts"] is None
