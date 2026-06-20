"""Filesystem resolution shared by the BlockMatrix reader and variant index."""

from typing import Optional, Tuple

from fsspec.core import url_to_fs
from fsspec.utils import get_protocol


def resolve_filesystem(path: str, storage_options: Optional[dict] = None) -> Tuple[object, str]:
    """Return ``(fs, inner_path)`` for ``path``, defaulting ``s3://`` to anonymous access.

    ``s3://`` with no explicit ``storage_options`` -> ``anon=True`` so public buckets
    (e.g. Pan-UKB ``s3://pan-ukb-us-east-1/...``) work out of the box. Any provided
    ``storage_options`` dict (even ``{}``) is used verbatim, so callers can force the
    normal AWS credential chain (``{}``) or pass ``key``/``secret``/``endpoint_url``/
    ``requester_pays`` explicitly. ``gs://`` and local paths are unaffected.
    """
    if storage_options is None and get_protocol(path) == "s3":
        storage_options = {"anon": True}
    fs, inner_path = url_to_fs(path, **(storage_options or {}))
    return fs, inner_path
