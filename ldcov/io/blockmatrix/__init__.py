"""Pure-Python reader for Hail BlockMatrix on-disk format (filesystem-agnostic)."""

from .block_codec import decode_block
from .reader import HailBlockMatrixReader

__all__ = ["decode_block", "HailBlockMatrixReader"]
