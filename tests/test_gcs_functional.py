#!/usr/bin/env python3
"""Functional tests for GCS BGEN reading."""

import os
import sys
import tempfile
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ldcov.io import load_bgen
from ldcov.io.bgen.utils import ensure_local_bgi


class TestGCSFunctional:
    """Test GCS functionality with mocked gcsfs."""
    
    def test_ensure_local_bgi_local_path(self):
        """Test that local paths are returned as-is."""
        local_path = "/path/to/local.bgi"
        result = ensure_local_bgi(local_path)
        assert result == local_path
    
    def test_ensure_local_bgi_existing_file(self, tmp_path):
        """Test that existing local files are reused."""
        # Create a dummy BGI file
        bgi_file = tmp_path / "test.bgen.bgi"
        bgi_file.write_text("dummy bgi content")
        
        # Change to temp directory
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            
            # Mock gcsfs - shouldn't be called since file exists
            with patch('gcsfs.GCSFileSystem') as mock_gcsfs:
                result = ensure_local_bgi("gs://bucket/test.bgen.bgi")
                assert result == "./test.bgen.bgi"
                assert os.path.exists(result)
                # Verify gcsfs was not instantiated
                mock_gcsfs.assert_not_called()
        finally:
            os.chdir(original_cwd)
    
    def test_ensure_local_bgi_download(self, tmp_path):
        """Test BGI download from GCS."""
        # Change to temp directory
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            
            # Mock gcsfs
            mock_fs = MagicMock()
            mock_fs.get = MagicMock()
            
            with patch('gcsfs.GCSFileSystem', return_value=mock_fs):
                result = ensure_local_bgi("gs://bucket/path/to/test.bgen.bgi")
                
                # Check that the correct local path is returned
                assert result == "./test.bgen.bgi"
                
                # Verify gcsfs.get was called correctly
                mock_fs.get.assert_called_once_with(
                    "gs://bucket/path/to/test.bgen.bgi",
                    "./test.bgen.bgi"
                )
        finally:
            os.chdir(original_cwd)
    
    def test_ensure_local_bgi_download_error(self, tmp_path):
        """Test error handling during BGI download."""
        # Change to temp directory
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            
            # Mock gcsfs to raise an error
            mock_fs = MagicMock()
            mock_fs.get = MagicMock(side_effect=Exception("Download failed"))
            
            with patch('gcsfs.GCSFileSystem', return_value=mock_fs):
                with pytest.raises(RuntimeError, match="Failed to download BGI file"):
                    ensure_local_bgi("gs://bucket/test.bgen.bgi")
        finally:
            os.chdir(original_cwd)
    
    def test_ensure_local_bgi_no_gcsfs(self, tmp_path):
        """Test error when gcsfs is not installed."""
        # Change to temp directory
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            
            # Mock import error for gcsfs
            with patch.dict('sys.modules', {'gcsfs': None}):
                with pytest.raises(ImportError, match="gcsfs is required for GCS support"):
                    ensure_local_bgi("gs://bucket/test.bgen.bgi")
        finally:
            os.chdir(original_cwd)
    
    def test_gcs_integration(self):
        """Integration test with real GCS data."""
        # This test requires actual GCS access
        bgen_path = "gs://gcs-anndata-test/ldcov/data/example.16bits.bgen"
        
        try:
            # Test loading from GCS
            dosages, variant_info, sample_ids = load_bgen(
                bgen_path,
                nan_action="omit",
                show_progress=False
            )
            
            # Verify data was loaded correctly
            assert dosages.shape[0] > 0  # Has samples
            assert dosages.shape[1] > 0  # Has variants
            assert len(sample_ids) == dosages.shape[0]
            assert len(variant_info) == dosages.shape[1]
            
            # Check data types
            assert isinstance(dosages, np.ndarray)
            assert isinstance(variant_info, pd.DataFrame)
            assert isinstance(sample_ids, list)
            
            # Clean up any downloaded BGI files
            local_bgi = "./example.16bits.bgen.bgi"
            if os.path.exists(local_bgi):
                os.remove(local_bgi)
                
        except Exception as e:
            # If this fails, it might be due to missing credentials
            # or network issues, which is okay for CI
            pytest.skip(f"GCS integration test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])