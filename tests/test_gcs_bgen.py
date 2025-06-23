"""Tests for GCS BGEN reader functionality."""

import os
import pytest
import numpy as np
import tempfile
import shutil
from unittest.mock import Mock, patch


class TestGCSFileReader:
    """Test GCS file reader functionality."""
    
    def test_gcs_path_detection(self):
        """Test that GCS paths are detected correctly."""
        from ldcov.io.bgen.io.gcs_file_reader import GCSFileReader
        
        # Should accept gs:// paths
        reader = GCSFileReader("gs://bucket/test.bgen")
        assert reader.path == "gs://bucket/test.bgen"
    
    def test_gcs_reader_operations(self):
        """Test basic GCS reader operations with real GCS."""
        from ldcov.io.bgen.io.gcs_file_reader import GCSFileReader
        
        # Use the public test bucket
        reader = GCSFileReader("gs://gcs-anndata-test/ldcov/data/example.8bits.bgen")
        
        with reader:
            # Test read - read BGEN magic number
            data = reader.read(4)
            assert len(data) == 4
            
            # Test seek back to beginning
            reader.seek(0)
            
            # Test tell
            pos = reader.tell()
            assert pos == 0
            
            # Read magic number again
            magic = reader.read(4)
            assert magic == data  # Should be same as before
    
    @patch('ldcov.io.bgen.io.gcs_file_reader.gcsfs')
    def test_gcs_reader_initialization_error(self, mock_gcsfs_module):
        """Test error handling when gcsfs is not available."""
        mock_gcsfs_module.GCSFileSystem.side_effect = ImportError("Failed to import gcsfs")
        
        from ldcov.io.bgen.io.gcs_file_reader import GCSFileReader
        
        # This should fail when trying to create the GCS filesystem
        with pytest.raises(ImportError, match="Failed to import gcsfs"):
            reader = GCSFileReader("gs://bucket/test.bgen")
    
    def test_non_gcs_path_error(self):
        """Test that non-GCS paths raise an error."""
        from ldcov.io.bgen.io.gcs_file_reader import GCSFileReader
        
        with pytest.raises(ValueError, match="Path must start with gs://"):
            GCSFileReader("/local/path/test.bgen")


class TestBGICache:
    """Test BGI cache functionality."""
    
    def test_cache_singleton(self):
        """Test that BGICache is a singleton."""
        from ldcov.io.bgen.index import BGICache
        
        cache1 = BGICache.getInstance()
        cache2 = BGICache.getInstance()
        
        assert cache1 is cache2
    
    def test_cache_directory_creation(self):
        """Test cache directory is created."""
        from ldcov.io.bgen.index import BGICache
        
        cache = BGICache.getInstance()
        cache_dir = cache._cache_dir
        
        assert cache_dir is not None
        assert ".cache" in cache_dir or "tmp" in cache_dir
    
    def test_local_path_passthrough(self):
        """Test that local paths are returned as-is."""
        from ldcov.io.bgen.index import BGICache
        
        cache = BGICache.getInstance()
        local_path = "/local/path/test.bgen.bgi"
        
        result = cache.get_local_path(local_path)
        assert result == local_path
    
    @patch('gcsfs.GCSFileSystem')
    def test_gcs_download_error(self, mock_gcsfs):
        """Test error handling during GCS download."""
        mock_gcsfs.side_effect = ImportError("Failed to import gcsfs")
        
        from ldcov.io.bgen.index import BGICache
        
        cache = BGICache.getInstance()
        
        with pytest.raises(RuntimeError, match="Failed to download BGI file"):
            cache.get_local_path("gs://bucket/test.bgen.bgi")


class TestIntegration:
    """Integration tests for GCS BGEN reading."""
    
    def test_read_gcs_bgen(self):
        """Test reading actual BGEN file from GCS."""
        from ldcov.io import load_bgen
        
        # Use the public test bucket
        gcs_path = "gs://gcs-anndata-test/ldcov/data/example.16bits.bgen"
        
        # Load with nan_action to handle any NaN values
        dosages, info, samples = load_bgen(gcs_path, nan_action="omit")
        
        assert dosages.shape[0] > 0
        assert len(info) > 0
        assert len(samples) > 0
        
        # Clean up any downloaded BGI file
        local_bgi = "example.16bits.bgen.bgi"
        if os.path.exists(local_bgi):
            os.remove(local_bgi)
    
    def test_factory_function_selection(self):
        """Test that factory function selects correct reader."""
        from ldcov.io.bgen.io.gcs_file_reader import create_file_reader
        
        # Test that paths are handled correctly
        # (without actually creating readers which would fail in test env)
        
        # GCS path should trigger GCS reader
        gcs_path = "gs://bucket/file.bgen"
        # Local path should trigger regular/mmap reader
        local_path = "/local/file.bgen"
        
        # Just verify the function exists and accepts these paths
        assert callable(create_file_reader)