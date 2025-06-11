"""
Test BGEN reader with mandatory BGI support.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

from ldcov.io.bgen_reader import BgenFileReader, load_bgen, BGEN_AVAILABLE
from ldcov.utils.variant_filter import read_z_file, create_variant_filter_from_z


class TestBgenReaderBGI:
    """Test BGEN reader with BGI optimization."""
    
    @pytest.fixture
    def example_bgen_path(self):
        """Path to example BGEN file with BGI."""
        return Path(__file__).parent.parent / "examples" / "data" / "data.bgen"
    
    @pytest.fixture
    def example_sample_path(self):
        """Path to example sample file."""
        return Path(__file__).parent.parent / "examples" / "data" / "data.sample"
    
    @pytest.fixture
    def example_z_path(self):
        """Path to example z file."""
        return Path(__file__).parent.parent / "examples" / "data" / "data.z"
    
    @pytest.mark.skipif(not BGEN_AVAILABLE, reason="bgen module not available")
    def test_bgen_reader_requires_bgi(self):
        """Test that BGEN reader requires BGI file."""
        # Create a temp BGEN file without BGI
        with tempfile.NamedTemporaryFile(suffix=".bgen") as f:
            # Should fail without BGI
            with pytest.raises(FileNotFoundError, match="BGI index required"):
                BgenFileReader(f.name)
    
    @pytest.mark.skipif(not BGEN_AVAILABLE, reason="bgen module not available")
    def test_bgen_reader_init(self, example_bgen_path, example_sample_path):
        """Test BGEN reader initialization with BGI."""
        # Should succeed with BGI present
        reader = BgenFileReader(
            str(example_bgen_path),
            sample_path=str(example_sample_path)
        )
        
        assert reader.n_samples == 5363  # Known from example data
        assert reader.n_variants == 55  # From BGI
        assert len(reader.sample_ids) == 5363
        
        reader.close()
    
    @pytest.mark.skipif(not BGEN_AVAILABLE, reason="bgen module not available")
    def test_load_all_variants(self, example_bgen_path, example_sample_path):
        """Test loading all variants."""
        reader = BgenFileReader(
            str(example_bgen_path),
            sample_path=str(example_sample_path)
        )
        
        dosages, variant_info = reader.load_all_variants()
        
        # Check dimensions
        assert dosages.shape == (5363, 55)  # 5363 samples, 55 variants
        assert len(variant_info) == 55
        
        # Check variant info columns
        expected_cols = {'chrom', 'pos', 'id', 'rsid', 'ref', 'alt', 'idx'}
        assert set(variant_info.columns) == expected_cols
        
        # Check first variant
        first = variant_info.iloc[0]
        assert first['chrom'] == '01'
        assert first['pos'] == 1
        assert first['rsid'] == 'rs1'
        assert first['ref'] == 'A'
        assert first['alt'] == 'G'
        
        # Check dosages are in valid range
        assert np.all(dosages >= 0) and np.all(dosages <= 2)
        
        reader.close()
    
    @pytest.mark.skipif(not BGEN_AVAILABLE, reason="bgen module not available")
    def test_load_region_variants(self, example_bgen_path):
        """Test loading variants from a region."""
        reader = BgenFileReader(str(example_bgen_path))
        
        # Load region with variants
        dosages, variant_info = reader.load_region_variants('01', 1, 10)
        
        assert dosages.shape[1] == 10  # 10 variants
        assert len(variant_info) == 10
        assert np.all(variant_info['pos'] >= 1)
        assert np.all(variant_info['pos'] <= 10)
        
        # Empty region
        dosages2, variant_info2 = reader.load_region_variants('01', 100000, 200000)
        assert dosages2.shape == (reader.n_samples, 0)
        assert len(variant_info2) == 0
        
        reader.close()
    
    @pytest.mark.skipif(not BGEN_AVAILABLE, reason="bgen module not available")
    def test_load_filtered_variants(self, example_bgen_path, example_z_path):
        """Test loading filtered variants from z file."""
        reader = BgenFileReader(str(example_bgen_path))
        
        # Create filter from z file
        z_df = read_z_file(str(example_z_path))
        variant_filter = create_variant_filter_from_z(z_df)
        
        # Load filtered variants
        dosages, variant_info = reader.load_filtered_variants(variant_filter)
        
        # Should have loaded the variants in z file
        assert dosages.shape[1] == len(variant_filter['positions'])
        assert len(variant_info) == len(variant_filter['positions'])
        
        # Check order matches z file
        assert list(variant_info['rsid']) == variant_filter['rsids']
        
        reader.close()
    
    @pytest.mark.skipif(not BGEN_AVAILABLE, reason="bgen module not available")
    def test_sample_filtering(self, example_bgen_path, example_sample_path):
        """Test sample filtering."""
        reader = BgenFileReader(
            str(example_bgen_path),
            sample_path=str(example_sample_path)
        )
        
        # Get subset of samples
        sample_ids_to_keep = reader.sample_ids[:3]  # First 3 samples
        sample_indices, filtered_ids = reader.get_sample_indices(sample_ids_to_keep)
        
        assert len(sample_indices) == 3
        assert len(filtered_ids) == 3
        assert filtered_ids == sample_ids_to_keep
        
        # Load with sample filtering
        dosages, variant_info = reader.load_all_variants(sample_indices)
        
        assert dosages.shape == (3, 55)  # 3 samples, 55 variants
        
        reader.close()
    
    @pytest.mark.skipif(not BGEN_AVAILABLE, reason="bgen module not available")
    def test_load_bgen_function(self, example_bgen_path, example_sample_path):
        """Test the main load_bgen function."""
        # Load all variants
        dosages, variant_info, sample_ids = load_bgen(
            str(example_bgen_path),
            sample_path=str(example_sample_path)
        )
        
        assert dosages.shape == (5363, 55)
        assert len(variant_info) == 55
        assert len(sample_ids) == 5363
        
        # With region
        dosages2, variant_info2, sample_ids2 = load_bgen(
            str(example_bgen_path),
            sample_path=str(example_sample_path),
            region="01:5-15"
        )
        
        assert dosages2.shape == (5363, 11)  # variants 5-15 inclusive
        assert len(variant_info2) == 11
        assert sample_ids2 == sample_ids
    
    @pytest.mark.skipif(not BGEN_AVAILABLE, reason="bgen module not available")
    def test_nan_handling(self, example_bgen_path):
        """Test NaN handling options."""
        # Note: The example BGEN file doesn't contain NaN values,
        # so nan_action is accepted but not actually triggered
        
        # Should not raise with valid nan_action
        for action in ['error', 'mean', 'omit']:
            dosages, variant_info, sample_ids = load_bgen(
                str(example_bgen_path),
                nan_action=action
            )
            assert dosages is not None
            assert not np.any(np.isnan(dosages))  # No NaN values in test data
        
        # Invalid nan_action is only checked when NaN values are present
        # Since test data has no NaN, this won't raise an error
        # TODO: Add test with BGEN file containing NaN values