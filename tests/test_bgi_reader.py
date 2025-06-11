"""
Test BGI (BGEN Index) reader functionality.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from ldcov.io.bgi_reader import BGIReader


class TestBGIReader:
    """Test BGI reader functionality."""
    
    @pytest.fixture
    def example_bgi_path(self):
        """Path to example BGI file."""
        # Use the example data BGI file
        return Path(__file__).parent.parent / "examples" / "data" / "data.bgen.bgi"
    
    def test_bgi_reader_init(self, example_bgi_path):
        """Test BGI reader initialization."""
        # Should succeed with valid BGI
        reader = BGIReader(str(example_bgi_path))
        assert reader is not None
        reader.close()
        
        # Should fail with non-existent file
        with pytest.raises(FileNotFoundError):
            BGIReader("/non/existent/file.bgi")
        
        # Should fail with invalid file
        with tempfile.NamedTemporaryFile(suffix=".bgi") as f:
            f.write(b"not a valid bgi file")
            f.flush()
            with pytest.raises(ValueError, match="Error reading BGI file"):
                BGIReader(f.name)
    
    def test_get_variant_count(self, example_bgi_path):
        """Test getting variant count."""
        with BGIReader(str(example_bgi_path)) as reader:
            count = reader.get_variant_count()
            assert count == 55  # Known count from example data
            
            # Should be cached
            count2 = reader.get_variant_count()
            assert count2 == count
    
    def test_get_all_variants(self, example_bgi_path):
        """Test getting all variant metadata."""
        with BGIReader(str(example_bgi_path)) as reader:
            variants = reader.get_all_variants()
            
            # Check structure
            assert len(variants) == 55
            assert variants.dtype.names == ('chrom', 'pos', 'rsid', 'n_alleles', 
                                           'ref', 'alt', 'file_offset', 'size_bytes')
            
            # Check first variant
            first = variants[0]
            assert first['chrom'] == '01'
            assert first['pos'] == 1
            assert first['rsid'] == 'rs1'
            assert first['n_alleles'] == 2
            assert first['ref'] == 'A'
            assert first['alt'] == 'G'
            assert first['file_offset'] > 0
            assert first['size_bytes'] > 0
            
            # Check ordering by file offset
            offsets = variants['file_offset']
            assert np.all(offsets[1:] > offsets[:-1])  # Strictly increasing
    
    def test_get_variants_in_region(self, example_bgi_path):
        """Test getting variants in a genomic region."""
        with BGIReader(str(example_bgi_path)) as reader:
            # Region with variants
            variants = reader.get_variants_in_region('01', 1, 10)
            assert len(variants) == 10
            assert np.all(variants['chrom'] == '01')
            assert np.all(variants['pos'] >= 1)
            assert np.all(variants['pos'] <= 10)
            
            # Empty region
            variants = reader.get_variants_in_region('01', 100000, 200000)
            assert len(variants) == 0
            
            # Different chromosome
            variants = reader.get_variants_in_region('02', 1, 100)
            assert len(variants) == 0
    
    def test_find_variants_by_filter(self, example_bgi_path):
        """Test finding variants by position/allele/rsid."""
        with BGIReader(str(example_bgi_path)) as reader:
            # Create filter matching some variants
            positions = np.array([1, 5, 10, 99999], dtype=np.int32)
            alleles1 = ['A', 'A', 'A', 'X']
            alleles2 = ['G', 'G', 'G', 'Y']
            rsids = ['rs1', 'rs5', 'rs10', 'rs_missing']
            
            matched, indices = reader.find_variants_by_filter(
                positions, alleles1, alleles2, rsids
            )
            
            # Should find 3 out of 4
            assert len(matched) == 3
            assert len(indices) == 3
            assert np.array_equal(indices, [0, 1, 2])
            
            # Check matched variants
            assert matched[0]['pos'] == 1
            assert matched[0]['rsid'] == 'rs1'
            assert matched[1]['pos'] == 5
            assert matched[1]['rsid'] == 'rs5'
            assert matched[2]['pos'] == 10
            assert matched[2]['rsid'] == 'rs10'
            
            # Test with swapped alleles (should still match)
            alleles1_swap = ['G', 'G', 'G', 'Y']
            alleles2_swap = ['A', 'A', 'A', 'X']
            
            matched2, indices2 = reader.find_variants_by_filter(
                positions, alleles1_swap, alleles2_swap, rsids
            )
            
            assert len(matched2) == 3
            assert np.array_equal(matched2['pos'], matched['pos'])
    
    def test_empty_results(self, example_bgi_path):
        """Test empty result handling."""
        with BGIReader(str(example_bgi_path)) as reader:
            # No matching variants
            positions = np.array([99999, 88888], dtype=np.int32)
            alleles1 = ['X', 'Y']
            alleles2 = ['Z', 'W']
            rsids = ['rs_none1', 'rs_none2']
            
            matched, indices = reader.find_variants_by_filter(
                positions, alleles1, alleles2, rsids
            )
            
            assert len(matched) == 0
            assert len(indices) == 0
            assert matched.dtype.names == ('chrom', 'pos', 'rsid', 'n_alleles',
                                          'ref', 'alt', 'file_offset', 'size_bytes')
    
    def test_context_manager(self, example_bgi_path):
        """Test context manager usage."""
        with BGIReader(str(example_bgi_path)) as reader:
            count = reader.get_variant_count()
            assert count > 0
        
        # Connection should be closed after context
        # (Can't easily test this without accessing private attributes)