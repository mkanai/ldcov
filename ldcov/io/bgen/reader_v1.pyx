# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION
# Main BGEN reader implementation

import os
import struct
import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.string cimport string
from libc.stdint cimport uint32_t, uint64_t
import pandas as pd
from typing import Optional, List, Tuple, Dict, Any, Callable
import logging

from ._bgen cimport HeaderInfo, VariantInfo
from libcpp cimport bool
from .header import BgenHeader
from .variant import BgenVariant

logger = logging.getLogger(__name__)

# Import memory-mapped reader if available
try:
    from .mmap_reader import MMapBgenFile
    HAS_MMAP = True
except ImportError:
    HAS_MMAP = False


np.import_array()


cdef class BgenReader:
    """Main BGEN file reader class."""
    
    cdef:
        object file_handle
        object file_path
        object sample_path
        object header_obj  # BgenHeader instance
        list sample_ids
        bool has_index
        object bgi_reader
        HeaderInfo header_info
        bool is_open
        bool _use_mmap
        bool _use_read_ahead
        object _decompressor
        object _decompressor_backend
    
    def __init__(self, file_path, sample_path='', bgi_path=None, use_mmap=True, use_read_ahead=False, decompressor_backend='auto'):
        """
        Initialize BGEN reader.
        
        Parameters
        ----------
        file_path : str
            Path to BGEN file
        sample_path : str, optional
            Path to sample file
        bgi_path : str, optional
            Path to BGI index file. If None, will look for file_path + '.bgi'
        use_mmap : bool, optional
            Whether to use memory-mapped file I/O (default: True)
        use_read_ahead : bool, optional
            Whether to use read-ahead decompression (default: False)
        decompressor_backend : str, optional
            Backend for batch decompressor: 'auto', 'sequential', 'batch', 'cython', or 'python' (default: 'auto')
        """
        self.file_path = file_path
        self.sample_path = sample_path if sample_path else None
        self.is_open = False
        self.has_index = False
        self.bgi_reader = None
        self._use_mmap = use_mmap and HAS_MMAP
        self._use_read_ahead = use_read_ahead
        self._decompressor = None
        self._decompressor_backend = decompressor_backend
        
        # Determine BGI path
        if bgi_path is None:
            bgi_path = file_path + '.bgi'
        
        # Check if BGI index exists
        if os.path.exists(bgi_path):
            from .bgi import BGIReader
            self.bgi_reader = BGIReader(bgi_path)
            self.has_index = True
        
        # Open file
        self._open()
    
    def _open(self):
        """Open BGEN file and read header."""
        if self.is_open:
            return
        
        try:
            # Open file with memory mapping if requested and available
            if self._use_mmap:
                self.file_handle = MMapBgenFile(self.file_path)
            else:
                self.file_handle = open(self.file_path, 'rb')
            
            # Parse header
            self.header_obj = BgenHeader(self.file_handle)
            self.header_info = self.header_obj.info
            
            # Check if uncompressed BGEN - reject it
            if self.header_info.compression == 0:
                raise RuntimeError(
                    "Uncompressed BGEN files are not supported. "
                    "Please use compressed BGEN files (zlib or zstd). "
                    "You can compress your BGEN file using bgenix or qctool2."
                )
            
            # Read sample IDs
            self._read_samples()
            
            self.is_open = True
            
        except Exception as e:
            if hasattr(self, 'file_handle') and self.file_handle:
                self.file_handle.close()
            raise
    
    def _read_samples(self):
        """Read sample IDs from file or sample file."""
        self.sample_ids = []
        
        if self.sample_path and os.path.exists(self.sample_path):
            # Read from .sample file
            self._read_sample_file()
        elif self.header_obj.has_sample_ids:
            # Read from BGEN file
            self._read_sample_block()
        else:
            # Generate default sample IDs
            self.sample_ids = [f"sample_{i}" for i in range(self.header_info.nsamples)]
    
    def _read_sample_block(self):
        """Read sample ID block from BGEN file."""
        # Sample block comes right after the header, not at the offset
        # The offset points to where variant data starts, after the sample block
        # Current position should already be at the end of header
        
        # Read sample block length
        data = self.file_handle.read(4)
        sample_block_length = struct.unpack('<I', data)[0]
        
        # Read number of samples
        data = self.file_handle.read(4)
        n_samples = struct.unpack('<I', data)[0]
        
        if n_samples != self.header_info.nsamples:
            raise ValueError(f"Sample count mismatch: {n_samples} vs {self.header_info.nsamples}")
        
        # Read each sample ID
        self.sample_ids = []
        for i in range(n_samples):
            # Read ID length
            data = self.file_handle.read(2)
            id_length = struct.unpack('<H', data)[0]
            
            # Read ID string
            sample_id = self.file_handle.read(id_length).decode('utf-8', errors='replace')
            self.sample_ids.append(sample_id)
        
        # Update offset to skip sample block
        self.header_info.offset += sample_block_length + 4
    
    def _read_sample_file(self):
        """Read sample IDs from .sample file."""
        with open(self.sample_path, 'r') as f:
            # Skip header lines
            next(f)  # ID_1 ID_2 missing ...
            next(f)  # 0 0 0 ...
            
            # Read sample IDs
            self.sample_ids = []
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    # Use second column (ID_2) as primary ID
                    self.sample_ids.append(parts[1])
    
    @property
    def samples(self):
        """Get list of sample IDs."""
        return self.sample_ids
    
    def __len__(self):
        """Return number of samples."""
        return len(self.sample_ids)
    
    @property
    def header(self):
        """Get header information."""
        return self.header_obj
    
    @property
    def bgi(self):
        """Get BGI reader if available."""
        return self.bgi_reader
    
    @property
    def nsamples(self):
        """Get number of samples."""
        return self.header_info.nsamples
    
    @property
    def compression(self):
        """Get compression type."""
        return self.header_info.compression
    
    @property
    def layout(self):
        """Get layout version."""  
        return self.header_info.layout
    
    @property
    def file_handle(self):
        """Get the file handle (for C++ decompressor access)."""
        return self.file_handle
    
    @property  
    def file_path(self):
        """Get the file path (for C++ decompressor access)."""
        return self.file_path
    
    def read_variants_at_offsets(self, offsets):
        """
        Read variants at specific file offsets.
        
        Parameters
        ----------
        offsets : list of int
            File offsets for variants
        
        Returns
        -------
        list of BgenVariant
            Variant objects
        """
        if not self.is_open:
            raise ValueError("BGEN file is not open")
        
        variants = []
        for offset in offsets:
            var = BgenVariant(
                self.file_handle,
                offset,
                self.header_info.layout,
                self.header_info.compression,
                self.header_info.nsamples
            )
            variants.append(var)
        
        return variants
    
    def create_variant_at_offset(self, offset):
        """
        Create a BgenVariant object at a specific file offset.
        
        This is useful for parallel/async operations where we want to create
        variants in different threads.
        
        Parameters
        ----------
        offset : int
            File offset for the variant
            
        Returns
        -------
        BgenVariant
            Variant object
        """
        if not self.is_open:
            raise ValueError("BGEN file is not open")
        
        return BgenVariant(
            self.file_handle,
            offset,
            self.header_info.layout,
            self.header_info.compression,
            self.header_info.nsamples
        )
    
    def close(self):
        """Close the BGEN file."""
        if self.is_open and self.file_handle:
            self.file_handle.close()
            self.is_open = False
        
        if self.bgi_reader:
            self.bgi_reader.close()
            self.bgi_reader = None
        
        if self._decompressor:
            self._decompressor.close()
            self._decompressor = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def __del__(self):
        """Destructor."""
        self.close()
    
    def __repr__(self):
        """String representation."""
        return f'BgenReader("{self.file_path}")'
    
    
    def get_sample_indices(self, sample_ids_to_keep):
        """
        Map sample IDs to their indices in the BGEN file.
        
        Parameters
        ----------
        sample_ids_to_keep : list of str
            Sample IDs to keep
        
        Returns
        -------
        tuple
            (indices, filtered_sample_ids) where indices are positions in BGEN file
            and filtered_sample_ids are the IDs that were found
        """
        # Convert to numpy arrays for efficient operations
        sample_ids_array = np.array(self.sample_ids)
        ids_to_keep_array = np.array(sample_ids_to_keep)
        
        # Find which requested samples exist in BGEN
        mask = np.isin(sample_ids_array, ids_to_keep_array)
        bgen_indices = np.where(mask)[0]
        found_ids = sample_ids_array[mask]
        
        # Create a mapping to preserve the order of sample_ids_to_keep
        # Use searchsorted for efficient ordering
        sorter = np.argsort(ids_to_keep_array)
        sorted_keep = ids_to_keep_array[sorter]
        
        # Find where each found ID would be inserted in the sorted array
        insert_positions = np.searchsorted(sorted_keep, found_ids)
        
        # Get the original positions in sample_ids_to_keep
        original_positions = sorter[insert_positions]
        
        # Sort by original order
        order = np.argsort(original_positions)
        filtered_ids = found_ids[order].tolist()
        indices = bgen_indices[order].tolist()
        
        return indices, filtered_ids
    
    def load_variants(
        self,
        region_chrom=None,
        region_start=None,
        region_end=None,
        variant_filter=None,
        sample_indices=None,
        dtype=np.float64,
        progress_callback=None
    ):
        """
        Unified method to load variants with various filtering options.
        
        Parameters
        ----------
        region_chrom : str, optional
            Chromosome for region query
        region_start : int, optional
            Start position for region query (inclusive)
        region_end : int, optional
            End position for region query (inclusive)
        variant_filter : dict, optional
            Variant filter from .z file
        sample_indices : List[int], optional
            Sample indices to keep
        dtype : np.dtype
            Data type for dosages
        progress_callback : callable, optional
            Function to call with progress updates
        
        Returns
        -------
        Tuple[np.ndarray, pd.DataFrame]
            (dosages, variant_info)
        """
        if not self.is_open:
            raise ValueError("BGEN file is not open")
        
        if not self.has_index:
            raise ValueError("BGI index is required for efficient variant loading")
        
        # Determine which variants to load based on parameters
        if variant_filter is not None:
            # Load filtered variants
            chromosome = variant_filter["chromosome"]
            positions = np.array(variant_filter["positions"], dtype=np.int32)
            variant_metadata = self.bgi_reader.find_variants_by_filter(
                chromosome, positions, variant_filter["allele1"], variant_filter["allele2"]
            )
        elif region_chrom is not None:
            # Load region variants
            variant_metadata = self.bgi_reader.get_variants_in_region(
                region_chrom, region_start, region_end
            )
        else:
            # Load all variants
            variant_metadata = self.bgi_reader.get_all_variants()
        
        # Load variants from metadata
        return self._load_from_metadata(
            variant_metadata, sample_indices, dtype, progress_callback
        )
    
    def _load_from_metadata(
        self,
        variant_metadata,
        sample_indices=None,
        dtype=np.float64,
        progress_callback=None
    ):
        """
        Load variant dosages using metadata from BGI.
        
        Parameters
        ----------
        variant_metadata : pd.DataFrame
            DataFrame from BGI with variant metadata
        sample_indices : List[int], optional
            Sample indices to keep
        dtype : np.dtype
            Data type for dosages
        progress_callback : callable, optional
            Function to call with progress updates
        
        Returns
        -------
        Tuple[np.ndarray, pd.DataFrame]
            (dosages, variant_info)
        """
        n_variants = len(variant_metadata)
        if n_variants == 0:
            n_samples_out = len(sample_indices) if sample_indices is not None else self.header_info.nsamples
            return np.empty((n_samples_out, 0), dtype=dtype), pd.DataFrame()
        
        # Determine output dimensions
        n_samples_out = len(sample_indices) if sample_indices is not None else self.header_info.nsamples
        
        # Pre-allocate dosage array
        dosages = np.empty((n_samples_out, n_variants), dtype=dtype)
        
        # Extract file offsets from metadata
        file_offsets = variant_metadata["file_offset"].values
        
        # Use read-ahead decompression if enabled
        if self._use_read_ahead:
            # Initialize decompressor if not already done
            if self._decompressor is None:
                from .legacy.batch_decompressor import BatchDecompressor
                # Use the specified backend or automatic selection
                self._decompressor = BatchDecompressor(self, backend=self._decompressor_backend)
            
            # Process with read-ahead - BatchDecompressor.process_variants expects variant_metadata
            dosages_out, variant_info_list = self._decompressor.process_variants(
                file_offsets.tolist(), variant_metadata, sample_indices, progress_callback
            )
            dosages[:] = dosages_out
        else:
            # Original synchronous processing
            variants = self.read_variants_at_offsets(file_offsets.tolist())
            
            # Process variants
            for i, variant in enumerate(variants):
                if sample_indices is not None:
                    # Use the efficient filtered computation
                    dosages[:, i] = variant.get_dosage_for_samples(sample_indices)
                else:
                    # Get all samples
                    dosages[:, i] = variant.alt_dosage
                
                # Call progress callback if provided
                if progress_callback is not None:
                    progress_callback(i + 1)
        
        # Prepare variant info DataFrame with consistent column names
        variant_info = variant_metadata[["chrom", "pos", "rsid", "ref", "alt"]].copy()
        
        return dosages, variant_info