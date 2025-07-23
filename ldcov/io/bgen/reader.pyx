# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
# distutils: language=c++
# distutils: extra_compile_args = -std=c++11
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION
# New BGEN reader implementation with improved C++ integration

import os
import struct
import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.memory cimport unique_ptr, make_unique
from libc.stdint cimport uint32_t, uint64_t, uint8_t, uint16_t
from libc.string cimport memcpy
import pandas as pd
from typing import Optional, List, Tuple, Dict, Any, Callable, Union
import logging

# Note: Since this file is reader.pyx and has a corresponding reader.pxd,
# we don't need to explicitly import - the declarations are automatically available

logger = logging.getLogger(__name__)

np.import_array()

# Helper to create file reader from Python file object
cdef unique_ptr[FileReader] create_file_reader(file_obj):
    """Create a C++ FileReader from a Python file object."""
    return unique_ptr[FileReader](new FileReaderWrapper(file_obj))


cdef class BgenReader:
    """
    High-performance BGEN file reader with C++ integration.
    
    This reader uses an optimized decompressor architecture for better performance
    and automatic optimization based on access patterns.
    """
    
    def __init__(self, file_path: str, bgi_path: Optional[str] = None,
                 sample_path: Optional[str] = None, 
                 decompressor_type: str = 'adaptive',
                 num_threads: int = 0):
        """
        Initialize BGEN reader.
        
        Parameters
        ----------
        file_path : str
            Path to BGEN file
        bgi_path : str, optional
            Path to BGI index file. If None, will look for file_path + '.bgi'
        sample_path : str, optional
            Path to sample file
        decompressor_type : str, optional
            Type of decompressor: 'adaptive', 'sequential', 'parallel' (default: 'adaptive')
        num_threads : int, optional
            Number of threads for parallel decompressor (0 = auto-detect)
        """
        self.file_path = file_path
        self.bgi_path = bgi_path or (file_path + '.bgi')
        self.is_open = False
        self.sample_ids = []
        self.sample_filter = None
        
        # Check files exist (skip for GCS paths as they'll be handled by C++)
        if not self.file_path.startswith('gs://'):
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"BGEN file not found: {self.file_path}")
        if not self.bgi_path.startswith('gs://'):
            if not os.path.exists(self.bgi_path):
                raise FileNotFoundError(f"BGI index not found: {self.bgi_path}")
        
        # Initialize C++ components
        self._init_reader()
        
        # Mark as open before configuring decompressor
        self.is_open = True
        
        # Configure decompressor
        self.set_decompressor_type(decompressor_type, num_threads)
        
        # Load samples
        if sample_path:
            self._load_sample_file(sample_path)
        else:
            self._load_samples_from_bgen()
    
    cdef void _init_reader(self) except *:
        """Initialize C++ reader components."""
        cdef string cpp_file_path = self.file_path.encode('utf-8')
        cdef string cpp_bgi_path
        
        # Handle BGI cache for GCS paths
        if self.bgi_path.startswith('gs://'):
            # Download BGI file to current directory before passing to C++
            from .utils import ensure_local_bgi
            local_bgi_path = ensure_local_bgi(self.bgi_path)
            cpp_bgi_path = local_bgi_path.encode('utf-8')
            # Update self.bgi_path to the local path for consistency
            self.bgi_path = local_bgi_path
        else:
            cpp_bgi_path = self.bgi_path.encode('utf-8')
        
        try:
            # Create main reader
            self.impl.reset(new BgenReaderImpl(cpp_file_path, cpp_bgi_path))
            
            # Store header info
            self.header_info = self.impl.get().header()
            
            # Create BGI reader separately
            self.bgi_reader.reset(new BGIReader(cpp_bgi_path))
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize BGEN reader: {e}")
    
    def _load_sample_file(self, sample_path: str):
        """Load sample IDs from .sample file."""
        with open(sample_path, 'r') as f:
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
    
    cdef void _load_samples_from_bgen(self) except *:
        """Load sample IDs from BGEN file."""
        cdef vector[string] cpp_samples = self.impl.get().get_sample_ids()
        self.sample_ids = [s.decode('utf-8') for s in cpp_samples]
    
    def set_decompressor_type(self, decompressor_type: str, num_threads: int = 0):
        """
        Set the decompressor type.
        
        Parameters
        ----------
        decompressor_type : str
            Type of decompressor: 'adaptive', 'sequential', 'parallel'
        num_threads : int
            Number of threads for parallel decompressor (0 = auto-detect)
        """
        self._ensure_open()
        
        cdef string cpp_type = decompressor_type.encode('utf-8')
        self.impl.get().set_decompressor_type(cpp_type)
        
        if decompressor_type == 'parallel' and num_threads > 0:
            self.impl.get().set_num_threads(num_threads)
    
    def set_sample_filter(self, sample_ids_to_keep: List[str]):
        """
        Set sample filter for efficient subset loading.
        
        Parameters
        ----------
        sample_ids_to_keep : List[str]
            Sample IDs to keep
        
        Returns
        -------
        Tuple[np.ndarray, List[str]]
            (indices, filtered_sample_ids)
        """
        self._ensure_open()
        
        # Convert to numpy arrays for efficient operations
        sample_ids_array = np.array(self.sample_ids)
        ids_to_keep_array = np.array(sample_ids_to_keep)
        
        # Find which requested samples exist in BGEN
        mask = np.isin(sample_ids_array, ids_to_keep_array)
        bgen_indices = np.where(mask)[0].astype(np.uint32)
        found_ids = sample_ids_array[mask]
        
        # Preserve order of sample_ids_to_keep
        sorter = np.argsort(ids_to_keep_array)
        sorted_keep = ids_to_keep_array[sorter]
        insert_positions = np.searchsorted(sorted_keep, found_ids)
        original_positions = sorter[insert_positions]
        order = np.argsort(original_positions)
        
        filtered_ids = found_ids[order].tolist()
        indices = bgen_indices[order]
        
        # Store filter
        self.sample_filter = indices
        
        # Set filter in C++ reader
        cdef vector[uint32_t] cpp_indices
        for idx in indices:
            cpp_indices.push_back(idx)
        self.impl.get().set_sample_filter(cpp_indices)
        
        return indices, filtered_ids
    
    def load_variants(
        self,
        region_chrom: Optional[str] = None,
        region_start: Optional[int] = None,
        region_end: Optional[int] = None,
        variant_filter: Optional[Dict] = None,
        sample_indices: Optional[np.ndarray] = None,
        dtype = np.float64,
        progress_callback: Optional[Callable[[int], None]] = None
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Load variants with various filtering options.
        
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
        sample_indices : np.ndarray, optional
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
        self._ensure_open()
        
        # Get variant metadata based on query type
        cdef vector[VariantMetadata] variant_metadata
        cdef string cpp_chrom
        
        if variant_filter is not None:
            # Filtered variants
            variant_metadata = self._get_filtered_variants(variant_filter)
        elif region_chrom is not None:
            # Region query
            cpp_chrom = region_chrom.encode('utf-8')
            variant_infos = self.bgi_reader.get().query_region(
                cpp_chrom, region_start or 0, region_end or 0xFFFFFFFF
            )
            # Convert VariantInfo to VariantMetadata by reading from file
            variant_metadata = vector[VariantMetadata]()
            for info in variant_infos:
                variant_metadata.push_back(self.impl.get().read_variant_metadata(info.file_offset))
        else:
            # All variants - use efficient batch query
            variant_infos = self.bgi_reader.get().get_all_variants()
            variant_metadata = vector[VariantMetadata]()
            
            # Convert VariantInfo to VariantMetadata by reading from file
            for info in variant_infos:
                variant_metadata.push_back(self.impl.get().read_variant_metadata(info.file_offset))
        
        # Load variant data
        return self._load_variants_from_metadata(
            variant_metadata, sample_indices, dtype, progress_callback
        )
    
    cdef vector[VariantMetadata] _get_filtered_variants(self, variant_filter) except *:
        """Get variants matching filter criteria using optimized BGI method."""
        cdef vector[VariantMetadata] filtered
        cdef vector[VariantInfo] variant_infos
        cdef string cpp_chrom = variant_filter["chromosome"].encode('utf-8')
        cdef vector[uint32_t] positions
        cdef vector[string] alleles1
        cdef vector[string] alleles2
        
        # Convert Python lists to C++ vectors
        for pos in variant_filter["positions"]:
            positions.push_back(pos)
        
        for a1 in variant_filter["allele1"]:
            alleles1.push_back(a1.encode('utf-8'))
            
        for a2 in variant_filter["allele2"]:
            alleles2.push_back(a2.encode('utf-8'))
        
        # Use the optimized find_variants_by_filter method
        variant_infos = self.bgi_reader.get().find_variants_by_filter(
            cpp_chrom, positions, alleles1, alleles2, 1000
        )
        
        # Convert VariantInfo to VariantMetadata by reading from file
        for info in variant_infos:
            filtered.push_back(self.impl.get().read_variant_metadata(info.file_offset))
        
        return filtered
    
    cdef tuple _load_variants_from_metadata(
        self,
        vector[VariantMetadata]& variant_metadata,
        np.ndarray sample_indices,
        dtype,
        progress_callback
    ):
        """Load variant dosages from metadata."""
        cdef int n_variants = variant_metadata.size()
        if n_variants == 0:
            n_samples = len(sample_indices) if sample_indices is not None else self.header_info.nsamples
            return np.empty((n_samples, 0), dtype=dtype), pd.DataFrame()
        
        # Determine output dimensions
        cdef int n_samples_out
        if sample_indices is not None:
            n_samples_out = len(sample_indices)
        elif self.sample_filter is not None:
            n_samples_out = len(self.sample_filter)
        else:
            n_samples_out = self.header_info.nsamples
        
        # OPTIMIZATION: Pre-allocate the entire dosage array at once
        # This avoids reallocation and improves memory locality
        dosages = np.empty((n_samples_out, n_variants), dtype=dtype, order='F')  # Fortran order for column-wise access
        
        # Process variants in batches for better performance
        # OPTIMIZATION: Dynamic batch size based on file size and variant count
        cdef int batch_size
        if n_variants < 1000:
            batch_size = 100  # Small files: smaller batches
        elif n_variants < 10000:
            batch_size = 1000  # Medium files: standard batches
        else:
            batch_size = 5000  # Large files: larger batches for better efficiency
        cdef int i, batch_start, batch_end
        cdef np.ndarray variant_dosages
        
        for batch_start in range(0, n_variants, batch_size):
            batch_end = min(batch_start + batch_size, n_variants)
            
            # Process batch of variants
            for i in range(batch_start, batch_end):
                # Read and process variant directly without storing DecompressedData
                # This avoids copy assignment issues with move-only types
                variant_dosages = self._read_and_parse_single_variant(variant_metadata[i], sample_indices)
                dosages[:, i] = variant_dosages
                
                # Progress callback
                if progress_callback is not None:
                    progress_callback(i + 1)
        
        # Create variant info DataFrame
        variant_info = self._create_variant_info(variant_metadata)
        
        return dosages, variant_info
    
    cdef np.ndarray _read_and_parse_single_variant(self, const VariantMetadata& metadata, 
                                                   np.ndarray sample_indices):
        """Read and parse a single variant without storing DecompressedData."""
        # This method combines reading and parsing to avoid DecompressedData copy issues
        # We let the C++ code handle the DecompressedData lifetime
        
        # Get sample filter info
        cdef uint32_t n_samples = self.header_info.nsamples
        cdef uint32_t n_samples_out
        cdef const uint32_t* sample_indices_ptr = NULL
        cdef uint32_t n_indices = 0
        
        if sample_indices is not None:
            n_samples_out = len(sample_indices)
            sample_indices_ptr = <const uint32_t*>np.PyArray_DATA(sample_indices)
            n_indices = n_samples_out
        else:
            n_samples_out = n_samples
        
        # Allocate output array
        cdef np.ndarray[np.float32_t, ndim=1] dosages = np.empty(n_samples_out, dtype=np.float32)
        
        # Read and parse in one go using a helper that doesn't return DecompressedData
        # Use the new API that returns unique_ptr
        cdef unique_ptr[DecompressedData] data_ptr
        cdef LayoutType layout_type = LayoutType_V11 if self.header_info.layout == 1 else LayoutType_V12
        cdef CompressionType comp_type = <CompressionType>self.header_info.compression
        
        # Get the decompressed data
        data_ptr = move(self.impl.get().read_variant_genotypes(metadata))
        
        if not data_ptr or not data_ptr.get().is_valid():
            error_msg = "Unknown error" if not data_ptr else data_ptr.get().error_message.decode('utf-8')
            raise RuntimeError(f"Failed to decompress variant at offset {metadata.file_offset}: {error_msg}")
        
        # Get parser buffer info
        cdef const uint8_t* parser_buffer = data_ptr.get().data()
        cdef size_t parser_size = data_ptr.get().size
        
        # Parse genotypes - data is already decompressed, so pass CompressionType.None
        if sample_indices_ptr != NULL:
            GenotypeParser.computeDosagesFiltered(
                parser_buffer,
                parser_size,
                layout_type,
                CompressionType_None,  # Data is already decompressed
                n_samples,
                metadata.n_alleles,
                <const int*>sample_indices_ptr,
                n_indices,
                <float*>np.PyArray_DATA(dosages)
            )
        else:
            GenotypeParser.computeDosagesDirect(
                parser_buffer,
                parser_size,
                layout_type,
                CompressionType_None,  # Data is already decompressed
                n_samples,
                metadata.n_alleles,
                <float*>np.PyArray_DATA(dosages)
            )
                
        return dosages
    
    cdef np.ndarray _parse_genotypes(self, const DecompressedData& data, 
                                    const VariantMetadata& metadata):
        """Parse genotype data to dosages."""
        cdef uint32_t n_samples = self.header_info.nsamples
        cdef uint32_t n_samples_out
        cdef const uint32_t* sample_indices_ptr = NULL
        cdef uint32_t n_indices = 0
        
        # Determine output size and sample filter
        if self.sample_filter is not None:
            n_samples_out = len(self.sample_filter)
            sample_indices_ptr = <const uint32_t*>np.PyArray_DATA(self.sample_filter)
            n_indices = n_samples_out
        else:
            n_samples_out = n_samples
        
        # Allocate output array
        cdef np.ndarray[np.float32_t, ndim=1] dosages = np.empty(n_samples_out, dtype=np.float32)
        
        # Parse genotypes
        # Need to handle layout conversion and determine compression
        cdef LayoutType layout_type = LayoutType_V11 if self.header_info.layout == 1 else LayoutType_V12
        cdef CompressionType comp_type = <CompressionType>self.header_info.compression
        
        if sample_indices_ptr != NULL:
            GenotypeParser.computeDosagesFiltered(
                data.data(),
                data.size,
                layout_type,
                comp_type,
                n_samples,
                metadata.n_alleles,
                <const int*>sample_indices_ptr,
                n_indices,
                <float*>np.PyArray_DATA(dosages)
            )
        else:
            GenotypeParser.computeDosagesDirect(
                data.data(),
                data.size,
                layout_type,
                comp_type,
                n_samples,
                metadata.n_alleles,
                <float*>np.PyArray_DATA(dosages)
            )
        
        return dosages
    
    def _create_variant_info(self, vector[VariantMetadata]& metadata) -> pd.DataFrame:
        """Create variant info DataFrame from metadata."""
        info_list = []
        cdef VariantMetadata var
        
        for var in metadata:
            info_list.append({
                'chrom': var.chrom.decode('utf-8'),
                'pos': var.pos,
                'rsid': var.rsid.decode('utf-8'),
                'ref': var.alleles[0].decode('utf-8') if var.alleles.size() > 0 else '',
                'alt': var.alleles[1].decode('utf-8') if var.alleles.size() > 1 else ''
            })
        
        return pd.DataFrame(info_list)
    
    def read_variant(self, offset: int) -> Variant:
        """
        Read a single variant at the given offset.
        
        Parameters
        ----------
        offset : int
            File offset of the variant
        
        Returns
        -------
        Variant
            Variant object
        """
        self._ensure_open()
        
        cdef VariantMetadata metadata = self.impl.get().read_variant_metadata(offset)
        variant = Variant()
        variant.metadata = metadata
        variant.reader = self
        variant._genotypes_loaded = False
        
        return variant
    
    cdef void _ensure_open(self) except *:
        """Ensure reader is open."""
        if not self.is_open:
            raise ValueError("BGEN reader is closed")
    
    def close(self):
        """Close the BGEN reader."""
        if self.is_open:
            if self.impl:
                self.impl.get().close()
            self.is_open = False
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def __dealloc__(self):
        """Cleanup."""
        self.close()
    
    # Properties
    @property
    def nsamples(self) -> int:
        """Number of samples."""
        return self.header_info.nsamples
    
    @property
    def nvariants(self) -> int:
        """Number of variants."""
        if self.bgi_reader:
            return self.bgi_reader.get().get_variant_count()
        return self.header_info.nvariants
    
    @property
    def samples(self) -> List[str]:
        """List of sample IDs."""
        return self.sample_ids
    
    @property
    def compression(self) -> str:
        """Compression type."""
        if self.header_info.compression == 0:
            return "none"
        elif self.header_info.compression == 1:
            return "zlib"
        elif self.header_info.compression == 2:
            return "zstd"
        else:
            return "unknown"
    
    @property
    def layout(self) -> int:
        """Layout version."""
        return self.header_info.layout
    
    def get_sample_indices(self, sample_ids: List[str]) -> Tuple[List[int], List[str]]:
        """
        Get indices of requested samples.
        
        Parameters
        ----------
        sample_ids : List[str]
            Sample IDs to find
        
        Returns
        -------
        Tuple[List[int], List[str]]
            (indices, found_sample_ids)
        """
        sample_map = {sid: i for i, sid in enumerate(self.sample_ids)}
        indices = []
        found_ids = []
        
        for sid in sample_ids:
            if sid in sample_map:
                indices.append(sample_map[sid])
                found_ids.append(sid)
        
        return indices, found_ids


cdef class Variant:
    """Represents a single BGEN variant."""
    
    @property
    def chrom(self) -> str:
        """Chromosome."""
        return self.metadata.chrom.decode('utf-8')
    
    @property
    def pos(self) -> int:
        """Position."""
        return self.metadata.pos
    
    @property
    def rsid(self) -> str:
        """RS ID."""
        return self.metadata.rsid.decode('utf-8')
    
    @property
    def varid(self) -> str:
        """Variant ID."""
        return self.metadata.varid.decode('utf-8')
    
    @property
    def alleles(self) -> List[str]:
        """List of alleles."""
        return [a.decode('utf-8') for a in self.metadata.alleles]
    
    @property
    def ref(self) -> str:
        """Reference allele."""
        if self.metadata.alleles.size() > 0:
            return self.metadata.alleles[0].decode('utf-8')
        return ""
    
    @property
    def alt(self) -> str:
        """Alternate allele."""
        if self.metadata.alleles.size() > 1:
            return self.metadata.alleles[1].decode('utf-8')
        return ""
    
    @property
    def n_alleles(self) -> int:
        """Number of alleles."""
        return self.metadata.n_alleles
    
    @property
    def alt_dosage(self) -> np.ndarray:
        """Get alt allele dosages for all samples."""
        if not self._genotypes_loaded:
            self._load_genotypes()
        return self._dosages
    
    def get_dosage_for_samples(self, sample_indices: Union[List[int], np.ndarray]) -> np.ndarray:
        """
        Get dosages for specific samples.
        
        Parameters
        ----------
        sample_indices : array-like
            Indices of samples to extract
        
        Returns
        -------
        np.ndarray
            Dosages for requested samples
        """
        if not self._genotypes_loaded:
            self._load_genotypes()
        
        # Convert to numpy array if needed
        if not isinstance(sample_indices, np.ndarray):
            sample_indices = np.array(sample_indices, dtype=np.int32)
        
        return self._dosages[sample_indices]
    
    cdef void _load_genotypes(self) except *:
        """Load and parse genotype data."""
        if self._genotypes_loaded:
            return
        
        # Read and parse directly without storing DecompressedData
        self._dosages = self.reader._read_and_parse_single_variant(self.metadata, None)
        self._genotypes_loaded = True
    
