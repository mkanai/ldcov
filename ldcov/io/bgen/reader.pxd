# cython: language_level=3
# Cython declarations for new BGEN reader implementation

from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t, int32_t
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.memory cimport unique_ptr, shared_ptr
from libcpp.utility cimport move
from libcpp cimport bool
from libcpp.unordered_map cimport unordered_map

# Import numpy array API
cimport numpy as np

# Forward declarations for decompressor types
cdef extern from "decompress/decompressor.h" namespace "ldcov::bgen::decompress":
    cdef enum CompressionType:
        CompressionType_None "ldcov::bgen::decompress::CompressionType::None"
        CompressionType_Zlib "ldcov::bgen::decompress::CompressionType::Zlib"
        CompressionType_Zstd "ldcov::bgen::decompress::CompressionType::Zstd"
        CompressionType_Unknown "ldcov::bgen::decompress::CompressionType::Unknown"
    
    cdef cppclass CompressedVariant:
        uint64_t offset
        const uint8_t* data
        size_t compressed_size
        size_t uncompressed_size
        CompressionType compression_type
        string variant_id
        
        CompressedVariant(uint64_t, const uint8_t*, size_t, size_t, CompressionType)
    
    cdef cppclass DecompressedData:
        uint8_t* data() const
        size_t size
        uint64_t offset
        bool success
        bool is_valid() const
        string error_message
    
    cdef cppclass VariantDecompressor:
        pass
    
    # Factory functions
    unique_ptr[VariantDecompressor] create_adaptive_decompressor(void*, const void*) except +
    unique_ptr[VariantDecompressor] create_sequential_decompressor(void*, const void*, bool) except +
    unique_ptr[VariantDecompressor] create_parallel_decompressor(void*, size_t, const void*) except +

# File reader interface
cdef extern from "io/reader_interface.h" namespace "ldcov::io::bgen":
    cdef cppclass FileReader:
        size_t read(uint8_t* buffer, size_t size) except +
        size_t read_at(uint64_t offset, uint8_t* buffer, size_t size) except +
        void seek(uint64_t offset) except +
        uint64_t tell() const
        uint64_t size() const
        bool is_open() const
        void close()
        const string& filename() const

# File reader wrapper for Python objects
cdef extern from "file_reader_wrapper.h" namespace "ldcov::io::bgen":
    cdef cppclass FileReaderWrapper(FileReader):
        FileReaderWrapper(object py_file) except +

# BGEN structures from C++
cdef extern from "bgen_reader_impl.h" namespace "ldcov::io::bgen":
    cdef struct BgenHeader:
        uint32_t offset
        uint32_t nvariants
        uint32_t nsamples
        uint32_t flags
        uint8_t compression
        uint8_t layout
        bool has_sample_ids

cdef extern from "format/variant_parser.h" namespace "ldcov::bgen":
    cdef struct VariantMetadata:
        uint64_t file_offset
        string varid
        string rsid
        string chrom
        uint32_t pos
        uint16_t n_alleles
        vector[string] alleles
        uint64_t genotype_offset
        uint32_t genotype_length

# BGI variant info structure
cdef extern from "bgi_reader.h" namespace "ldcov::io::bgen::index":
    cdef struct VariantInfo:
        uint64_t file_offset
        uint32_t variant_size
        string chromosome
        uint32_t position
        string rsid
        string varid
        uint16_t n_alleles
        string allele1
        string allele2

# BGI index reader
cdef extern from "bgi_reader.h" namespace "ldcov::io::bgen::index":
    cdef cppclass BGIReader:
        BGIReader(const string& filename) except +
        BGIReader(const string& filename, size_t cache_size) except +
        
        # Query methods
        vector[VariantInfo] query_region(
            const string& chromosome, uint32_t start_pos, uint32_t end_pos) except +
        vector[VariantInfo] query_position(
            const string& chromosome, uint32_t position) except +
        vector[VariantInfo] query_variant_id(const string& variant_id) except +
        VariantInfo get_variant(size_t index) except +
        vector[VariantInfo] find_variants_by_filter(
            const string& chromosome,
            const vector[uint32_t]& positions,
            const vector[string]& alleles1,
            const vector[string]& alleles2,
            size_t batch_size) except +
        vector[VariantInfo] get_all_variants() except +
        
        # Index info
        size_t get_variant_count() const
        bool is_open() const
        void close()

# Genotype parser
cdef extern from "format/genotype_parser.h" namespace "ldcov::bgen":
    cdef enum LayoutType:
        LayoutType_V11 "ldcov::bgen::LayoutType::V11"
        LayoutType_V12 "ldcov::bgen::LayoutType::V12"
    
    cdef enum CompressionType:
        CompressionType_None "ldcov::bgen::CompressionType::None"
        CompressionType_Zlib "ldcov::bgen::CompressionType::Zlib"
        CompressionType_Zstd "ldcov::bgen::CompressionType::Zstd"
        CompressionType_Unknown "ldcov::bgen::CompressionType::Unknown"
    
    cdef cppclass GenotypeParser:
        @staticmethod
        void computeDosagesFiltered(
            const uint8_t* buffer,
            size_t size,
            LayoutType layout,
            CompressionType compression,
            uint32_t n_samples,
            uint16_t n_alleles,
            const int* sample_indices,
            int n_indices,
            float* output
        ) except +
        
        @staticmethod
        void computeDosagesDirect(
            const uint8_t* buffer,
            size_t size,
            LayoutType layout,
            CompressionType compression,
            uint32_t n_samples,
            uint16_t n_alleles,
            float* output
        ) except +

# Main BGEN reader class
cdef extern from "bgen_reader_impl.h" namespace "ldcov::io::bgen":
    cdef cppclass BgenReaderImpl:
        BgenReaderImpl(const string& filename, const string& bgi_filename) except +
        
        # Header access
        const BgenHeader& header() const
        
        # Sample access
        vector[string] get_sample_ids() except +
        void set_sample_filter(const vector[uint32_t]& indices) except +
        
        # Variant access
        VariantMetadata read_variant_metadata(uint64_t offset) except +
        unique_ptr[DecompressedData] read_variant_genotypes(const VariantMetadata& metadata) except +
        
        # Batch operations
        vector[unique_ptr[DecompressedData]] read_variants_batch(
            const vector[VariantMetadata]& variants) except +
        
        # Decompressor configuration
        void set_decompressor_type(const string& type) except +
        void set_num_threads(size_t n) except +
        
        # File info
        bool is_open() const
        void close()

# Genotype parsing
cdef extern from "genotype_parser.h" namespace "ldcov::io::bgen":
    cdef cppclass GenotypeParser:
        @staticmethod
        void parse_genotypes_to_dosages(
            const uint8_t* data,
            size_t size,
            uint8_t layout,
            uint32_t nsamples,
            uint16_t nalleles,
            float* output,
            const uint32_t* sample_indices,
            uint32_t n_indices
        ) except +

# Cython wrapper classes
cdef class BgenReader:
    cdef unique_ptr[BgenReaderImpl] impl
    cdef unique_ptr[BGIReader] bgi_reader
    cdef BgenHeader header_info
    cdef unique_ptr[VariantDecompressor] decompressor
    cdef list sample_ids
    cdef bool is_open
    cdef str file_path
    cdef str bgi_path
    cdef object sample_filter  # numpy array of indices
    
    # Private methods
    cdef void _init_reader(self) except *
    cdef void _load_samples_from_bgen(self) except *
    cdef void _ensure_open(self) except *
    cdef vector[VariantMetadata] _get_filtered_variants(self, variant_filter) except *
    cdef tuple _load_variants_from_metadata(
        self, vector[VariantMetadata]& metadata, np.ndarray sample_indices,
        dtype, progress_callback)
    cdef np.ndarray _read_and_parse_single_variant(self, const VariantMetadata& metadata,
                                                   np.ndarray sample_indices)
    cdef np.ndarray _parse_genotypes(self, const DecompressedData& data, 
                                    const VariantMetadata& metadata)

cdef class Variant:
    """Represents a single BGEN variant with metadata and genotype data."""
    cdef VariantMetadata metadata
    cdef BgenReader reader  # Reference to parent reader
    cdef bool _genotypes_loaded
    cdef np.ndarray _dosages  # Cached dosages
    
    # Internal methods
    cdef void _load_genotypes(self) except *