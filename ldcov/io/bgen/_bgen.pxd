# cython: language_level=3
# Cython declarations for BGEN module

from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t, int32_t
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool

# BGEN file format constants
cdef enum:
    BGEN_COMPRESSED_ZLIB = 1
    BGEN_COMPRESSED_ZSTD = 2
    BGEN_LAYOUT_V11 = 1
    BGEN_LAYOUT_V12 = 2

# Declare C++ structures
cdef extern from *:
    """
    #include <cstdint>
    #include <string>
    #include <vector>
    """

# Header information structure
cdef struct HeaderInfo:
    uint32_t offset
    uint32_t nvariants  
    uint32_t nsamples
    uint32_t flags
    int compression
    int layout
    bool has_sample_ids

# Variant information structure
cdef struct VariantInfo:
    uint64_t file_offset
    string varid
    string rsid
    string chrom
    uint32_t pos
    uint16_t n_alleles
    vector[string] alleles
    uint32_t geno_offset
    uint32_t geno_length

# C++ decompression functions
cdef extern from "decompress.h":
    int decompress_zlib(const uint8_t* compressed, size_t compressed_size,
                       uint8_t* decompressed, size_t* decompressed_size) nogil
    int decompress_zstd(const uint8_t* compressed, size_t compressed_size,
                       uint8_t* decompressed, size_t* decompressed_size) nogil

# Forward declaration - implementation in genotypes.pyx
cdef class GenotypeData:
    cdef:
        uint32_t n_samples
        uint16_t n_alleles
        uint8_t* raw_data
        uint32_t data_length
        int compression
        bool phased
        uint8_t* ploidy
        float* probs
        bool _initialized
        bool constant_ploidy
        uint8_t max_ploidy
        bool has_missing
    
    cdef void parse_layout1(self, uint8_t* data, uint32_t length)
    cdef void parse_layout2(self, uint8_t* data, uint32_t length)
    cdef void decompress_data(self, bint has_length_prefix=*)
    cdef void compute_dosages(self, float* output) nogil
    cdef void compute_dosages_filtered(self, int* sample_indices, int n_indices, float* output) nogil


# C memory allocation functions
cdef extern from "Python.h":
    void* PyMem_Malloc(size_t n)
    void PyMem_Free(void* p)