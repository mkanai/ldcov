# cython: language_level=3
# Cython declarations for batch and sequential decompressors

from libc.stdint cimport uint8_t, uint32_t, uint64_t
from libcpp.vector cimport vector
from libcpp cimport bool
from libcpp.memory cimport unique_ptr, shared_ptr
from libcpp.string cimport string

# Forward declare C++ decompressor classes
cdef extern from "decompression_result.h" namespace "ldcov::bgen":
    cdef cppclass DecompressionResult:
        uint64_t offset
        const uint8_t* data
        size_t size
        bool success
        uint8_t error_code

cdef extern from "batch_decompressor.h" namespace "ldcov::bgen":
    
    cdef cppclass DecompressionTask:
        uint64_t offset
        uint32_t compressed_size
        uint32_t expected_uncompressed_size
        uint8_t compression_type
        
    cdef cppclass BatchDecompressor:
        BatchDecompressor(int num_threads, int queue_size) except +
        
        # Submit a batch of tasks
        void submit_batch(const vector[DecompressionTask]& tasks, const string& filename) except +
        
        # Get completed results (blocking)
        vector[DecompressionResult] get_results(int count) except +
        
        # Check if results are ready (non-blocking)
        bool has_results(int count) const
        
        # Shutdown and cleanup
        void shutdown()
        
        # Get statistics
        uint64_t total_bytes_read() const
        uint64_t total_bytes_decompressed() const
        uint64_t total_tasks_completed() const
        double average_decompression_time_ms() const

cdef extern from "sequential_decompressor.h" namespace "ldcov::bgen":
    cdef cppclass SequentialDecompressor:
        SequentialDecompressor(const string& filename) except +
        
        # Single variant decompression
        DecompressionResult decompress_variant(
            uint64_t offset,
            uint32_t compressed_size,
            uint32_t expected_size,
            uint8_t compression_type
        )
        
        # Single variant decompression with allocated buffer (safe for batch)
        DecompressionResult decompress_variant_allocated(
            uint64_t offset,
            uint32_t compressed_size,
            uint32_t expected_size,
            uint8_t compression_type
        )
        
        # Batch sequential decompression
        vector[DecompressionResult] decompress_sequential(
            const vector[uint64_t]& offsets,
            const vector[uint32_t]& compressed_sizes,
            const vector[uint32_t]& expected_sizes,
            const vector[uint8_t]& compression_types,
            bool enable_readahead
        )
        
        # Check if sequential pattern
        @staticmethod
        bool is_sequential_pattern(const vector[uint64_t]& offsets, uint64_t max_gap)
        
        # Statistics
        uint64_t total_bytes_read() const
        uint64_t total_bytes_decompressed() const
        uint64_t total_variants_processed() const


# Cython wrapper classes
cdef class CyBatchDecompressor:
    cdef unique_ptr[BatchDecompressor] decompressor
    cdef int num_threads
    cdef int queue_size
    cdef bool is_active
    
    # Internal methods
    cdef void _ensure_active(self) except *


cdef class CySequentialDecompressor:
    cdef unique_ptr[SequentialDecompressor] decompressor
    cdef string filename
    cdef bool is_active
    
    # Internal methods
    cdef void _ensure_active(self) except *