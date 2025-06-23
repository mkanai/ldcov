#ifndef LDCOV_BGEN_SEQUENTIAL_DECOMPRESSOR_H
#define LDCOV_BGEN_SEQUENTIAL_DECOMPRESSOR_H

#include <string>
#include <vector>
#include <cstdint>
#include <memory>
#include <future>
#include <fcntl.h>

#include "decompression_result.h"
#include "buffer_pool.h"

namespace ldcov {
namespace bgen {

// Optimized sequential decompressor for contiguous variant access
class SequentialDecompressor {
private:
    // File handling - direct file descriptor for zero-copy I/O
    std::string filename_;
    int fd_;
    uint64_t file_size_;
    
    // Read-ahead buffer for sequential optimization
    struct ReadAheadBuffer {
        std::future<ssize_t> future;
        std::unique_ptr<uint8_t[]> buffer;
        uint64_t offset;
        size_t size;
        bool is_active;
        
        ReadAheadBuffer() : offset(0), size(0), is_active(false) {}
    };
    std::unique_ptr<ReadAheadBuffer> read_ahead_;
    
    // Statistics
    uint64_t total_bytes_read_;
    uint64_t total_bytes_decompressed_;
    uint64_t total_variants_processed_;
    
    // Helper methods
    ssize_t read_at_offset(uint64_t offset, uint8_t* buffer, size_t size);
    void start_read_ahead(uint64_t offset, size_t size);
    ssize_t complete_read_ahead();
    
    // Inline decompression functions for performance
    inline bool decompress_zlib_inline(const uint8_t* compressed_data,
                                       size_t compressed_size,
                                       uint8_t* output_buffer,
                                       size_t expected_size,
                                       size_t* actual_size);
    
    inline bool decompress_zstd_inline(const uint8_t* compressed_data,
                                       size_t compressed_size,
                                       uint8_t* output_buffer,
                                       size_t expected_size,
                                       size_t* actual_size);
    
public:
    explicit SequentialDecompressor(const std::string& filename);
    ~SequentialDecompressor();
    
    // Delete copy operations
    SequentialDecompressor(const SequentialDecompressor&) = delete;
    SequentialDecompressor& operator=(const SequentialDecompressor&) = delete;
    
    // Move operations
    SequentialDecompressor(SequentialDecompressor&& other) noexcept;
    SequentialDecompressor& operator=(SequentialDecompressor&& other) noexcept;
    
    // Single variant decompression - returns zero-copy result
    DecompressionResult decompress_variant(
        uint64_t offset,
        uint32_t compressed_size,
        uint32_t expected_size,
        uint8_t compression_type
    );
    
    // Single variant decompression with allocated buffer - for safe batch processing
    // Returns a result with data pointing to newly allocated memory that caller must free
    DecompressionResult decompress_variant_allocated(
        uint64_t offset,
        uint32_t compressed_size,
        uint32_t expected_size,
        uint8_t compression_type
    );
    
    // Batch sequential decompression with read-ahead optimization
    std::vector<DecompressionResult> decompress_sequential(
        const std::vector<uint64_t>& offsets,
        const std::vector<uint32_t>& compressed_sizes,
        const std::vector<uint32_t>& expected_sizes,
        const std::vector<uint8_t>& compression_types,
        bool enable_readahead = true
    );
    
    // Check if offsets suggest sequential access pattern
    static bool is_sequential_pattern(const std::vector<uint64_t>& offsets,
                                     uint64_t max_gap = 100 * 1024);  // 100KB default
    
    // Statistics
    uint64_t total_bytes_read() const { return total_bytes_read_; }
    uint64_t total_bytes_decompressed() const { return total_bytes_decompressed_; }
    uint64_t total_variants_processed() const { return total_variants_processed_; }
    double read_amplification() const {
        return total_bytes_decompressed_ > 0 ? 
            static_cast<double>(total_bytes_read_) / total_bytes_decompressed_ : 0.0;
    }
};

} // namespace bgen
} // namespace ldcov

#endif // LDCOV_BGEN_SEQUENTIAL_DECOMPRESSOR_H