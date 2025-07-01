#ifndef LDCOV_BGEN_DECOMPRESS_SEQUENTIAL_FAST_PATH_H
#define LDCOV_BGEN_DECOMPRESS_SEQUENTIAL_FAST_PATH_H

#include <cstdint>
#include <cstring>
#include <memory>

#include "../io/reader_interface.h"
#include "decompressor.h"  // This defines CompressionType in decompress namespace

namespace ldcov {
namespace bgen {
namespace decompress {

/**
 * Fast path implementation for sequential decompression.
 *
 * This class provides optimized sequential decompression with:
 * - No virtual function overhead
 * - Reusable buffers (no allocations per variant)
 * - Direct decompression into output
 * - CPU prefetching
 * - Minimal abstraction layers
 *
 * Designed to match or exceed Python implementation performance.
 */
class SequentialFastPath {
   public:
    // Configuration for fast path
    struct Config {
        size_t initial_buffer_size;
        bool enable_prefetch;
        size_t prefetch_distance;

        // Constructor with defaults
        Config()
            : initial_buffer_size(4 * 1024 * 1024),  // 4MB
              enable_prefetch(true),
              prefetch_distance(64) {}  // Cache line prefetch distance
    };

    explicit SequentialFastPath(ldcov::io::bgen::FileReader* reader,
                                const Config& config = Config());
    ~SequentialFastPath();

    // Delete copy/move operations
    SequentialFastPath(const SequentialFastPath&) = delete;
    SequentialFastPath& operator=(const SequentialFastPath&) = delete;
    SequentialFastPath(SequentialFastPath&&) = delete;
    SequentialFastPath& operator=(SequentialFastPath&&) = delete;

    /**
     * Fast decompression directly into output buffer.
     *
     * @param file_offset Offset in file where compressed data starts
     * @param compressed_size Size of compressed data
     * @param uncompressed_size Expected size after decompression
     * @param compression_type Type of compression
     * @param output Pre-allocated output buffer (must be at least uncompressed_size)
     * @return True if successful, false on error
     */
    bool decompress_direct(uint64_t file_offset, size_t compressed_size, size_t uncompressed_size,
                           CompressionType compression_type, uint8_t* output);

    /**
     * Fast batch decompression for multiple sequential variants.
     *
     * @param offsets File offsets for each variant
     * @param compressed_sizes Compressed sizes for each variant
     * @param uncompressed_sizes Uncompressed sizes for each variant
     * @param compression_types Compression type for each variant
     * @param outputs Pre-allocated output buffers for each variant
     * @param n_variants Number of variants to decompress
     * @return Number of successfully decompressed variants
     */
    size_t decompress_batch_direct(const uint64_t* offsets, const size_t* compressed_sizes,
                                   const size_t* uncompressed_sizes,
                                   const CompressionType* compression_types, uint8_t** outputs,
                                   size_t n_variants);

    /**
     * Get current error message if last operation failed.
     */
    const char* get_last_error() const {
        return last_error_;
    }

    /**
     * Reset internal buffers and state.
     */
    void reset();

   private:
    // File reader (not owned)
    ldcov::io::bgen::FileReader* file_reader_;

    // Configuration
    Config config_;

    // Reusable buffers to avoid allocations
    std::unique_ptr<uint8_t[]> read_buffer_;
    size_t read_buffer_size_;

    // Cached compression state (for zlib/zstd)
    void* zlib_stream_;
    void* zstd_dctx_;

    // Error handling
    static constexpr size_t ERROR_MSG_SIZE = 256;
    char last_error_[ERROR_MSG_SIZE];

    // Helper methods
    void ensure_read_buffer(size_t size);
    bool decompress_variant(const uint8_t* compressed, size_t compressed_size,
                            size_t uncompressed_size, CompressionType type, uint8_t* output);

    // CPU prefetch helper
    inline void prefetch_read(const void* addr) {
#ifdef __builtin_prefetch
        if (config_.enable_prefetch) {
            __builtin_prefetch(addr, 0, 1);  // Read, low temporal locality
        }
#endif
    }

    inline void prefetch_write(void* addr) {
#ifdef __builtin_prefetch
        if (config_.enable_prefetch) {
            __builtin_prefetch(addr, 1, 1);  // Write, low temporal locality
        }
#endif
    }
};

}  // namespace decompress
}  // namespace bgen
}  // namespace ldcov

#endif  // LDCOV_BGEN_DECOMPRESS_SEQUENTIAL_FAST_PATH_H