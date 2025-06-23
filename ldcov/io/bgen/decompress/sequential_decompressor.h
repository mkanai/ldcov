#ifndef LDCOV_BGEN_DECOMPRESS_SEQUENTIAL_DECOMPRESSOR_H
#define LDCOV_BGEN_DECOMPRESS_SEQUENTIAL_DECOMPRESSOR_H

#include <memory>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <future>

#include "decompressor.h"
#include "buffer_manager.h"
#include "compression_utils.h"
#include "../io/reader_interface.h"

namespace ldcov {
namespace bgen {
namespace decompress {

// Import FileReader into this namespace
using FileReader = ::ldcov::io::bgen::FileReader;

/**
 * SequentialDecompressor - Optimized decompressor for sequential access patterns
 * 
 * This decompressor is optimized for reading variants in sequential order from
 * BGEN files. It uses read-ahead buffering to prefetch the next variant while
 * processing the current one, minimizing I/O wait times.
 * 
 * Key features:
 * - Single-threaded design with async I/O for simplicity and safety
 * - Read-ahead buffer for next variant
 * - Efficient buffer reuse through BufferManager
 * - Direct decompression into output buffers when possible
 * - Minimal memory allocations during operation
 */
class SequentialDecompressor : public VariantDecompressor {
public:
    // Extended configuration for sequential decompressor
    struct SequentialConfig : public VariantDecompressor::Config {
        // Enable read-ahead prefetching
        bool enable_readahead = true;
        
        // Size of read-ahead buffer (bytes)
        size_t readahead_buffer_size = 4 * 1024 * 1024; // 4MB
        
        // Maximum distance between variants to consider sequential (bytes)
        size_t sequential_threshold = 1024 * 1024; // 1MB
        
        // File reader to use (required)
        FileReader* file_reader = nullptr;
    };
    
    /**
     * Constructor
     * @param config Configuration for the decompressor
     */
    explicit SequentialDecompressor(const SequentialConfig& config);
    
    /**
     * Destructor
     */
    ~SequentialDecompressor() override;
    
    // Delete copy operations
    SequentialDecompressor(const SequentialDecompressor&) = delete;
    SequentialDecompressor& operator=(const SequentialDecompressor&) = delete;
    
    // Delete move operations for simplicity
    SequentialDecompressor(SequentialDecompressor&&) = delete;
    SequentialDecompressor& operator=(SequentialDecompressor&&) = delete;
    
    /**
     * Decompress a single variant
     * 
     * @param variant Compressed variant data
     * @return Decompressed data
     */
    DecompressedData decompress(const CompressedVariant& variant) override;
    
    /**
     * Decompress multiple variants in batch
     * 
     * For sequential access, this method can optimize by prefetching
     * subsequent variants while processing current ones.
     * 
     * @param variants Vector of compressed variants
     * @return Vector of decompressed data
     */
    std::vector<DecompressedData> decompress_batch(
        const std::vector<CompressedVariant>& variants) override;
    
    /**
     * Get decompression statistics
     * @return Current statistics
     */
    Statistics get_statistics() const override;
    
    /**
     * Reset statistics counters
     */
    void reset_statistics() override;
    
    /**
     * Check if a set of variants follows a sequential access pattern
     * 
     * @param variants Variants to check
     * @return True if access pattern is sequential
     */
    bool is_sequential_pattern(const std::vector<CompressedVariant>& variants) const;
    
private:
    // Configuration
    SequentialConfig config_;
    
    // Buffer manager (owned if not provided)
    std::unique_ptr<BufferManager> owned_buffer_manager_;
    BufferManager* buffer_manager_;
    
    // File reader (not owned)
    FileReader* file_reader_;
    
    // Double-buffering state for synchronous prefetch
    struct DoubleBuffer {
        // Two buffers for ping-pong operation
        BufferHandle buffer1;
        BufferHandle buffer2;
        
        // Current active buffer (0 or 1)
        int active_buffer;
        
        // Metadata for each buffer
        struct BufferInfo {
            uint64_t offset;
            size_t size;
            bool valid;
            
            BufferInfo() : offset(0), size(0), valid(false) {}
        };
        BufferInfo info[2];
        
        // Constructor
        DoubleBuffer() 
            : buffer1(std::unique_ptr<uint8_t[]>(nullptr), 0, nullptr, nullptr)
            , buffer2(std::unique_ptr<uint8_t[]>(nullptr), 0, nullptr, nullptr)
            , active_buffer(0)
        {}
        
        // Get current buffer
        BufferHandle& current() { 
            return active_buffer == 0 ? buffer1 : buffer2; 
        }
        
        // Get next buffer
        BufferHandle& next() { 
            return active_buffer == 0 ? buffer2 : buffer1; 
        }
        
        // Swap buffers
        void swap() { 
            active_buffer = 1 - active_buffer; 
        }
        
        // Clear all state
        void clear() {
            info[0] = BufferInfo();
            info[1] = BufferInfo();
            active_buffer = 0;
            // Buffers will be cleared automatically when reassigned
        }
    };
    DoubleBuffer double_buffer_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    Statistics stats_;
    
    // Helper methods
    
    /**
     * Prefetch next variant data into the next buffer
     * 
     * @param offset File offset to read from
     * @param size Number of bytes to read
     * @return True if prefetch was successful
     */
    bool prefetch_next(uint64_t offset, size_t size);
    
    /**
     * Check if we have prefetched data for the given offset
     * 
     * @param offset Expected offset
     * @param size Expected size
     * @return True if prefetched data is available
     */
    bool has_prefetched_data(uint64_t offset, size_t size) const;
    
    /**
     * Read compressed data for a variant
     * 
     * This will use read-ahead buffer if available, otherwise read directly.
     * 
     * @param variant Variant to read
     * @return Buffer containing compressed data
     */
    BufferHandle read_compressed_data(const CompressedVariant& variant);
    
    /**
     * Decompress data using the appropriate algorithm
     * 
     * @param compressed Compressed data
     * @param compressed_size Size of compressed data
     * @param expected_size Expected uncompressed size
     * @param compression_type Type of compression
     * @return Decompressed data
     */
    DecompressedData decompress_data(const uint8_t* compressed,
                                     size_t compressed_size,
                                     size_t expected_size,
                                     CompressionType compression_type,
                                     uint64_t offset);
    
    /**
     * Update statistics
     * 
     * @param variant Processed variant
     * @param success Whether decompression was successful
     */
    void update_statistics(const CompressedVariant& variant, bool success);
    
    // Removed separate fast path - integrated directly into main path
    
    // Persistent decompression contexts to avoid recreation
    struct DecompressionContexts {
        void* zlib_stream;      // z_stream for zlib
        void* zstd_context;     // ZSTD_DCtx for zstd
        
        DecompressionContexts() : zlib_stream(nullptr), zstd_context(nullptr) {}
        ~DecompressionContexts();
        
        // Initialize contexts if needed
        bool ensure_zlib();
        bool ensure_zstd();
    };
    DecompressionContexts decomp_contexts_;
    
    // Direct decompression buffer for fast path
    std::unique_ptr<uint8_t[]> direct_read_buffer_;
    size_t direct_read_buffer_size_;
    
    // Performance optimization flags
    bool enable_prefetch_;
    size_t prefetch_distance_;
    
    // Helper for direct decompression
    DecompressedData decompress_direct(const CompressedVariant& variant);
    
    // Optimized path for uncompressed data
    DecompressedData decompress_uncompressed_direct(const CompressedVariant& variant);
    
    // CPU prefetch helpers
    inline void prefetch_read(const void* addr) {
#ifdef __builtin_prefetch
        if (enable_prefetch_) {
            __builtin_prefetch(addr, 0, 1);  // Read, low temporal locality
        }
#endif
    }
    
    inline void prefetch_write(void* addr) {
#ifdef __builtin_prefetch
        if (enable_prefetch_) {
            __builtin_prefetch(addr, 1, 1);  // Write, low temporal locality
        }
#endif
    }
};

} // namespace decompress
} // namespace bgen
} // namespace ldcov

#endif // LDCOV_BGEN_DECOMPRESS_SEQUENTIAL_DECOMPRESSOR_H