#ifndef LDCOV_BGEN_DECOMPRESS_BUFFER_MANAGER_H
#define LDCOV_BGEN_DECOMPRESS_BUFFER_MANAGER_H

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <vector>

namespace ldcov {
namespace bgen {
namespace decompress {

// Forward declaration
class BufferManager;

/**
 * BufferHandle - RAII wrapper for a buffer that automatically returns it to the pool
 *
 * This class ensures that buffers are properly returned to the pool even in the
 * presence of exceptions. The buffer is owned by this handle and will be returned
 * to the pool when the handle is destroyed.
 */
class BufferHandle {
   public:
    // Constructor - takes ownership of buffer
    BufferHandle(std::unique_ptr<uint8_t[]> buffer, size_t size, BufferManager* manager,
                 std::function<void(std::unique_ptr<uint8_t[]>, size_t)> return_func);

    // Move constructor
    BufferHandle(BufferHandle&& other) noexcept;

    // Move assignment
    BufferHandle& operator=(BufferHandle&& other) noexcept;

    // Destructor - returns buffer to pool
    ~BufferHandle();

    // Delete copy operations
    BufferHandle(const BufferHandle&) = delete;
    BufferHandle& operator=(const BufferHandle&) = delete;

    // Access the buffer
    uint8_t* data() {
        return buffer_.get();
    }
    const uint8_t* data() const {
        return buffer_.get();
    }

    // Get buffer size
    size_t size() const {
        return size_;
    }

    // Check if handle owns a buffer
    bool valid() const {
        return buffer_ != nullptr;
    }

    // Release ownership without returning to pool
    std::unique_ptr<uint8_t[]> release();

   private:
    std::unique_ptr<uint8_t[]> buffer_;
    size_t size_;
    BufferManager* manager_;
    std::function<void(std::unique_ptr<uint8_t[]>, size_t)> return_func_;
};

/**
 * BufferManager - Thread-safe buffer pool manager
 *
 * This class manages a pool of reusable buffers to avoid frequent allocations.
 * It provides thread-safe buffer allocation and automatic return through RAII.
 * Unlike the previous thread-local design, this uses explicit ownership through
 * unique_ptr and BufferHandle to prevent memory corruption.
 */
class BufferManager {
   public:
    // Configuration for buffer manager
    struct Config {
        size_t initial_buffer_size;  // 1MB default
        size_t max_buffer_size;      // 256MB max
        size_t max_pool_size;        // Max buffers to keep in pool
        double growth_factor;        // Growth factor when resizing
        bool enable_statistics;      // Track allocation statistics

        // Constructor with defaults
        Config()
            : initial_buffer_size(1024 * 1024),
              max_buffer_size(256 * 1024 * 1024),
              max_pool_size(16),
              growth_factor(2.0),
              enable_statistics(false) {}
    };

    // Constructor with configuration
    explicit BufferManager(const Config& config = Config());

    // Destructor
    ~BufferManager();

    // Get a buffer of at least the requested size
    // Returns a BufferHandle that automatically returns the buffer when destroyed
    BufferHandle get_buffer(size_t required_size);

    // Get buffer for decompression (semantic alias)
    BufferHandle get_decompression_buffer(size_t required_size) {
        return get_buffer(required_size);
    }

    // Get buffer for compressed data (semantic alias)
    BufferHandle get_work_buffer(size_t required_size) {
        return get_buffer(required_size);
    }

    // Statistics
    struct Statistics {
        std::atomic<size_t> total_allocations{0};
        std::atomic<size_t> total_deallocations{0};
        std::atomic<size_t> pool_hits{0};
        std::atomic<size_t> pool_misses{0};
        std::atomic<size_t> current_pool_size{0};
        std::atomic<size_t> total_memory_allocated{0};
        std::atomic<size_t> peak_memory_allocated{0};
    };

    // Get statistics (if enabled)
    const Statistics& get_statistics() const {
        return stats_;
    }

    // Clear the pool (useful for testing or memory pressure)
    void clear_pool();

   private:
    // Pool entry
    struct PoolEntry {
        std::unique_ptr<uint8_t[]> buffer;
        size_t size;

        PoolEntry(std::unique_ptr<uint8_t[]> buf, size_t sz) : buffer(std::move(buf)), size(sz) {}
    };

    // Return a buffer to the pool
    void return_buffer(std::unique_ptr<uint8_t[]> buffer, size_t size);

    // Friend class for access to return_buffer
    friend class BufferHandle;

    // Configuration
    Config config_;

    // Buffer pool
    std::vector<PoolEntry> pool_;
    std::mutex pool_mutex_;

    // Statistics
    Statistics stats_;

    // Track current memory usage
    std::atomic<size_t> current_memory_usage_{0};
};

// Global buffer manager instance (optional, for convenience)
// Applications can create their own instances if needed
BufferManager& get_global_buffer_manager();

}  // namespace decompress
}  // namespace bgen
}  // namespace ldcov

#endif  // LDCOV_BGEN_DECOMPRESS_BUFFER_MANAGER_H