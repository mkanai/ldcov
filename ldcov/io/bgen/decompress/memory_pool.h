#ifndef LDCOV_IO_BGEN_DECOMPRESS_MEMORY_POOL_H
#define LDCOV_IO_BGEN_DECOMPRESS_MEMORY_POOL_H

#include <algorithm>
#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>
#include <queue>
#include <vector>

namespace ldcov {
namespace io {
namespace bgen {

/**
 * High-performance memory pool with size buckets for reduced fragmentation
 */
class MemoryPool {
   public:
    struct Config {
        size_t small_buffer_size;
        size_t medium_buffer_size;
        size_t large_buffer_size;

        size_t max_small_buffers;
        size_t max_medium_buffers;
        size_t max_large_buffers;

        bool pre_allocate;
        bool zero_on_acquire;

        // Constructor with defaults
        Config()
            : small_buffer_size(64 * 1024),
              medium_buffer_size(1024 * 1024),
              large_buffer_size(16 * 1024 * 1024),
              max_small_buffers(32),
              max_medium_buffers(16),
              max_large_buffers(4),
              pre_allocate(true),
              zero_on_acquire(false) {}
    };

    explicit MemoryPool(const Config& config = Config());
    ~MemoryPool();

    // Acquire buffer of at least the requested size
    std::unique_ptr<uint8_t[]> acquire(size_t size);

    // Release buffer back to pool
    void release(std::unique_ptr<uint8_t[]> buffer, size_t size);

    // Get statistics
    struct Stats {
        std::atomic<size_t> small_acquisitions;
        std::atomic<size_t> medium_acquisitions;
        std::atomic<size_t> large_acquisitions;
        std::atomic<size_t> custom_allocations;

        std::atomic<size_t> small_hits;
        std::atomic<size_t> medium_hits;
        std::atomic<size_t> large_hits;

        std::atomic<size_t> total_bytes_allocated;
        std::atomic<size_t> total_bytes_reused;

        Stats()
            : small_acquisitions(0),
              medium_acquisitions(0),
              large_acquisitions(0),
              custom_allocations(0),
              small_hits(0),
              medium_hits(0),
              large_hits(0),
              total_bytes_allocated(0),
              total_bytes_reused(0) {}
    };

    const Stats& get_stats() const {
        return stats_;
    }
    void reset_stats() {
        stats_.small_acquisitions = 0;
        stats_.medium_acquisitions = 0;
        stats_.large_acquisitions = 0;
        stats_.custom_allocations = 0;
        stats_.small_hits = 0;
        stats_.medium_hits = 0;
        stats_.large_hits = 0;
        stats_.total_bytes_allocated = 0;
        stats_.total_bytes_reused = 0;
    }

   private:
    // Buffer bucket for a specific size
    struct BufferBucket {
        size_t buffer_size;
        size_t max_buffers;
        std::queue<std::unique_ptr<uint8_t[]>> available;
        std::mutex mutex;

        BufferBucket(size_t size, size_t max) : buffer_size(size), max_buffers(max) {}
    };

    Config config_;
    Stats stats_;

    // Size-specific buffer pools
    BufferBucket small_buffers_;
    BufferBucket medium_buffers_;
    BufferBucket large_buffers_;

    // Pre-allocate buffers
    void pre_allocate_buffers();

    // Select appropriate bucket
    BufferBucket* select_bucket(size_t size);
};

/**
 * RAII buffer handle that automatically returns buffer to pool
 */
class PooledBuffer {
   public:
    PooledBuffer() = default;

    PooledBuffer(std::unique_ptr<uint8_t[]> buffer, size_t size, MemoryPool* pool)
        : buffer_(std::move(buffer)), size_(size), pool_(pool) {}

    ~PooledBuffer() {
        if (buffer_ && pool_) {
            pool_->release(std::move(buffer_), size_);
        }
    }

    // Move semantics
    PooledBuffer(PooledBuffer&& other) noexcept
        : buffer_(std::move(other.buffer_)), size_(other.size_), pool_(other.pool_) {
        other.size_ = 0;
        other.pool_ = nullptr;
    }

    PooledBuffer& operator=(PooledBuffer&& other) noexcept {
        if (this != &other) {
            // Release current buffer
            if (buffer_ && pool_) {
                pool_->release(std::move(buffer_), size_);
            }

            // Take ownership
            buffer_ = std::move(other.buffer_);
            size_ = other.size_;
            pool_ = other.pool_;

            other.size_ = 0;
            other.pool_ = nullptr;
        }
        return *this;
    }

    // Delete copy operations
    PooledBuffer(const PooledBuffer&) = delete;
    PooledBuffer& operator=(const PooledBuffer&) = delete;

    // Access
    uint8_t* data() {
        return buffer_.get();
    }
    const uint8_t* data() const {
        return buffer_.get();
    }
    size_t size() const {
        return size_;
    }
    bool valid() const {
        return buffer_ != nullptr;
    }

    // Release ownership
    std::unique_ptr<uint8_t[]> release() {
        pool_ = nullptr;
        return std::move(buffer_);
    }

   private:
    std::unique_ptr<uint8_t[]> buffer_;
    size_t size_ = 0;
    MemoryPool* pool_ = nullptr;
};

}  // namespace bgen
}  // namespace io
}  // namespace ldcov

#endif  // LDCOV_IO_BGEN_DECOMPRESS_MEMORY_POOL_H