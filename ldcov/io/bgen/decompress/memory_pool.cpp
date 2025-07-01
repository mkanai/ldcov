#include "memory_pool.h"

#include <cstring>
#include <stdexcept>

namespace ldcov {
namespace io {
namespace bgen {

MemoryPool::MemoryPool(const Config& config)
    : config_(config),
      small_buffers_(config.small_buffer_size, config.max_small_buffers),
      medium_buffers_(config.medium_buffer_size, config.max_medium_buffers),
      large_buffers_(config.large_buffer_size, config.max_large_buffers) {
    if (config_.pre_allocate) {
        pre_allocate_buffers();
    }
}

MemoryPool::~MemoryPool() {
    // Buffers will be automatically cleaned up by unique_ptr destructors
}

void MemoryPool::pre_allocate_buffers() {
    // Pre-allocate small buffers
    for (size_t i = 0; i < config_.max_small_buffers / 2; ++i) {
        auto buffer = std::unique_ptr<uint8_t[]>(new uint8_t[config_.small_buffer_size]);
        std::lock_guard<std::mutex> lock(small_buffers_.mutex);
        small_buffers_.available.push(std::move(buffer));
        stats_.total_bytes_allocated += config_.small_buffer_size;
    }

    // Pre-allocate medium buffers
    for (size_t i = 0; i < config_.max_medium_buffers / 2; ++i) {
        auto buffer = std::unique_ptr<uint8_t[]>(new uint8_t[config_.medium_buffer_size]);
        std::lock_guard<std::mutex> lock(medium_buffers_.mutex);
        medium_buffers_.available.push(std::move(buffer));
        stats_.total_bytes_allocated += config_.medium_buffer_size;
    }

    // Pre-allocate large buffers
    for (size_t i = 0; i < config_.max_large_buffers / 2; ++i) {
        auto buffer = std::unique_ptr<uint8_t[]>(new uint8_t[config_.large_buffer_size]);
        std::lock_guard<std::mutex> lock(large_buffers_.mutex);
        large_buffers_.available.push(std::move(buffer));
        stats_.total_bytes_allocated += config_.large_buffer_size;
    }
}

MemoryPool::BufferBucket* MemoryPool::select_bucket(size_t size) {
    if (size <= config_.small_buffer_size) {
        return &small_buffers_;
    } else if (size <= config_.medium_buffer_size) {
        return &medium_buffers_;
    } else if (size <= config_.large_buffer_size) {
        return &large_buffers_;
    }
    return nullptr;
}

std::unique_ptr<uint8_t[]> MemoryPool::acquire(size_t size) {
    // Select appropriate bucket
    BufferBucket* bucket = select_bucket(size);

    if (bucket) {
        // Try to get from pool
        std::unique_lock<std::mutex> lock(bucket->mutex);
        if (!bucket->available.empty()) {
            auto buffer = std::move(bucket->available.front());
            bucket->available.pop();
            lock.unlock();

            // Update statistics
            if (bucket == &small_buffers_) {
                stats_.small_acquisitions++;
                stats_.small_hits++;
                stats_.total_bytes_reused += config_.small_buffer_size;
            } else if (bucket == &medium_buffers_) {
                stats_.medium_acquisitions++;
                stats_.medium_hits++;
                stats_.total_bytes_reused += config_.medium_buffer_size;
            } else {
                stats_.large_acquisitions++;
                stats_.large_hits++;
                stats_.total_bytes_reused += config_.large_buffer_size;
            }

            // Zero memory if requested
            if (config_.zero_on_acquire) {
                std::memset(buffer.get(), 0, bucket->buffer_size);
            }

            return buffer;
        }

        // Pool empty, allocate new buffer
        lock.unlock();

        // Update statistics
        if (bucket == &small_buffers_) {
            stats_.small_acquisitions++;
        } else if (bucket == &medium_buffers_) {
            stats_.medium_acquisitions++;
        } else {
            stats_.large_acquisitions++;
        }

        auto buffer = std::unique_ptr<uint8_t[]>(new uint8_t[bucket->buffer_size]);
        stats_.total_bytes_allocated += bucket->buffer_size;

        if (config_.zero_on_acquire) {
            std::memset(buffer.get(), 0, bucket->buffer_size);
        }

        return buffer;
    }

    // Size too large for pools, allocate custom
    stats_.custom_allocations++;
    stats_.total_bytes_allocated += size;

    auto buffer = std::unique_ptr<uint8_t[]>(new uint8_t[size]);
    if (config_.zero_on_acquire) {
        std::memset(buffer.get(), 0, size);
    }

    return buffer;
}

void MemoryPool::release(std::unique_ptr<uint8_t[]> buffer, size_t size) {
    if (!buffer) {
        return;
    }

    // Select appropriate bucket
    BufferBucket* bucket = select_bucket(size);

    if (bucket) {
        std::lock_guard<std::mutex> lock(bucket->mutex);

        // Only keep if pool not full
        if (bucket->available.size() < bucket->max_buffers) {
            bucket->available.push(std::move(buffer));
            return;
        }
    }

    // Buffer will be destroyed when unique_ptr goes out of scope
}

}  // namespace bgen
}  // namespace io
}  // namespace ldcov