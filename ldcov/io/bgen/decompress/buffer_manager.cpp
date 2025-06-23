#include "buffer_manager.h"
#include <algorithm>
#include <stdexcept>

namespace ldcov {
namespace bgen {
namespace decompress {

//==============================================================================
// BufferHandle implementation
//==============================================================================

BufferHandle::BufferHandle(std::unique_ptr<uint8_t[]> buffer, size_t size,
                           BufferManager* manager, 
                           std::function<void(std::unique_ptr<uint8_t[]>, size_t)> return_func)
    : buffer_(std::move(buffer))
    , size_(size)
    , manager_(manager)
    , return_func_(std::move(return_func)) {
}

BufferHandle::BufferHandle(BufferHandle&& other) noexcept
    : buffer_(std::move(other.buffer_))
    , size_(other.size_)
    , manager_(other.manager_)
    , return_func_(std::move(other.return_func_)) {
    other.size_ = 0;
    other.manager_ = nullptr;
}

BufferHandle& BufferHandle::operator=(BufferHandle&& other) noexcept {
    if (this != &other) {
        // Return current buffer if we have one
        if (buffer_ && manager_ && return_func_) {
            return_func_(std::move(buffer_), size_);
        }
        
        // Take ownership of other's buffer
        buffer_ = std::move(other.buffer_);
        size_ = other.size_;
        manager_ = other.manager_;
        return_func_ = std::move(other.return_func_);
        
        other.size_ = 0;
        other.manager_ = nullptr;
    }
    return *this;
}

BufferHandle::~BufferHandle() {
    if (buffer_ && manager_ && return_func_) {
        return_func_(std::move(buffer_), size_);
    }
}

std::unique_ptr<uint8_t[]> BufferHandle::release() {
    manager_ = nullptr;
    return_func_ = nullptr;
    return std::move(buffer_);
}

//==============================================================================
// BufferManager implementation
//==============================================================================

BufferManager::BufferManager(const Config& config)
    : config_(config) {
    if (config_.initial_buffer_size == 0) {
        throw std::invalid_argument("Initial buffer size must be greater than 0");
    }
    if (config_.max_buffer_size < config_.initial_buffer_size) {
        throw std::invalid_argument("Max buffer size must be >= initial buffer size");
    }
    if (config_.growth_factor <= 1.0) {
        throw std::invalid_argument("Growth factor must be greater than 1.0");
    }
}

BufferManager::~BufferManager() {
    // Pool entries will be automatically cleaned up by vector destructor
    // unique_ptr in PoolEntry will free the buffers
}

BufferHandle BufferManager::get_buffer(size_t required_size) {
    if (required_size == 0) {
        throw std::invalid_argument("Buffer size must be greater than 0");
    }
    
    if (required_size > config_.max_buffer_size) {
        throw std::invalid_argument("Requested buffer size exceeds maximum allowed size");
    }
    
    std::unique_ptr<uint8_t[]> buffer;
    size_t buffer_size = 0;
    
    // Try to get a buffer from the pool
    {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        
        // Look for a buffer that's large enough
        auto it = std::find_if(pool_.begin(), pool_.end(),
            [required_size](const PoolEntry& entry) {
                return entry.size >= required_size;
            });
        
        if (it != pool_.end()) {
            // Found a suitable buffer
            buffer = std::move(it->buffer);
            buffer_size = it->size;
            pool_.erase(it);
            
            if (config_.enable_statistics) {
                stats_.pool_hits.fetch_add(1, std::memory_order_relaxed);
                stats_.current_pool_size.fetch_sub(1, std::memory_order_relaxed);
            }
        }
    }
    
    // If we didn't find a buffer, allocate a new one
    if (!buffer) {
        // Calculate appropriate size with growth factor
        buffer_size = std::max(required_size, config_.initial_buffer_size);
        
        // Apply growth factor if this is larger than initial size
        if (required_size > config_.initial_buffer_size) {
            size_t grown_size = static_cast<size_t>(required_size * config_.growth_factor);
            buffer_size = std::min(grown_size, config_.max_buffer_size);
        }
        
        try {
            buffer.reset(new uint8_t[buffer_size]);
        } catch (const std::bad_alloc&) {
            // Try again with exact size if growth failed
            buffer_size = required_size;
            buffer.reset(new uint8_t[buffer_size]);
        }
        
        if (config_.enable_statistics) {
            stats_.pool_misses.fetch_add(1, std::memory_order_relaxed);
            stats_.total_allocations.fetch_add(1, std::memory_order_relaxed);
            stats_.total_memory_allocated.fetch_add(buffer_size, std::memory_order_relaxed);
            
            // Update current and peak memory usage
            size_t current = current_memory_usage_.fetch_add(buffer_size, std::memory_order_relaxed) + buffer_size;
            size_t peak = stats_.peak_memory_allocated.load(std::memory_order_relaxed);
            while (current > peak && !stats_.peak_memory_allocated.compare_exchange_weak(peak, current)) {
                // Loop until we successfully update peak or current is no longer greater
            }
        }
    }
    
    // Create return function that captures this
    auto return_func = [this](std::unique_ptr<uint8_t[]> buf, size_t size) {
        this->return_buffer(std::move(buf), size);
    };
    
    return BufferHandle(std::move(buffer), buffer_size, this, return_func);
}

void BufferManager::return_buffer(std::unique_ptr<uint8_t[]> buffer, size_t size) {
    if (!buffer) {
        return;
    }
    
    bool returned_to_pool = false;
    
    {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        
        // Only return to pool if we haven't exceeded the pool size limit
        if (pool_.size() < config_.max_pool_size) {
            pool_.emplace_back(std::move(buffer), size);
            returned_to_pool = true;
            
            if (config_.enable_statistics) {
                stats_.current_pool_size.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
    
    // If we didn't return to pool, the buffer will be deallocated
    if (!returned_to_pool && config_.enable_statistics) {
        stats_.total_deallocations.fetch_add(1, std::memory_order_relaxed);
        current_memory_usage_.fetch_sub(size, std::memory_order_relaxed);
    }
}

void BufferManager::clear_pool() {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    if (config_.enable_statistics) {
        // Update statistics for buffers being cleared
        for (const auto& entry : pool_) {
            stats_.total_deallocations.fetch_add(1, std::memory_order_relaxed);
            current_memory_usage_.fetch_sub(entry.size, std::memory_order_relaxed);
        }
        stats_.current_pool_size.store(0, std::memory_order_relaxed);
    }
    
    pool_.clear();
}

// Global buffer manager instance
BufferManager& get_global_buffer_manager() {
    static BufferManager manager;
    return manager;
}

} // namespace decompress
} // namespace bgen
} // namespace ldcov