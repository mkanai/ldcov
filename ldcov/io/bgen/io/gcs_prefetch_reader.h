#pragma once

#include "gcs_file_reader.h"
#include <thread>
#include <atomic>
#include <condition_variable>
#include <queue>

namespace ldcov {
namespace io {
namespace bgen {

/**
 * GCS reader with intelligent prefetching for sequential access patterns
 */
class GCSPrefetchReader : public GCSFileReader {
public:
    explicit GCSPrefetchReader(const std::string& filename);
    ~GCSPrefetchReader();
    
    // Override read to use prefetch buffer
    size_t read(uint8_t* buffer, size_t size) override;
    void seek(uint64_t position) override;
    
    // Prefetch configuration
    void set_prefetch_size(size_t size) { prefetch_size_ = size; }
    void enable_prefetching(bool enable) { prefetch_enabled_ = enable; }
    
private:
    // Prefetch buffer structure
    struct PrefetchBuffer {
        std::vector<uint8_t> data;
        uint64_t offset;
        size_t valid_size;
        bool ready;
    };
    
    // Double buffering for prefetch
    PrefetchBuffer prefetch_buffers_[2];
    std::atomic<int> active_prefetch_buffer_{0};
    
    // Prefetch thread management
    std::thread prefetch_thread_;
    std::atomic<bool> stop_prefetching_{false};
    std::condition_variable prefetch_cv_;
    std::mutex prefetch_mutex_;
    
    // Prefetch configuration
    size_t prefetch_size_ = 10 * 1024 * 1024;  // 10MB default
    bool prefetch_enabled_ = true;
    
    // Access pattern detection
    uint64_t last_read_offset_ = 0;
    size_t sequential_read_count_ = 0;
    static constexpr size_t SEQUENTIAL_THRESHOLD = 3;
    
    // Methods
    void prefetch_worker();
    void trigger_prefetch(uint64_t offset);
    bool is_sequential_access(uint64_t offset) const;
    bool try_read_from_prefetch(uint64_t offset, uint8_t* buffer, size_t size, size_t& bytes_read);
};

// Implementation
inline GCSPrefetchReader::GCSPrefetchReader(const std::string& filename)
    : GCSFileReader(filename) {
    
    // Initialize prefetch buffers
    for (auto& buffer : prefetch_buffers_) {
        buffer.data.resize(prefetch_size_);
        buffer.ready = false;
    }
    
    // Start prefetch thread
    if (prefetch_enabled_) {
        prefetch_thread_ = std::thread(&GCSPrefetchReader::prefetch_worker, this);
    }
}

inline GCSPrefetchReader::~GCSPrefetchReader() {
    // Stop prefetch thread
    stop_prefetching_ = true;
    prefetch_cv_.notify_all();
    
    if (prefetch_thread_.joinable()) {
        prefetch_thread_.join();
    }
}

inline void GCSPrefetchReader::prefetch_worker() {
    while (!stop_prefetching_) {
        std::unique_lock<std::mutex> lock(prefetch_mutex_);
        prefetch_cv_.wait(lock, [this] { 
            return stop_prefetching_ || 
                   !prefetch_buffers_[1 - active_prefetch_buffer_].ready; 
        });
        
        if (stop_prefetching_) break;
        
        // Determine which buffer to fill
        int buffer_to_fill = 1 - active_prefetch_buffer_;
        auto& buffer = prefetch_buffers_[buffer_to_fill];
        
        // Calculate prefetch offset
        uint64_t current_pos = tell();
        uint64_t prefetch_offset = current_pos + prefetch_size_;
        
        // Don't prefetch beyond file size
        if (prefetch_offset >= size()) {
            continue;
        }
        
        lock.unlock();
        
        // Perform the prefetch read
        try {
            // Save current position
            uint64_t saved_pos = tell();
            
            // Read ahead
            GCSFileReader::seek(prefetch_offset);
            buffer.valid_size = GCSFileReader::read(
                buffer.data.data(), 
                std::min(prefetch_size_, size() - prefetch_offset)
            );
            buffer.offset = prefetch_offset;
            buffer.ready = true;
            
            // Restore position
            GCSFileReader::seek(saved_pos);
            
        } catch (const std::exception& e) {
            // Log error but don't crash
            buffer.ready = false;
        }
    }
}

inline bool GCSPrefetchReader::is_sequential_access(uint64_t offset) const {
    // Check if this read follows the previous one
    return offset == last_read_offset_;
}

inline bool GCSPrefetchReader::try_read_from_prefetch(
    uint64_t offset, uint8_t* buffer, size_t size, size_t& bytes_read) {
    
    // Check both buffers
    for (int i = 0; i < 2; ++i) {
        const auto& prefetch_buffer = prefetch_buffers_[i];
        
        if (!prefetch_buffer.ready) continue;
        
        // Check if requested data is in this buffer
        if (offset >= prefetch_buffer.offset && 
            offset < prefetch_buffer.offset + prefetch_buffer.valid_size) {
            
            // Calculate how much we can read from this buffer
            uint64_t buffer_offset = offset - prefetch_buffer.offset;
            size_t available = prefetch_buffer.valid_size - buffer_offset;
            bytes_read = std::min(size, available);
            
            // Copy data
            std::memcpy(buffer, 
                       prefetch_buffer.data.data() + buffer_offset, 
                       bytes_read);
            
            // Switch active buffer if we used the inactive one
            if (i != active_prefetch_buffer_) {
                active_prefetch_buffer_ = i;
            }
            
            return true;
        }
    }
    
    return false;
}

inline size_t GCSPrefetchReader::read(uint8_t* buffer, size_t size) {
    uint64_t current_offset = tell();
    
    // Try to read from prefetch buffer first
    size_t bytes_read = 0;
    if (prefetch_enabled_ && try_read_from_prefetch(current_offset, buffer, size, bytes_read)) {
        // Update position
        seek(current_offset + bytes_read);
        
        // Trigger next prefetch if sequential
        if (is_sequential_access(current_offset)) {
            sequential_read_count_++;
            if (sequential_read_count_ >= SEQUENTIAL_THRESHOLD) {
                trigger_prefetch(current_offset + bytes_read);
            }
        }
        
        last_read_offset_ = current_offset + bytes_read;
        return bytes_read;
    }
    
    // Fall back to regular read
    bytes_read = GCSFileReader::read(buffer, size);
    
    // Update access pattern tracking
    if (is_sequential_access(current_offset)) {
        sequential_read_count_++;
    } else {
        sequential_read_count_ = 0;
    }
    
    last_read_offset_ = current_offset + bytes_read;
    
    // Start prefetching if we detect sequential pattern
    if (prefetch_enabled_ && sequential_read_count_ >= SEQUENTIAL_THRESHOLD) {
        trigger_prefetch(current_offset + bytes_read);
    }
    
    return bytes_read;
}

inline void GCSPrefetchReader::seek(uint64_t position) {
    GCSFileReader::seek(position);
    
    // Reset sequential detection on seek
    sequential_read_count_ = 0;
    
    // Clear prefetch buffers as they're likely invalid now
    for (auto& buffer : prefetch_buffers_) {
        buffer.ready = false;
    }
}

inline void GCSPrefetchReader::trigger_prefetch(uint64_t offset) {
    // Mark the inactive buffer as not ready and wake up prefetch thread
    prefetch_buffers_[1 - active_prefetch_buffer_].ready = false;
    prefetch_cv_.notify_one();
}

} // namespace bgen
} // namespace io
} // namespace ldcov