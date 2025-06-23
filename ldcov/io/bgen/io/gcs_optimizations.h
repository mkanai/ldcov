#pragma once

#include <vector>
#include <deque>
#include <future>
#include <memory>
#include <chrono>
#include <unordered_map>
#include <optional>

namespace ldcov {
namespace io {
namespace bgen {

/**
 * Optimized GCS reader with prefetching and caching
 */
class OptimizedGCSReader : public FileReader {
public:
    static constexpr size_t DEFAULT_BUFFER_SIZE = 10 * 1024 * 1024;  // 10MB
    static constexpr size_t PREFETCH_QUEUE_SIZE = 5;
    static constexpr size_t CACHE_SIZE = 100 * 1024 * 1024;  // 100MB
    
    explicit OptimizedGCSReader(const std::string& filename);
    ~OptimizedGCSReader() override;
    
    // FileReader interface
    size_t read(uint8_t* buffer, size_t size) override;
    void seek(uint64_t position) override;
    uint64_t tell() const override { return position_; }
    uint64_t size() const override { return file_size_; }
    
    // Optimization features
    void enable_prefetching(bool enable) { prefetch_enabled_ = enable; }
    void set_cache_size(size_t size) { max_cache_size_ = size; }
    void prefetch_range(uint64_t start, uint64_t end);
    
private:
    // Prefetching
    struct PrefetchRequest {
        uint64_t offset;
        size_t size;
        std::future<std::vector<uint8_t>> future;
    };
    
    std::deque<PrefetchRequest> prefetch_queue_;
    bool prefetch_enabled_ = true;
    std::thread prefetch_thread_;
    std::atomic<bool> stop_prefetching_{false};
    
    void prefetch_worker();
    void queue_prefetch(uint64_t offset, size_t size);
    std::optional<std::vector<uint8_t>> get_from_prefetch(uint64_t offset, size_t size);
    
    // Caching
    struct CacheEntry {
        std::vector<uint8_t> data;
        std::chrono::steady_clock::time_point last_access;
        size_t access_count = 0;
    };
    
    std::unordered_map<uint64_t, CacheEntry> cache_;
    size_t current_cache_size_ = 0;
    size_t max_cache_size_ = CACHE_SIZE;
    
    std::optional<std::vector<uint8_t>> get_from_cache(uint64_t offset, size_t size);
    void add_to_cache(uint64_t offset, const std::vector<uint8_t>& data);
    void evict_cache_entries(size_t required_space);
    
    // Double buffering
    std::vector<uint8_t> buffers_[2];
    int active_buffer_ = 0;
    uint64_t buffer_starts_[2] = {0, 0};
    size_t buffer_valid_[2] = {0, 0};
    
    // Retry logic
    template<typename Func>
    auto retry_with_backoff(Func&& func, size_t max_retries = 3);
    
    // GCS-specific
    uint64_t position_ = 0;
    uint64_t file_size_ = 0;
    size_t buffer_size_ = DEFAULT_BUFFER_SIZE;
    
    // Batch read optimization
    std::vector<std::pair<uint64_t, std::vector<uint8_t>>> 
    batch_read(const std::vector<std::pair<uint64_t, size_t>>& ranges);
};

/**
 * GCS-aware variant decompressor with prefetching
 */
class GCSVariantDecompressor : public VariantDecompressor {
public:
    explicit GCSVariantDecompressor(std::unique_ptr<OptimizedGCSReader> reader);
    
    std::vector<uint8_t> decompress_variant(
        uint64_t offset,
        const VariantMetadata& metadata) override;
    
    void decompress_variants_batch(
        const std::vector<uint64_t>& offsets,
        const std::vector<VariantMetadata>& metadata,
        std::function<void(size_t, std::vector<uint8_t>)> callback) override;
    
    void hint_access_pattern(AccessPattern pattern) override;
    
private:
    std::unique_ptr<OptimizedGCSReader> reader_;
    
    // Prefetch the next N variants based on access pattern
    void prefetch_next_variants(size_t current_index, size_t count);
    
    // Optimize read size based on network characteristics
    size_t calculate_optimal_read_size(size_t requested_size);
};

/**
 * Connection pool for GCS to reduce connection overhead
 */
class GCSConnectionPool {
public:
    static GCSConnectionPool& instance() {
        static GCSConnectionPool pool;
        return pool;
    }
    
    std::shared_ptr<OptimizedGCSReader> get_reader(const std::string& path);
    void return_reader(const std::string& path, std::shared_ptr<OptimizedGCSReader> reader);
    
private:
    struct PoolEntry {
        std::shared_ptr<OptimizedGCSReader> reader;
        std::chrono::steady_clock::time_point last_used;
        bool in_use = false;
    };
    
    std::unordered_map<std::string, std::vector<PoolEntry>> pool_;
    std::mutex pool_mutex_;
    size_t max_connections_per_file_ = 4;
    
    void cleanup_stale_connections();
};

} // namespace bgen
} // namespace io
} // namespace ldcov