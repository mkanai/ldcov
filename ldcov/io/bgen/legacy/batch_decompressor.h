#ifndef LDCOV_BGEN_BATCH_DECOMPRESSOR_H
#define LDCOV_BGEN_BATCH_DECOMPRESSOR_H

#include <fcntl.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "buffer_pool.h"
#include "decompression_result.h"

namespace ldcov {
namespace bgen {

// Task for decompression
struct DecompressionTask {
    uint64_t offset;
    uint32_t compressed_size;
    uint32_t expected_uncompressed_size;
    uint8_t compression_type;  // 0=none, 1=zlib, 2=zstd
};

// Thread-safe queue for tasks and results
template <typename T>
class ThreadSafeQueue {
   private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<T> queue_;
    std::atomic<bool> shutdown_{false};

   public:
    void push(T item) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(std::move(item));
        }
        cv_.notify_one();
    }

    bool pop(T& item, bool blocking = true) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (blocking) {
            cv_.wait(lock, [this] { return !queue_.empty() || shutdown_; });
        }

        if (queue_.empty()) {
            return false;
        }

        item = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    void shutdown() {
        shutdown_ = true;
        cv_.notify_all();
    }
};

// Structure to hold decompressed data with ownership
struct OwnedDecompressionResult {
    DecompressionResult result;
    std::unique_ptr<uint8_t[]> owned_data;  // Owns the data pointed to by result.data

    // Default constructor
    OwnedDecompressionResult() = default;

    // Move constructor
    OwnedDecompressionResult(OwnedDecompressionResult&& other) noexcept
        : result(other.result), owned_data(std::move(other.owned_data)) {
        // Update pointer if we own the data
        if (owned_data) {
            result.data = owned_data.get();
        }
    }

    // Move assignment operator
    OwnedDecompressionResult& operator=(OwnedDecompressionResult&& other) noexcept {
        if (this != &other) {
            result = other.result;
            owned_data = std::move(other.owned_data);
            if (owned_data) {
                result.data = owned_data.get();
            }
        }
        return *this;
    }

    // Deleted copy operations
    OwnedDecompressionResult(const OwnedDecompressionResult&) = delete;
    OwnedDecompressionResult& operator=(const OwnedDecompressionResult&) = delete;
};

// Main batch decompressor class - optimized for parallel random access
class BatchDecompressor {
   private:
    // Configuration
    const int num_threads_;
    const size_t max_queue_size_;

    // Thread pool
    std::vector<std::thread> worker_threads_;

    // Queues
    ThreadSafeQueue<DecompressionTask> task_queue_;
    ThreadSafeQueue<OwnedDecompressionResult> result_queue_;

    // File handling - use file descriptors for thread safety
    std::string current_filename_;
    std::vector<int> file_descriptors_;  // One per worker thread
    std::mutex file_mutex_;

    // Statistics
    std::atomic<uint64_t> total_bytes_read_{0};
    std::atomic<uint64_t> total_bytes_decompressed_{0};
    std::atomic<uint64_t> total_tasks_completed_{0};
    std::atomic<uint64_t> total_decompression_time_us_{0};

    // Worker thread function
    void worker_thread_func(int thread_id);

    // Helper to open file descriptors
    void open_file_descriptors(const std::string& filename);
    void close_file_descriptors();

    // Inline decompression functions
    inline bool decompress_zlib_inline(const uint8_t* compressed_data, size_t compressed_size,
                                       uint8_t* output_buffer, size_t expected_size,
                                       size_t* actual_size);

    inline bool decompress_zstd_inline(const uint8_t* compressed_data, size_t compressed_size,
                                       uint8_t* output_buffer, size_t expected_size,
                                       size_t* actual_size);

   public:
    BatchDecompressor(int num_threads = 2, int queue_size = 100);
    ~BatchDecompressor();

    // Delete copy operations
    BatchDecompressor(const BatchDecompressor&) = delete;
    BatchDecompressor& operator=(const BatchDecompressor&) = delete;

    // Submit a batch of decompression tasks
    void submit_batch(const std::vector<DecompressionTask>& tasks, const std::string& filename);

    // Get completed results (blocking if not enough ready)
    std::vector<DecompressionResult> get_results(int count);

    // Check if results are ready (non-blocking)
    bool has_results(int count) const {
        return result_queue_.size() >= static_cast<size_t>(count);
    }

    // Shutdown the decompressor
    void shutdown();

    // Statistics
    uint64_t total_bytes_read() const {
        return total_bytes_read_;
    }
    uint64_t total_bytes_decompressed() const {
        return total_bytes_decompressed_;
    }
    uint64_t total_tasks_completed() const {
        return total_tasks_completed_;
    }
    double average_decompression_time_ms() const {
        uint64_t tasks = total_tasks_completed_;
        if (tasks == 0)
            return 0.0;
        return static_cast<double>(total_decompression_time_us_) / (tasks * 1000.0);
    }
};

}  // namespace bgen
}  // namespace ldcov

#endif  // LDCOV_BGEN_BATCH_DECOMPRESSOR_H