#ifndef LDCOV_BGEN_DECOMPRESS_PARALLEL_DECOMPRESSOR_H
#define LDCOV_BGEN_DECOMPRESS_PARALLEL_DECOMPRESSOR_H

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <vector>

#include "../io/reader_interface.h"
#include "buffer_manager.h"
#include "compression_utils.h"
#include "decompressor.h"

namespace ldcov {
namespace bgen {
namespace decompress {

// Import FileReader into this namespace
using FileReader = ::ldcov::io::bgen::FileReader;

/**
 * ParallelDecompressor - High-performance parallel decompressor for random access patterns
 *
 * This decompressor is optimized for random access patterns where multiple variants
 * may be requested in any order. It uses a thread pool to decompress variants in
 * parallel while maintaining order in the output.
 *
 * Key features:
 * - Thread pool with configurable number of workers
 * - Lock-free work distribution using atomics where possible
 * - Each worker has its own buffers (no shared state during decompression)
 * - Results are collected in order using a priority queue
 * - Integrates with FileReader for efficient I/O
 * - Comprehensive error handling and statistics
 */
class ParallelDecompressor : public VariantDecompressor {
   public:
    // Extended configuration for parallel decompressor
    struct ParallelConfig : public Config {
        // Number of worker threads (0 = auto-detect based on hardware)
        size_t num_threads;

        // Maximum number of tasks to queue (prevents excessive memory usage)
        size_t max_queue_size;

        // Whether to use pinned threads for better cache locality
        bool pin_threads;

        // Size of I/O buffer per thread
        size_t io_buffer_size;

        // Whether to prefetch data for better performance
        bool enable_prefetch;

        // Number of variants to prefetch ahead
        size_t prefetch_distance;

        // Constructor with defaults
        ParallelConfig()
            : Config(),
              num_threads(0),
              max_queue_size(1000),
              pin_threads(false),
              io_buffer_size(4 * 1024 * 1024)  // 4MB
              ,
              enable_prefetch(true),
              prefetch_distance(10) {}
    };

    /**
     * Constructor
     *
     * @param file_reader FileReader instance for reading variant data
     * @param config Configuration for the parallel decompressor
     */
    ParallelDecompressor(FileReader* file_reader, const ParallelConfig& config = ParallelConfig());

    /**
     * Destructor - ensures all threads are properly shut down
     */
    ~ParallelDecompressor() override;

    // Delete copy operations
    ParallelDecompressor(const ParallelDecompressor&) = delete;
    ParallelDecompressor& operator=(const ParallelDecompressor&) = delete;

    /**
     * Decompress a single variant
     *
     * @param variant The compressed variant to decompress
     * @return DecompressedData containing the result
     */
    DecompressedData decompress(const CompressedVariant& variant) override;

    /**
     * Decompress a batch of variants in parallel
     *
     * @param variants Vector of compressed variants to decompress
     * @return Vector of decompressed data in the same order as input
     */
    std::vector<DecompressedData> decompress_batch(
        const std::vector<CompressedVariant>& variants) override;

    /**
     * Get decompression statistics
     *
     * @return Statistics struct with performance metrics
     */
    Statistics get_statistics() const override;

    /**
     * Reset statistics counters
     */
    void reset_statistics() override;

   private:
    // Internal task representation
    struct DecompressionTask {
        size_t task_id;                          // Unique task ID for ordering
        CompressedVariant variant;               // Variant to decompress
        std::promise<DecompressedData> promise;  // Promise for result

        DecompressionTask(size_t id, CompressedVariant v) : task_id(id), variant(std::move(v)) {}
    };

    // Worker thread state
    struct WorkerState {
        std::thread thread;
        std::unique_ptr<BufferManager> buffer_manager;
        std::unique_ptr<uint8_t[]> io_buffer;
        size_t thread_id;

        // Per-thread statistics
        std::atomic<uint64_t> variants_processed{0};
        std::atomic<uint64_t> bytes_decompressed{0};
        std::atomic<uint64_t> decompression_time_ns{0};
    };

    // Thread-safe task queue
    class TaskQueue {
       public:
        void push(std::unique_ptr<DecompressionTask> task);
        std::unique_ptr<DecompressionTask> pop();
        std::unique_ptr<DecompressionTask> pop_with_timeout(std::chrono::milliseconds timeout);
        void shutdown();
        size_t size() const;
        bool is_shutdown() const {
            return shutdown_.load();
        }

       private:
        mutable std::mutex mutex_;
        std::condition_variable cv_;
        std::queue<std::unique_ptr<DecompressionTask>> queue_;
        std::atomic<bool> shutdown_{false};
    };

    // Result collector that maintains order
    class ResultCollector {
       public:
        void add_result(size_t task_id, DecompressedData result);
        std::vector<DecompressedData> collect_results(size_t count);
        void reset();
        void report_error(const std::string& error_message);
        size_t ready_count() const;

       private:
        mutable std::mutex mutex_;
        std::condition_variable cv_;
        std::unordered_map<size_t, DecompressedData> results_;
        size_t next_expected_id_ = 0;
        std::vector<DecompressedData> ready_results_;
        std::atomic<bool> error_occurred_{false};
        std::string error_message_;
    };

    // Worker thread function
    void worker_thread_function(WorkerState* state);

    // Process a single task (called by workers and main thread)
    void process_single_task(std::unique_ptr<DecompressionTask> task, WorkerState* state);

    // Decompress a single variant (called by workers)
    DecompressedData decompress_variant(const CompressedVariant& variant, WorkerState* state);

    // Helper to determine optimal thread count
    static size_t determine_thread_count(size_t requested);

    // Helper to pin thread to CPU core
    void pin_thread_to_core(std::thread& thread, size_t core_id);

    // Prefetch data for upcoming variants
    void prefetch_variants(const std::vector<CompressedVariant>& variants, size_t start_idx);

    // Member variables
    ParallelConfig parallel_config_;
    FileReader* file_reader_;

    // Thread pool
    std::vector<std::unique_ptr<WorkerState>> workers_;
    TaskQueue task_queue_;

    // Result handling
    ResultCollector result_collector_;
    std::atomic<size_t> next_task_id_{0};

    // Global buffer manager for main thread
    std::unique_ptr<BufferManager> main_buffer_manager_;

    // Statistics
    mutable std::mutex stats_mutex_;
    Statistics stats_;
    std::chrono::steady_clock::time_point start_time_;

    // Shutdown flag
    std::atomic<bool> shutdown_{false};
};

}  // namespace decompress
}  // namespace bgen
}  // namespace ldcov

#endif  // LDCOV_BGEN_DECOMPRESS_PARALLEL_DECOMPRESSOR_H