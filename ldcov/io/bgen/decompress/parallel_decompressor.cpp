#include "parallel_decompressor.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <sstream>
#include <thread>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#elif defined(__APPLE__)
#include <mach/thread_act.h>
#include <mach/thread_policy.h>
#include <pthread.h>
#endif

namespace ldcov {
namespace bgen {
namespace decompress {

// TaskQueue implementation
void ParallelDecompressor::TaskQueue::push(std::unique_ptr<DecompressionTask> task) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (shutdown_) {
            throw std::runtime_error("Cannot push to shutdown queue");
        }
        queue_.push(std::move(task));
    }
    cv_.notify_one();
}

std::unique_ptr<ParallelDecompressor::DecompressionTask> ParallelDecompressor::TaskQueue::pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return !queue_.empty() || shutdown_; });

    if (queue_.empty()) {
        return nullptr;
    }

    auto task = std::move(queue_.front());
    queue_.pop();
    return task;
}

std::unique_ptr<ParallelDecompressor::DecompressionTask> ParallelDecompressor::TaskQueue::pop_with_timeout(
    std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    bool got_item = cv_.wait_for(lock, timeout, [this] { return !queue_.empty() || shutdown_; });

    if (!got_item || queue_.empty()) {
        return nullptr;
    }

    auto task = std::move(queue_.front());
    queue_.pop();
    return task;
}

void ParallelDecompressor::TaskQueue::shutdown() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        shutdown_ = true;
    }
    cv_.notify_all();
}

size_t ParallelDecompressor::TaskQueue::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
}

// ResultCollector implementation
void ParallelDecompressor::ResultCollector::add_result(size_t task_id, DecompressedData result) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Store the result
    results_[task_id] = std::move(result);

    // Check if we can move any results to ready queue
    while (results_.find(next_expected_id_) != results_.end()) {
        ready_results_.push_back(std::move(results_[next_expected_id_]));
        results_.erase(next_expected_id_);
        next_expected_id_++;
    }

    cv_.notify_all();
}

std::vector<DecompressedData> ParallelDecompressor::ResultCollector::collect_results(size_t count) {
    std::unique_lock<std::mutex> lock(mutex_);

    // Wait with timeout (30 seconds)
    auto timeout = std::chrono::seconds(30);
    bool success = cv_.wait_for(lock, timeout, [this, count] {
        return ready_results_.size() >= count || error_occurred_.load();
    });

    // Check for timeout
    if (!success) {
        throw std::runtime_error(
            "Timeout waiting for decompression results after 30 seconds. "
            "This may indicate a deadlock or extremely slow decompression.");
    }

    // Check for errors
    if (error_occurred_.load()) {
        throw std::runtime_error("Error occurred during parallel decompression: " + error_message_);
    }

    // Extract the requested number of results
    std::vector<DecompressedData> results;
    results.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        results.push_back(std::move(ready_results_[i]));
    }

    // Remove collected results
    ready_results_.erase(ready_results_.begin(), ready_results_.begin() + count);

    return results;
}

void ParallelDecompressor::ResultCollector::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    results_.clear();
    ready_results_.clear();
    next_expected_id_ = 0;
    error_occurred_ = false;
    error_message_.clear();
}

void ParallelDecompressor::ResultCollector::report_error(const std::string& error_message) {
    std::lock_guard<std::mutex> lock(mutex_);
    error_occurred_ = true;
    error_message_ = error_message;
    cv_.notify_all();  // Wake up any waiting threads
}

size_t ParallelDecompressor::ResultCollector::ready_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return ready_results_.size();
}

// ParallelDecompressor implementation
ParallelDecompressor::ParallelDecompressor(FileReader* file_reader, const ParallelConfig& config)
    : VariantDecompressor(config),
      parallel_config_(config),
      file_reader_(file_reader),
      start_time_(std::chrono::steady_clock::now()) {
    if (!file_reader_) {
        throw std::invalid_argument("FileReader cannot be null");
    }

    // Determine number of threads
    size_t num_threads = determine_thread_count(parallel_config_.num_threads);

    // Create main thread buffer manager
    BufferManager::Config buffer_config;
    buffer_config.initial_buffer_size = parallel_config_.io_buffer_size;
    buffer_config.max_buffer_size = parallel_config_.max_decompressed_size;
    buffer_config.enable_statistics = true;
    main_buffer_manager_ = std::unique_ptr<BufferManager>(new BufferManager(buffer_config));

    // Create worker threads
    workers_.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
        auto worker = std::unique_ptr<WorkerState>(new WorkerState());
        worker->thread_id = i;

        // Create per-thread buffer manager
        worker->buffer_manager = std::unique_ptr<BufferManager>(new BufferManager(buffer_config));

        // Allocate I/O buffer
        worker->io_buffer =
            std::unique_ptr<uint8_t[]>(new uint8_t[parallel_config_.io_buffer_size]);

        // Start worker thread
        worker->thread =
            std::thread(&ParallelDecompressor::worker_thread_function, this, worker.get());

        // Pin thread to core if requested
        if (parallel_config_.pin_threads) {
            pin_thread_to_core(worker->thread, i % std::thread::hardware_concurrency());
        }

        workers_.push_back(std::move(worker));
    }

// Debug: Print initialization info
#ifdef DEBUG_PARALLEL_DECOMPRESSOR
    std::cerr << "ParallelDecompressor initialized with " << num_threads << " worker threads"
              << std::endl;
#endif
}

ParallelDecompressor::~ParallelDecompressor() {
    // Signal shutdown
    shutdown_ = true;
    task_queue_.shutdown();

    // Wait for all workers to finish
    for (auto& worker : workers_) {
        if (worker->thread.joinable()) {
            worker->thread.join();
        }
    }
}

DecompressedData ParallelDecompressor::decompress(const CompressedVariant& variant) {
    // For single variant, use batch interface
    std::vector<CompressedVariant> variants = {variant};
    auto results = decompress_batch(variants);
    return std::move(results[0]);
}

std::vector<DecompressedData> ParallelDecompressor::decompress_batch(
    const std::vector<CompressedVariant>& variants) {
    if (variants.empty()) {
        return {};
    }

    // Reset result collector and task ID counter for new batch
    result_collector_.reset();
    next_task_id_ = 0;  // Reset task ID counter to ensure proper ordering

    // Submit all tasks
    for (size_t i = 0; i < variants.size(); ++i) {
        // Check queue size with timeout
        auto queue_wait_start = std::chrono::steady_clock::now();
        while (task_queue_.size() >= parallel_config_.max_queue_size) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));

            // Check for timeout while waiting for queue space
            auto elapsed = std::chrono::steady_clock::now() - queue_wait_start;
            if (elapsed > std::chrono::seconds(5)) {
                throw std::runtime_error(
                    "Timeout waiting for task queue space. Queue may be stuck.");
            }
        }

        // Create task with sequential ID for proper ordering
        auto task = std::unique_ptr<DecompressionTask>(new DecompressionTask(i, variants[i]));

        // Submit task
        task_queue_.push(std::move(task));

        // Prefetch if enabled
        if (parallel_config_.enable_prefetch &&
            i + parallel_config_.prefetch_distance < variants.size()) {
            prefetch_variants(variants, i + 1);
        }
    }

    // Main thread participates in decompression work
    // Process tasks until all results are collected
    size_t results_needed = variants.size();
    size_t results_collected = 0;
    
    // Create a main thread state for decompression
    WorkerState main_state;
    main_state.thread_id = static_cast<size_t>(-1);  // Special ID for main thread
    main_state.buffer_manager = std::move(main_buffer_manager_);
    main_state.io_buffer = std::unique_ptr<uint8_t[]>(new uint8_t[parallel_config_.io_buffer_size]);
    
    try {
        while (results_collected < results_needed) {
            // Try to get a task from the queue with a short timeout
            auto task = task_queue_.pop_with_timeout(std::chrono::milliseconds(10));
            
            if (task) {
                // Process the task
                process_single_task(std::move(task), &main_state);
            }
            
            // Check how many results are ready
            results_collected = result_collector_.ready_count();
        }
    } catch (...) {
        // Restore main_buffer_manager_ before rethrowing
        main_buffer_manager_ = std::move(main_state.buffer_manager);
        throw;
    }
    
    // Restore main_buffer_manager_
    main_buffer_manager_ = std::move(main_state.buffer_manager);
    
    // Collect results in order
    return result_collector_.collect_results(variants.size());
}

void ParallelDecompressor::process_single_task(std::unique_ptr<DecompressionTask> task,
                                               WorkerState* state) {
    auto start = std::chrono::high_resolution_clock::now();

    // Decompress the variant
    DecompressedData result = decompress_variant(task->variant, state);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    // Update worker statistics
    state->variants_processed.fetch_add(1);
    if (result.success) {
        state->bytes_decompressed.fetch_add(result.size);
    }
    state->decompression_time_ns.fetch_add(duration.count());

    // Add to result collector first (moves result)
    result_collector_.add_result(task->task_id, std::move(result));
}

void ParallelDecompressor::worker_thread_function(WorkerState* state) {
    try {
        while (!shutdown_) {
            // Get next task
            auto task = task_queue_.pop();
            if (!task) {
                break;  // Queue was shut down
            }

            // Process the task using the shared helper
            process_single_task(std::move(task), state);
        }
    } catch (const std::exception& e) {
        // Report error to result collector
        result_collector_.report_error(std::string("Worker thread ") +
                                       std::to_string(state->thread_id) + " error: " + e.what());
    } catch (...) {
        // Report unknown error
        result_collector_.report_error(std::string("Worker thread ") +
                                       std::to_string(state->thread_id) +
                                       " encountered unknown error");
    }
}

DecompressedData ParallelDecompressor::decompress_variant(const CompressedVariant& variant,
                                                          WorkerState* state) {
    try {
        // Validate input
        if (!variant.data && variant.compressed_size > 0) {
            // Need to read from file
            if (!file_reader_->is_open()) {
                return DecompressedData(variant.offset, DecompressedData::INVALID_INPUT,
                                        "File reader is not open");
            }

            // Read compressed data into I/O buffer
            if (variant.compressed_size > parallel_config_.io_buffer_size) {
                return DecompressedData(variant.offset, DecompressedData::MEMORY_ERROR,
                                        "Compressed size exceeds I/O buffer size");
            }

            size_t bytes_read = file_reader_->read_at(variant.offset, state->io_buffer.get(),
                                                      variant.compressed_size);
            if (bytes_read != variant.compressed_size) {
                return DecompressedData(variant.offset, DecompressedData::INVALID_INPUT,
                                        "Failed to read compressed data from file");
            }

            // Update variant with read data and continue processing
            const_cast<CompressedVariant&>(variant).data = state->io_buffer.get();
        }

        // Reject uncompressed data
        if (variant.compression_type == CompressionType::None) {
            return DecompressedData(variant.offset, DecompressedData::UNSUPPORTED_COMPRESSION,
                                    "Uncompressed BGEN data is not supported. "
                                    "Please use compressed BGEN files (zlib or zstd).");
        }

        // Get decompression buffer from worker's buffer manager
        auto buffer_handle =
            state->buffer_manager->get_decompression_buffer(variant.uncompressed_size);
        if (!buffer_handle.valid()) {
            return DecompressedData(variant.offset, DecompressedData::MEMORY_ERROR,
                                    "Failed to allocate decompression buffer");
        }

        // Decompress based on type
        size_t decompressed_size = 0;
        bool success = false;

        CompressionResult result;
        switch (variant.compression_type) {
            case CompressionType::Zlib:
                result = decompress_zlib(variant.data, variant.compressed_size,
                                         buffer_handle.data(), variant.uncompressed_size);
                success = result.success;
                decompressed_size = result.bytes_processed;
                break;

            case CompressionType::Zstd:
                result = decompress_zstd(variant.data, variant.compressed_size,
                                         buffer_handle.data(), variant.uncompressed_size);
                success = result.success;
                decompressed_size = result.bytes_processed;
                break;

            default:
                return DecompressedData(variant.offset, DecompressedData::UNSUPPORTED_COMPRESSION,
                                        "Unsupported compression type");
        }

        if (!success) {
            return DecompressedData(
                variant.offset, DecompressedData::COMPRESSION_ERROR,
                result.error_message.empty() ? "Decompression failed" : result.error_message);
        }

        // Validate size if requested
        if (config_.validate_size && decompressed_size != variant.uncompressed_size) {
            return DecompressedData(variant.offset, DecompressedData::SIZE_MISMATCH,
                                    "Decompressed size does not match expected size");
        }

        // Release buffer from handle and create result
        auto result_buffer = buffer_handle.release();
        return DecompressedData(std::move(result_buffer), decompressed_size, variant.offset);

    } catch (const std::exception& e) {
        return DecompressedData(variant.offset, DecompressedData::COMPRESSION_ERROR,
                                std::string("Exception during decompression: ") + e.what());
    }
}

VariantDecompressor::Statistics ParallelDecompressor::get_statistics() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    Statistics stats = stats_;

    // Aggregate worker statistics
    for (const auto& worker : workers_) {
        stats.total_variants += worker->variants_processed.load();
        stats.total_decompressed_bytes += worker->bytes_decompressed.load();
    }

    stats.successful_decompressions = stats.total_variants - stats.failed_decompressions;

    return stats;
}

void ParallelDecompressor::reset_statistics() {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    stats_ = Statistics();

    // Reset worker statistics
    for (auto& worker : workers_) {
        worker->variants_processed = 0;
        worker->bytes_decompressed = 0;
        worker->decompression_time_ns = 0;
    }

    start_time_ = std::chrono::steady_clock::now();
}

size_t ParallelDecompressor::determine_thread_count(size_t requested) {
    if (requested > 0) {
        return requested;
    }

    // Auto-detect based on hardware
    size_t hw_threads = std::thread::hardware_concurrency();
    if (hw_threads == 0) {
        hw_threads = 4;  // Fallback
    }

    // Use 75% of available threads, minimum 2, maximum 16
    size_t threads = static_cast<size_t>(hw_threads * 0.75);
    threads = std::max<size_t>(2, threads);
    threads = std::min<size_t>(16, threads);

    return threads;
}

void ParallelDecompressor::pin_thread_to_core(std::thread& thread, size_t core_id) {
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);

    int rc = pthread_setaffinity_np(thread.native_handle(), sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        std::cerr << "Warning: Failed to pin thread to core " << core_id << std::endl;
    }
#elif defined(__APPLE__)
    thread_affinity_policy_data_t policy = {static_cast<integer_t>(core_id)};
    thread_policy_set(pthread_mach_thread_np(thread.native_handle()), THREAD_AFFINITY_POLICY,
                      (thread_policy_t)&policy, THREAD_AFFINITY_POLICY_COUNT);
#else
    // Platform not supported
    (void)thread;
    (void)core_id;
#endif
}

void ParallelDecompressor::prefetch_variants(const std::vector<CompressedVariant>& variants,
                                             size_t start_idx) {
    // Prefetch is handled by the file reader if it supports it
    // This is a placeholder for future optimization
    (void)variants;
    (void)start_idx;
}

}  // namespace decompress
}  // namespace bgen
}  // namespace ldcov