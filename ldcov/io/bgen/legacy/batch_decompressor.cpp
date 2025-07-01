// DEPRECATED: This is the old batch decompressor implementation
// New implementation is in decompress/parallel_decompressor.cpp
// This file is kept for backward compatibility with existing code

#include "batch_decompressor.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>

// Include compression libraries
extern "C" {
#include "zlib.h"
#include "zstd.h"
}

namespace ldcov {
namespace bgen {

BatchDecompressor::BatchDecompressor(int num_threads, int queue_size)
    : num_threads_(num_threads), max_queue_size_(queue_size) {
    // Reserve space for file descriptors
    file_descriptors_.resize(num_threads_, -1);

    // Start worker threads
    worker_threads_.reserve(num_threads_);
    for (int i = 0; i < num_threads_; ++i) {
        worker_threads_.emplace_back(&BatchDecompressor::worker_thread_func, this, i);
    }
}

BatchDecompressor::~BatchDecompressor() {
    shutdown();
}

void BatchDecompressor::open_file_descriptors(const std::string& filename) {
    close_file_descriptors();

    for (int i = 0; i < num_threads_; ++i) {
        file_descriptors_[i] = ::open(filename.c_str(), O_RDONLY);
        if (file_descriptors_[i] == -1) {
            // Clean up any opened descriptors
            close_file_descriptors();
            throw std::runtime_error("Failed to open file: " + filename);
        }
    }
}

void BatchDecompressor::close_file_descriptors() {
    for (int& fd : file_descriptors_) {
        if (fd != -1) {
            ::close(fd);
            fd = -1;
        }
    }
}

void BatchDecompressor::submit_batch(const std::vector<DecompressionTask>& tasks,
                                     const std::string& filename) {
    // Update filename if changed
    {
        std::lock_guard<std::mutex> lock(file_mutex_);
        if (filename != current_filename_) {
            current_filename_ = filename;
            open_file_descriptors(filename);
        }
    }

    // Submit all tasks to queue
    for (const auto& task : tasks) {
        task_queue_.push(task);
    }
}

std::vector<DecompressionResult> BatchDecompressor::get_results(int count) {
    std::vector<DecompressionResult> results;
    results.reserve(count);

    for (int i = 0; i < count; ++i) {
        OwnedDecompressionResult owned_result;
        result_queue_.pop(owned_result, true);  // Blocking wait

        // Move the result, keeping data ownership in OwnedDecompressionResult
        results.push_back(owned_result.result);
    }

    return results;
}

void BatchDecompressor::shutdown() {
    // Signal shutdown
    task_queue_.shutdown();
    result_queue_.shutdown();

    // Wait for all threads to complete
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    worker_threads_.clear();

    // Close file descriptors
    close_file_descriptors();
}

inline bool BatchDecompressor::decompress_zlib_inline(const uint8_t* compressed_data,
                                                      size_t compressed_size,
                                                      uint8_t* output_buffer, size_t expected_size,
                                                      size_t* actual_size) {
    z_stream strm;
    strm.zalloc = Z_NULL;
    strm.zfree = Z_NULL;
    strm.opaque = Z_NULL;
    strm.avail_in = compressed_size;
    strm.next_in = const_cast<Bytef*>(compressed_data);
    strm.avail_out = expected_size;
    strm.next_out = output_buffer;

    // Check for zlib header to determine format
    // BGEN v1.1 uses standard zlib (with header)
    // BGEN v1.2 uses raw deflate (no header)
    int ret;
    if (compressed_size >= 2 && compressed_data[0] == 0x78 &&
        (compressed_data[1] == 0x01 || compressed_data[1] == 0x5E || compressed_data[1] == 0x9C ||
         compressed_data[1] == 0xDA)) {
        // Standard zlib format detected (v1.1)
        ret = inflateInit(&strm);
    } else {
        // Raw deflate format (v1.2)
        ret = inflateInit2(&strm, -15);
    }

    if (ret != Z_OK) {
        return false;
    }

    ret = inflate(&strm, Z_FINISH);
    *actual_size = strm.total_out;

    inflateEnd(&strm);
    return ret == Z_STREAM_END;
}

inline bool BatchDecompressor::decompress_zstd_inline(const uint8_t* compressed_data,
                                                      size_t compressed_size,
                                                      uint8_t* output_buffer, size_t expected_size,
                                                      size_t* actual_size) {
    size_t result = ZSTD_decompress(output_buffer, expected_size, compressed_data, compressed_size);

    if (ZSTD_isError(result)) {
        return false;
    }

    *actual_size = result;
    return true;
}

void BatchDecompressor::worker_thread_func(int thread_id) {
    // Get thread-specific file descriptor
    int fd = -1;

    while (true) {
        DecompressionTask task;
        if (!task_queue_.pop(task, true)) {
            break;  // Shutdown signal
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        // Create owned result
        OwnedDecompressionResult owned_result;
        owned_result.result.offset = task.offset;
        owned_result.result.success = false;
        owned_result.result.error_code = DecompressionResult::SUCCESS;

        // Get file descriptor for this thread
        {
            std::lock_guard<std::mutex> lock(file_mutex_);
            if (thread_id < static_cast<int>(file_descriptors_.size())) {
                fd = file_descriptors_[thread_id];
            }
        }

        if (fd == -1) {
            owned_result.result.error_code = DecompressionResult::READ_ERROR;
            result_queue_.push(std::move(owned_result));
            continue;
        }

        // Handle uncompressed data
        if (task.compression_type == 0) {
            // Allocate owned buffer
            owned_result.owned_data.reset(new uint8_t[task.compressed_size]);

            // Read directly into owned buffer
            ssize_t bytes_read =
                ::pread(fd, owned_result.owned_data.get(), task.compressed_size, task.offset);

            if (bytes_read == static_cast<ssize_t>(task.compressed_size)) {
                owned_result.result.data = owned_result.owned_data.get();
                owned_result.result.size = task.compressed_size;
                owned_result.result.success = true;
                total_bytes_read_ += task.compressed_size;
                total_bytes_decompressed_ += task.compressed_size;
            } else {
                owned_result.result.error_code = DecompressionResult::READ_ERROR;
            }
        } else {
            // Read compressed data into thread-local buffer
            uint8_t* work_buffer = BufferPool::get_work_buffer(task.compressed_size);
            ssize_t bytes_read = ::pread(fd, work_buffer, task.compressed_size, task.offset);

            if (bytes_read != static_cast<ssize_t>(task.compressed_size)) {
                owned_result.result.error_code = DecompressionResult::READ_ERROR;
                result_queue_.push(std::move(owned_result));
                continue;
            }

            total_bytes_read_ += task.compressed_size;

            // For compressed BGEN data, the block contains:
            // [4-byte uncompressed size][compressed payload]
            // We need to read the uncompressed size first, then decompress the payload

            // Read the uncompressed size from the first 4 bytes
            uint32_t uncompressed_size = *reinterpret_cast<uint32_t*>(work_buffer);
            uint32_t payload_size = task.compressed_size - 4;  // Skip the 4-byte size prefix

            // Allocate owned decompression buffer
            owned_result.owned_data.reset(new uint8_t[uncompressed_size]);
            size_t actual_size = 0;
            bool decomp_success = false;

            // Decompress only the payload (skip the 4-byte size prefix)
            switch (task.compression_type) {
                case 1:  // zlib
                    decomp_success = decompress_zlib_inline(work_buffer + 4, payload_size,
                                                            owned_result.owned_data.get(),
                                                            uncompressed_size, &actual_size);
                    break;
                case 2:  // zstd
                    decomp_success = decompress_zstd_inline(work_buffer + 4, payload_size,
                                                            owned_result.owned_data.get(),
                                                            uncompressed_size, &actual_size);
                    break;
                default:
                    owned_result.result.error_code = DecompressionResult::INVALID_COMPRESSION;
                    break;
            }

            if (decomp_success) {
                owned_result.result.data = owned_result.owned_data.get();
                owned_result.result.size = actual_size;
                owned_result.result.success = true;
                total_bytes_decompressed_ += actual_size;
            } else if (owned_result.result.error_code == DecompressionResult::SUCCESS) {
                owned_result.result.error_code = DecompressionResult::DECOMPRESS_ERROR;
            }
        }

        if (owned_result.result.success) {
            total_tasks_completed_++;

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration_us =
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time)
                    .count();
            total_decompression_time_us_ += duration_us;
        }

        // Push result to queue
        result_queue_.push(std::move(owned_result));
    }
}

}  // namespace bgen
}  // namespace ldcov