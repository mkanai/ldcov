// DEPRECATED: This is the old sequential decompressor implementation
// New implementation is in decompress/sequential_decompressor.cpp
// This file is kept for backward compatibility with existing code

#include "sequential_decompressor.h"

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <future>
#include <thread>

// Include compression libraries
extern "C" {
#include "zlib.h"
#include "zstd.h"
}

namespace ldcov {
namespace bgen {

SequentialDecompressor::SequentialDecompressor(const std::string& filename)
    : filename_(filename),
      fd_(-1),
      file_size_(0),
      total_bytes_read_(0),
      total_bytes_decompressed_(0),
      total_variants_processed_(0) {
    // Open file with O_DIRECT if available for better performance
#ifdef O_DIRECT
    fd_ = ::open(filename_.c_str(), O_RDONLY | O_DIRECT);
    if (fd_ == -1) {
        // Fallback without O_DIRECT
        fd_ = ::open(filename_.c_str(), O_RDONLY);
    }
#else
    fd_ = ::open(filename_.c_str(), O_RDONLY);
#endif

    if (fd_ == -1) {
        throw std::runtime_error("Failed to open file: " + filename_);
    }

    // Get file size
    struct stat st;
    if (fstat(fd_, &st) == -1) {
        ::close(fd_);
        throw std::runtime_error("Failed to stat file: " + filename_);
    }
    file_size_ = st.st_size;

    // Initialize read-ahead buffer
    read_ahead_.reset(new ReadAheadBuffer());
}

SequentialDecompressor::~SequentialDecompressor() {
    // Wait for any pending read-ahead
    if (read_ahead_ && read_ahead_->is_active && read_ahead_->future.valid()) {
        read_ahead_->future.wait();
    }

    if (fd_ != -1) {
        ::close(fd_);
    }
}

SequentialDecompressor::SequentialDecompressor(SequentialDecompressor&& other) noexcept
    : filename_(std::move(other.filename_)),
      fd_(other.fd_),
      file_size_(other.file_size_),
      read_ahead_(std::move(other.read_ahead_)),
      total_bytes_read_(other.total_bytes_read_),
      total_bytes_decompressed_(other.total_bytes_decompressed_),
      total_variants_processed_(other.total_variants_processed_) {
    other.fd_ = -1;
}

SequentialDecompressor& SequentialDecompressor::operator=(SequentialDecompressor&& other) noexcept {
    if (this != &other) {
        // Clean up current state
        if (fd_ != -1) {
            ::close(fd_);
        }

        // Move from other
        filename_ = std::move(other.filename_);
        fd_ = other.fd_;
        file_size_ = other.file_size_;
        read_ahead_ = std::move(other.read_ahead_);
        total_bytes_read_ = other.total_bytes_read_;
        total_bytes_decompressed_ = other.total_bytes_decompressed_;
        total_variants_processed_ = other.total_variants_processed_;

        other.fd_ = -1;
    }
    return *this;
}

ssize_t SequentialDecompressor::read_at_offset(uint64_t offset, uint8_t* buffer, size_t size) {
    ssize_t bytes_read = ::pread(fd_, buffer, size, offset);
    if (bytes_read > 0) {
        total_bytes_read_ += bytes_read;
    }
    return bytes_read;
}

void SequentialDecompressor::start_read_ahead(uint64_t offset, size_t size) {
    // Cancel any pending read-ahead
    if (read_ahead_->is_active && read_ahead_->future.valid()) {
        // We can't cancel, so just wait
        read_ahead_->future.wait();
    }

    // Allocate buffer if needed
    if (!read_ahead_->buffer || read_ahead_->size < size) {
        read_ahead_->buffer.reset(new uint8_t[size]);
        read_ahead_->size = size;
    }

    read_ahead_->offset = offset;
    read_ahead_->is_active = true;

    // Start async read
    int fd_copy = fd_;
    uint8_t* buffer_ptr = read_ahead_->buffer.get();

    read_ahead_->future = std::async(std::launch::async, [fd_copy, offset, buffer_ptr, size]() {
        return ::pread(fd_copy, buffer_ptr, size, offset);
    });
}

ssize_t SequentialDecompressor::complete_read_ahead() {
    if (!read_ahead_->is_active || !read_ahead_->future.valid()) {
        return -1;
    }

    ssize_t result = read_ahead_->future.get();
    read_ahead_->is_active = false;

    if (result > 0) {
        total_bytes_read_ += result;
    }

    return result;
}

inline bool SequentialDecompressor::decompress_zlib_inline(const uint8_t* compressed_data,
                                                           size_t compressed_size,
                                                           uint8_t* output_buffer,
                                                           size_t expected_size,
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

inline bool SequentialDecompressor::decompress_zstd_inline(const uint8_t* compressed_data,
                                                           size_t compressed_size,
                                                           uint8_t* output_buffer,
                                                           size_t expected_size,
                                                           size_t* actual_size) {
    size_t result = ZSTD_decompress(output_buffer, expected_size, compressed_data, compressed_size);

    if (ZSTD_isError(result)) {
        return false;
    }

    *actual_size = result;
    return true;
}

DecompressionResult SequentialDecompressor::decompress_variant(uint64_t offset,
                                                               uint32_t compressed_size,
                                                               uint32_t expected_size,
                                                               uint8_t compression_type) {
    DecompressionResult result;
    result.offset = offset;
    result.success = false;
    result.error_code = DecompressionResult::SUCCESS;

    // Handle uncompressed data
    if (compression_type == 0) {
        uint8_t* buffer = BufferPool::get_decompression_buffer(compressed_size);
        ssize_t bytes_read = read_at_offset(offset, buffer, compressed_size);

        if (bytes_read != static_cast<ssize_t>(compressed_size)) {
            result.error_code = DecompressionResult::READ_ERROR;
            return result;
        }

        result.data = buffer;
        result.size = compressed_size;
        result.success = true;
        total_bytes_decompressed_ += compressed_size;
        total_variants_processed_++;
        return result;
    }

    // For compressed BGEN data, the block contains:
    // [4-byte uncompressed size][compressed payload]
    // We need to read the uncompressed size first, then decompress the payload

    uint8_t* work_buffer = BufferPool::get_work_buffer(compressed_size);
    ssize_t bytes_read = read_at_offset(offset, work_buffer, compressed_size);

    if (bytes_read != static_cast<ssize_t>(compressed_size)) {
        result.error_code = DecompressionResult::READ_ERROR;
        return result;
    }

    // Read the uncompressed size from the first 4 bytes
    uint32_t uncompressed_size = *reinterpret_cast<uint32_t*>(work_buffer);
    uint32_t payload_size = compressed_size - 4;  // Skip the 4-byte size prefix

    // Get decompression buffer
    uint8_t* decomp_buffer = BufferPool::get_decompression_buffer(uncompressed_size);
    size_t actual_size = 0;
    bool decomp_success = false;

    // Decompress only the payload (skip the 4-byte size prefix)
    switch (compression_type) {
        case 1:  // zlib
            decomp_success = decompress_zlib_inline(work_buffer + 4, payload_size, decomp_buffer,
                                                    uncompressed_size, &actual_size);
            break;
        case 2:  // zstd
            decomp_success = decompress_zstd_inline(work_buffer + 4, payload_size, decomp_buffer,
                                                    uncompressed_size, &actual_size);
            break;
        default:
            result.error_code = DecompressionResult::INVALID_COMPRESSION;
            return result;
    }

    if (!decomp_success) {
        result.error_code = DecompressionResult::DECOMPRESS_ERROR;
        return result;
    }

    result.data = decomp_buffer;
    result.size = actual_size;
    result.success = true;
    total_bytes_decompressed_ += actual_size;
    total_variants_processed_++;

    return result;
}

DecompressionResult SequentialDecompressor::decompress_variant_allocated(uint64_t offset,
                                                                         uint32_t compressed_size,
                                                                         uint32_t expected_size,
                                                                         uint8_t compression_type) {
    // First decompress using the regular method
    DecompressionResult temp_result =
        decompress_variant(offset, compressed_size, expected_size, compression_type);

    // Create a new result with allocated memory
    DecompressionResult result;
    result.offset = temp_result.offset;
    result.success = temp_result.success;
    result.error_code = temp_result.error_code;

    if (temp_result.success && temp_result.data != nullptr && temp_result.size > 0) {
        // Allocate new buffer and copy data
        uint8_t* new_buffer = new uint8_t[temp_result.size];
        std::memcpy(new_buffer, temp_result.data, temp_result.size);
        result.data = new_buffer;
        result.size = temp_result.size;
    } else {
        result.data = nullptr;
        result.size = 0;
    }

    return result;
}

std::vector<DecompressionResult> SequentialDecompressor::decompress_sequential(
    const std::vector<uint64_t>& offsets, const std::vector<uint32_t>& compressed_sizes,
    const std::vector<uint32_t>& expected_sizes, const std::vector<uint8_t>& compression_types,
    bool enable_readahead) {
    size_t n_variants = offsets.size();
    std::vector<DecompressionResult> results;
    results.reserve(n_variants);

    // Determine if we should use read-ahead
    bool use_readahead = enable_readahead && n_variants > 1 && is_sequential_pattern(offsets);

    for (size_t i = 0; i < n_variants; ++i) {
        // Start read-ahead for next variant if applicable
        if (use_readahead && i + 1 < n_variants) {
            start_read_ahead(offsets[i + 1], compressed_sizes[i + 1]);
        }

        // Process current variant
        DecompressionResult result;

        // Check if we have this data from previous read-ahead
        if (use_readahead && i > 0 && read_ahead_->is_active && read_ahead_->offset == offsets[i]) {
            // Complete the read-ahead
            ssize_t bytes_read = complete_read_ahead();

            if (bytes_read == static_cast<ssize_t>(compressed_sizes[i])) {
                // Use read-ahead data
                if (compression_types[i] == 0) {
                    // Uncompressed - copy to decompression buffer
                    uint8_t* buffer = BufferPool::get_decompression_buffer(compressed_sizes[i]);
                    std::memcpy(buffer, read_ahead_->buffer.get(), compressed_sizes[i]);

                    result.offset = offsets[i];
                    result.data = buffer;
                    result.size = compressed_sizes[i];
                    result.success = true;
                    result.error_code = DecompressionResult::SUCCESS;
                    total_bytes_decompressed_ += compressed_sizes[i];
                    total_variants_processed_++;
                } else {
                    // Decompress from read-ahead buffer
                    // For compressed BGEN data, the block contains:
                    // [4-byte uncompressed size][compressed payload]

                    // Read the uncompressed size from the first 4 bytes
                    uint32_t uncompressed_size =
                        *reinterpret_cast<uint32_t*>(read_ahead_->buffer.get());
                    uint32_t payload_size = compressed_sizes[i] - 4;  // Skip the 4-byte size prefix

                    uint8_t* decomp_buffer =
                        BufferPool::get_decompression_buffer(uncompressed_size);
                    size_t actual_size = 0;
                    bool decomp_success = false;

                    // Decompress only the payload (skip the 4-byte size prefix)
                    switch (compression_types[i]) {
                        case 1:  // zlib
                            decomp_success = decompress_zlib_inline(
                                read_ahead_->buffer.get() + 4, payload_size, decomp_buffer,
                                uncompressed_size, &actual_size);
                            break;
                        case 2:  // zstd
                            decomp_success = decompress_zstd_inline(
                                read_ahead_->buffer.get() + 4, payload_size, decomp_buffer,
                                uncompressed_size, &actual_size);
                            break;
                    }

                    result.offset = offsets[i];
                    if (decomp_success) {
                        result.data = decomp_buffer;
                        result.size = actual_size;
                        result.success = true;
                        result.error_code = DecompressionResult::SUCCESS;
                        total_bytes_decompressed_ += actual_size;
                        total_variants_processed_++;
                    } else {
                        result.success = false;
                        result.error_code = DecompressionResult::DECOMPRESS_ERROR;
                    }
                }
            } else {
                // Read-ahead failed, fall back to regular read
                result = decompress_variant(offsets[i], compressed_sizes[i], expected_sizes[i],
                                            compression_types[i]);
            }
        } else {
            // Regular decompression
            result = decompress_variant(offsets[i], compressed_sizes[i], expected_sizes[i],
                                        compression_types[i]);
        }

        results.push_back(result);
    }

    return results;
}

bool SequentialDecompressor::is_sequential_pattern(const std::vector<uint64_t>& offsets,
                                                   uint64_t max_gap) {
    if (offsets.size() < 2) {
        return true;
    }

    // Check if offsets are monotonically increasing with reasonable gaps
    for (size_t i = 1; i < offsets.size(); ++i) {
        if (offsets[i] <= offsets[i - 1]) {
            return false;  // Not monotonic
        }

        uint64_t gap = offsets[i] - offsets[i - 1];
        if (gap > max_gap) {
            return false;  // Gap too large
        }
    }

    return true;
}

}  // namespace bgen
}  // namespace ldcov