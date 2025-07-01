#include "sequential_decompressor.h"

#include <zlib.h>
#include <zstd.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "../io/reader_interface.h"

namespace ldcov {
namespace bgen {
namespace decompress {

// Helper function to detect if storage is SSD
static bool is_ssd_storage() {
#ifdef __linux__
    // On Linux, check /sys/block/*/queue/rotational
    // Try common device names
    const std::vector<std::string> devices = {"sda", "sdb", "nvme0n1", "nvme1n1"};

    for (const auto& device : devices) {
        std::string path = "/sys/block/" + device + "/queue/rotational";
        std::ifstream file(path);
        if (file.is_open()) {
            int rotational = 1;
            file >> rotational;
            file.close();
            // If any device is SSD (rotational=0), assume SSD
            if (rotational == 0) {
                return true;
            }
        }
    }
    // Default to SSD on modern systems if can't detect
    return true;
#else
    // On non-Linux systems, assume SSD by default
    return true;
#endif
}

SequentialDecompressor::SequentialDecompressor(const SequentialConfig& config)
    : VariantDecompressor(config),
      config_(config),
      file_reader_(config.file_reader),
      direct_read_buffer_(nullptr),
      direct_read_buffer_size_(0),
      enable_prefetch_(!is_ssd_storage()),  // Disable prefetch on SSD
      prefetch_distance_(64) {              // Cache line size

    // Validate configuration
    if (!file_reader_) {
        throw std::invalid_argument("SequentialDecompressor: file_reader cannot be null");
    }

    if (!file_reader_->is_open()) {
        throw std::invalid_argument("SequentialDecompressor: file_reader must have an open file");
    }

    // Set up buffer manager
    if (config.buffer_manager) {
        buffer_manager_ = config.buffer_manager;
    } else {
        // Create our own buffer manager with appropriate configuration
        BufferManager::Config bm_config;
        bm_config.initial_buffer_size = config.readahead_buffer_size;
        bm_config.max_buffer_size =
            std::max(config.readahead_buffer_size * 2, config.max_decompressed_size);
        bm_config.max_pool_size = 4;  // Small pool for sequential access
        bm_config.enable_statistics = true;

        owned_buffer_manager_ = std::unique_ptr<BufferManager>(new BufferManager(bm_config));
        buffer_manager_ = owned_buffer_manager_.get();
    }

    // Pre-allocate direct read buffer
    direct_read_buffer_size_ = config.readahead_buffer_size;
    direct_read_buffer_.reset(new uint8_t[direct_read_buffer_size_]);

    // Initialize statistics
    reset_statistics();

    // Log prefetch status (useful for debugging)
    if (!enable_prefetch_) {
        // SSD detected, prefetching disabled for better performance
    }
}

SequentialDecompressor::~SequentialDecompressor() {
    // Destructor implementation moved to DecompressionContexts
}

DecompressedData SequentialDecompressor::decompress(const CompressedVariant& variant) {
    try {
        // OPTIMIZATION: Fast path for uncompressed data - skip buffer manager entirely
        if (variant.compression_type == CompressionType::None && variant.data == nullptr) {
            return decompress_uncompressed_direct(variant);
        }

        // Always try direct decompression path first for best performance
        // when we have file offset (not data pointer)
        if (variant.data == nullptr) {
            return decompress_direct(variant);
        }

        // Handle case where variant already has data pointer
        // If variant already has data pointer, use it directly
        if (variant.data != nullptr) {
            // Decompress the data directly from the provided pointer
            auto result =
                decompress_data(variant.data, variant.compressed_size, variant.uncompressed_size,
                                variant.compression_type, variant.offset);
            update_statistics(variant, result.success);
            return result;
        }

        // Otherwise, read compressed data (may use read-ahead buffer)
        auto compressed_buffer = read_compressed_data(variant);
        if (!compressed_buffer.valid()) {
            update_statistics(variant, false);
            return DecompressedData(variant.offset, DecompressedData::MEMORY_ERROR,
                                    "Failed to read compressed data");
        }

        // Decompress the data
        auto result =
            decompress_data(compressed_buffer.data(), variant.compressed_size,
                            variant.uncompressed_size, variant.compression_type, variant.offset);

        update_statistics(variant, result.success);
        return result;

    } catch (const std::exception& e) {
        update_statistics(variant, false);
        return DecompressedData(variant.offset, DecompressedData::COMPRESSION_ERROR,
                                std::string("Exception during decompression: ") + e.what());
    }
}

std::vector<DecompressedData> SequentialDecompressor::decompress_batch(
    const std::vector<CompressedVariant>& variants) {
    std::vector<DecompressedData> results;
    results.reserve(variants.size());

    // Check if this is a sequential pattern
    bool is_sequential = is_sequential_pattern(variants);

    if (is_sequential) {
        // Optimized batch processing for sequential access
        // Process in chunks to maximize cache efficiency
        const size_t chunk_size = 16;  // Process 16 variants at a time

        for (size_t chunk_start = 0; chunk_start < variants.size(); chunk_start += chunk_size) {
            size_t chunk_end = std::min(chunk_start + chunk_size, variants.size());

            // Prefetch metadata for the entire chunk
            if (enable_prefetch_) {
                for (size_t i = chunk_start; i < chunk_end; ++i) {
                    prefetch_read(&variants[i]);
                }
            }

            // Process chunk with prefetching
            for (size_t i = chunk_start; i < chunk_end; ++i) {
                // Prefetch next variant in chunk
                if (i + 1 < chunk_end && config_.enable_readahead) {
                    const auto& next_variant = variants[i + 1];
                    prefetch_next(next_variant.offset, next_variant.compressed_size);
                }

                // Decompress current variant
                results.push_back(decompress(variants[i]));
            }
        }
    } else {
        // Non-sequential access - process without prefetching
        for (const auto& variant : variants) {
            results.push_back(decompress(variant));
        }
    }

    return results;
}

VariantDecompressor::Statistics SequentialDecompressor::get_statistics() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void SequentialDecompressor::reset_statistics() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_ = Statistics();
}

bool SequentialDecompressor::is_sequential_pattern(
    const std::vector<CompressedVariant>& variants) const {
    if (variants.size() < 2) {
        return true;  // Single variant is trivially sequential
    }

    // Fast check: if all variants have nullptr data, they're likely from index
    bool all_from_file = std::all_of(variants.begin(), variants.end(),
                                     [](const CompressedVariant& v) { return v.data == nullptr; });
    if (!all_from_file) {
        return false;  // Mixed sources, not sequential
    }

    // Calculate statistics for smarter detection
    std::vector<uint64_t> gaps;
    gaps.reserve(variants.size() - 1);

    uint64_t total_gap = 0;
    uint64_t min_gap = std::numeric_limits<uint64_t>::max();
    uint64_t max_gap = 0;

    for (size_t i = 1; i < variants.size(); ++i) {
        uint64_t current_offset = variants[i].offset;
        uint64_t prev_end = variants[i - 1].offset + variants[i - 1].compressed_size;

        // Check if current variant starts after previous one
        if (current_offset < prev_end) {
            return false;  // Overlapping or backward access
        }

        uint64_t gap = current_offset - prev_end;
        gaps.push_back(gap);
        total_gap += gap;
        min_gap = std::min(min_gap, gap);
        max_gap = std::max(max_gap, gap);
    }

    // Calculate average gap
    uint64_t avg_gap = total_gap / gaps.size();

    // Sequential if:
    // 1. Gaps are reasonably consistent (max < 10x min)
    // 2. Average gap is within threshold
    // 3. No single gap exceeds threshold

    bool consistent_gaps = (min_gap == 0 || max_gap < 10 * min_gap);
    bool reasonable_avg = avg_gap <= config_.sequential_threshold;
    bool all_gaps_ok = max_gap <= config_.sequential_threshold;

    return consistent_gaps && reasonable_avg && all_gaps_ok;
}

bool SequentialDecompressor::prefetch_next(uint64_t offset, size_t size) {
    // Get the next buffer
    auto& next_buffer = double_buffer_.next();
    auto& next_info = double_buffer_.info[1 - double_buffer_.active_buffer];

    // If buffer is not large enough, get a new one
    if (!next_buffer.valid() || next_buffer.size() < size) {
        next_buffer = buffer_manager_->get_buffer(size);
        if (!next_buffer.valid()) {
            next_info.valid = false;
            return false;
        }
    }

    // Perform synchronous read
    size_t bytes_read = file_reader_->read_at(offset, next_buffer.data(), size);

    // Update metadata
    next_info.offset = offset;
    next_info.size = size;
    next_info.valid = (bytes_read == size);

    return next_info.valid;
}

bool SequentialDecompressor::has_prefetched_data(uint64_t offset, size_t size) const {
    const auto& current_info = double_buffer_.info[double_buffer_.active_buffer];
    return current_info.valid && current_info.offset == offset && current_info.size == size;
}

BufferHandle SequentialDecompressor::read_compressed_data(const CompressedVariant& variant) {
    // Check if we have prefetched data in the current buffer
    if (config_.enable_readahead && has_prefetched_data(variant.offset, variant.compressed_size)) {
        // Use the current buffer and swap for next iteration
        auto& current = double_buffer_.current();
        double_buffer_.swap();
        return std::move(current);
    }

    // No prefetched data available, do direct read
    auto& current_buffer = double_buffer_.current();
    auto& current_info = double_buffer_.info[double_buffer_.active_buffer];

    // Ensure buffer is large enough
    if (!current_buffer.valid() || current_buffer.size() < variant.compressed_size) {
        current_buffer = buffer_manager_->get_buffer(variant.compressed_size);
        if (!current_buffer.valid()) {
            // Return the invalid buffer handle
            return std::move(current_buffer);
        }
    }

    // Read data
    size_t bytes_read =
        file_reader_->read_at(variant.offset, current_buffer.data(), variant.compressed_size);

    if (bytes_read != variant.compressed_size) {
        current_info.valid = false;
        // Return an empty invalid buffer
        return BufferHandle(std::unique_ptr<uint8_t[]>(nullptr), 0, nullptr, nullptr);
    }

    // Update info
    current_info.offset = variant.offset;
    current_info.size = variant.compressed_size;
    current_info.valid = true;

    // OPTIMIZATION: For sequential read-ahead access, avoid unnecessary copy
    // by transferring ownership of the current buffer directly
    if (config_.enable_readahead && current_info.valid &&
        current_buffer.size() >= variant.compressed_size) {
        // Move the current buffer to result (transfers ownership)
        auto result = std::move(current_buffer);

        // Allocate a new buffer for current (will become next after swap)
        current_buffer = buffer_manager_->get_buffer(config_.readahead_buffer_size);
        if (!current_buffer.valid()) {
            // If we can't get a new buffer, restore the old one
            current_buffer = std::move(result);
            // Fall through to the copy path below
        } else {
            // Successfully transferred ownership
            return result;
        }
    }

    // Fallback path: Create a copy (for non-readahead or allocation failure)
    auto result_buffer = buffer_manager_->get_buffer(variant.compressed_size);
    if (!result_buffer.valid()) {
        return result_buffer;
    }
    // Copy the data
    std::memcpy(result_buffer.data(), current_buffer.data(), variant.compressed_size);

    return result_buffer;
}

DecompressedData SequentialDecompressor::decompress_data(const uint8_t* compressed,
                                                         size_t compressed_size,
                                                         size_t expected_size,
                                                         CompressionType compression_type,
                                                         uint64_t offset) {
    // Handle uncompressed data
    if (compression_type == CompressionType::None) {
        if (compressed_size != expected_size) {
            std::ostringstream oss;
            oss << "Uncompressed size mismatch: expected " << expected_size << ", got "
                << compressed_size;
            return DecompressedData(offset, DecompressedData::SIZE_MISMATCH, oss.str());
        }

        // Get buffer and copy data directly
        auto decomp_buffer = buffer_manager_->get_decompression_buffer(expected_size);
        if (!decomp_buffer.valid()) {
            return DecompressedData(offset, DecompressedData::MEMORY_ERROR,
                                    "Failed to allocate buffer for uncompressed data");
        }

        std::memcpy(decomp_buffer.data(), compressed, compressed_size);
        return DecompressedData(decomp_buffer.release(), expected_size, offset);
    }

    // Get decompression buffer
    auto decomp_buffer = buffer_manager_->get_decompression_buffer(expected_size);
    if (!decomp_buffer.valid()) {
        return DecompressedData(offset, DecompressedData::MEMORY_ERROR,
                                "Failed to allocate decompression buffer");
    }

    // Perform decompression
    CompressionResult result;

    switch (compression_type) {
        case CompressionType::Zlib:
            result =
                decompress_zlib(compressed, compressed_size, decomp_buffer.data(), expected_size);
            break;

        case CompressionType::Zstd:
            result =
                decompress_zstd(compressed, compressed_size, decomp_buffer.data(), expected_size);
            break;

        default:
            return DecompressedData(offset, DecompressedData::UNSUPPORTED_COMPRESSION,
                                    "Unsupported compression type");
    }

    if (!result.success) {
        return DecompressedData(offset, DecompressedData::COMPRESSION_ERROR, result.error_message);
    }

    // Validate decompressed size if configured
    if (config_.validate_size && result.bytes_processed != expected_size) {
        std::ostringstream oss;
        oss << "Decompressed size mismatch: expected " << expected_size << ", got "
            << result.bytes_processed;
        return DecompressedData(offset, DecompressedData::SIZE_MISMATCH, oss.str());
    }

    // Release buffer from handle and wrap in unique_ptr for DecompressedData
    return DecompressedData(decomp_buffer.release(), result.bytes_processed, offset);
}

void SequentialDecompressor::update_statistics(const CompressedVariant& variant, bool success) {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    stats_.total_variants++;
    if (success) {
        stats_.successful_decompressions++;
    } else {
        stats_.failed_decompressions++;
    }

    stats_.total_compressed_bytes += variant.compressed_size;
    if (success) {
        stats_.total_decompressed_bytes += variant.uncompressed_size;
    }

    switch (variant.compression_type) {
        case CompressionType::None:
            stats_.uncompressed_variants++;
            break;
        case CompressionType::Zlib:
            stats_.zlib_variants++;
            break;
        case CompressionType::Zstd:
            stats_.zstd_variants++;
            break;
        default:
            break;
    }
}

// DecompressionContexts implementation
SequentialDecompressor::DecompressionContexts::~DecompressionContexts() {
    if (zstd_context) {
        ZSTD_freeDCtx(static_cast<ZSTD_DCtx*>(zstd_context));
    }
    if (zlib_stream) {
        inflateEnd(static_cast<z_stream*>(zlib_stream));
        delete static_cast<z_stream*>(zlib_stream);
    }
}

bool SequentialDecompressor::DecompressionContexts::ensure_zlib() {
    if (!zlib_stream) {
        z_stream* stream = new z_stream();
        std::memset(stream, 0, sizeof(z_stream));

        if (inflateInit(stream) != Z_OK) {
            delete stream;
            return false;
        }

        zlib_stream = stream;
    }
    return true;
}

bool SequentialDecompressor::DecompressionContexts::ensure_zstd() {
    if (!zstd_context) {
        zstd_context = ZSTD_createDCtx();
        if (!zstd_context) {
            return false;
        }
    }
    return true;
}

// Direct decompression implementation
DecompressedData SequentialDecompressor::decompress_direct(const CompressedVariant& variant) {
    // Ensure direct read buffer is large enough
    if (variant.compressed_size > direct_read_buffer_size_) {
        // Round up to power of 2
        size_t new_size = 1;
        while (new_size < variant.compressed_size) {
            new_size *= 2;
        }
        direct_read_buffer_.reset(new uint8_t[new_size]);
        direct_read_buffer_size_ = new_size;
    }

    // Allocate output buffer
    auto output_buffer = buffer_manager_->get_decompression_buffer(variant.uncompressed_size);
    if (!output_buffer.valid()) {
        return DecompressedData(variant.offset, DecompressedData::MEMORY_ERROR,
                                "Failed to allocate output buffer");
    }

    // Prefetch output buffer for writing
    if (enable_prefetch_) {
        for (size_t i = 0; i < variant.uncompressed_size; i += prefetch_distance_) {
            prefetch_write(output_buffer.data() + i);
        }
    }

    // Read compressed data directly
    size_t bytes_read =
        file_reader_->read_at(variant.offset, direct_read_buffer_.get(), variant.compressed_size);
    if (bytes_read != variant.compressed_size) {
        return DecompressedData(variant.offset, DecompressedData::MEMORY_ERROR,
                                "Failed to read compressed data");
    }

    // Prefetch compressed data for reading
    if (enable_prefetch_) {
        for (size_t i = 0; i < variant.compressed_size; i += prefetch_distance_) {
            prefetch_read(direct_read_buffer_.get() + i);
        }
    }

    // Direct decompression based on type
    bool success = false;
    size_t decompressed_size = 0;

    switch (variant.compression_type) {
        case CompressionType::None:
            // Direct copy for uncompressed
            if (variant.compressed_size != variant.uncompressed_size) {
                return DecompressedData(variant.offset, DecompressedData::SIZE_MISMATCH,
                                        "Uncompressed size mismatch");
            }
            std::memcpy(output_buffer.data(), direct_read_buffer_.get(), variant.compressed_size);
            success = true;
            decompressed_size = variant.compressed_size;
            break;

        case CompressionType::Zlib: {
            // Use persistent context for better performance
            if (!decomp_contexts_.ensure_zlib()) {
                return DecompressedData(variant.offset, DecompressedData::COMPRESSION_ERROR,
                                        "Failed to initialize zlib context");
            }

            z_stream* stream = static_cast<z_stream*>(decomp_contexts_.zlib_stream);

            // Reset stream for new decompression
            if (inflateReset(stream) != Z_OK) {
                return DecompressedData(variant.offset, DecompressedData::COMPRESSION_ERROR,
                                        "Failed to reset zlib stream");
            }

            // Set input and output
            stream->next_in = direct_read_buffer_.get();
            stream->avail_in = variant.compressed_size;
            stream->next_out = output_buffer.data();
            stream->avail_out = variant.uncompressed_size;

            // Perform decompression
            int ret = inflate(stream, Z_FINISH);

            if (ret == Z_STREAM_END) {
                success = true;
                decompressed_size = variant.uncompressed_size - stream->avail_out;
            }
            break;
        }

        case CompressionType::Zstd: {
            // Use persistent context
            if (!decomp_contexts_.ensure_zstd()) {
                return DecompressedData(variant.offset, DecompressedData::COMPRESSION_ERROR,
                                        "Failed to initialize zstd context");
            }

            ZSTD_DCtx* dctx = static_cast<ZSTD_DCtx*>(decomp_contexts_.zstd_context);

            // Perform decompression
            size_t result =
                ZSTD_decompressDCtx(dctx, output_buffer.data(), variant.uncompressed_size,
                                    direct_read_buffer_.get(), variant.compressed_size);

            if (!ZSTD_isError(result)) {
                success = true;
                decompressed_size = result;
            }
            break;
        }

        default:
            return DecompressedData(variant.offset, DecompressedData::UNSUPPORTED_COMPRESSION,
                                    "Unsupported compression type");
    }

    if (!success) {
        return DecompressedData(variant.offset, DecompressedData::COMPRESSION_ERROR,
                                "Decompression failed");
    }

    // Validate size if configured
    if (config_.validate_size && decompressed_size != variant.uncompressed_size) {
        return DecompressedData(variant.offset, DecompressedData::SIZE_MISMATCH,
                                "Decompressed size mismatch");
    }

    update_statistics(variant, true);
    return DecompressedData(output_buffer.release(), decompressed_size, variant.offset);
}

// OPTIMIZATION: Fast path for uncompressed data that skips buffer manager
DecompressedData SequentialDecompressor::decompress_uncompressed_direct(
    const CompressedVariant& variant) {
    // Validate size
    if (variant.compressed_size != variant.uncompressed_size) {
        return DecompressedData(variant.offset, DecompressedData::SIZE_MISMATCH,
                                "Uncompressed size mismatch");
    }

    // Allocate output buffer directly (bypass buffer manager for uncompressed)
    std::unique_ptr<uint8_t[]> output(new uint8_t[variant.uncompressed_size]);
    if (!output) {
        return DecompressedData(variant.offset, DecompressedData::MEMORY_ERROR,
                                "Failed to allocate memory for uncompressed data");
    }

    // Direct read into output buffer - no intermediate buffers needed
    size_t bytes_read =
        file_reader_->read_at(variant.offset, output.get(), variant.compressed_size);
    if (bytes_read != variant.compressed_size) {
        return DecompressedData(variant.offset, DecompressedData::MEMORY_ERROR,
                                "Failed to read uncompressed data");
    }

    update_statistics(variant, true);
    return DecompressedData(std::move(output), variant.uncompressed_size, variant.offset);
}

}  // namespace decompress
}  // namespace bgen
}  // namespace ldcov