#include "sequential_fast_path.h"
#include <cstring>
#include <zlib.h>
#include <zstd.h>

namespace ldcov {
namespace bgen {
namespace decompress {

SequentialFastPath::SequentialFastPath(ldcov::io::bgen::FileReader* reader, const Config& config)
    : file_reader_(reader)
    , config_(config)
    , read_buffer_(nullptr)
    , read_buffer_size_(0)
    , zlib_stream_(nullptr)
    , zstd_dctx_(nullptr) {
    
    // Pre-allocate read buffer
    ensure_read_buffer(config_.initial_buffer_size);
    
    // Initialize decompression contexts
    // For zlib, we'll create the stream on demand
    // For zstd, create a reusable context
    zstd_dctx_ = ZSTD_createDCtx();
    
    // Clear error buffer
    last_error_[0] = '\0';
}

SequentialFastPath::~SequentialFastPath() {
    if (zstd_dctx_) {
        ZSTD_freeDCtx(static_cast<ZSTD_DCtx*>(zstd_dctx_));
    }
    if (zlib_stream_) {
        inflateEnd(static_cast<z_stream*>(zlib_stream_));
        delete static_cast<z_stream*>(zlib_stream_);
    }
}

void SequentialFastPath::ensure_read_buffer(size_t size) {
    if (size > read_buffer_size_) {
        // Round up to next power of 2 for better allocation
        size_t new_size = 1;
        while (new_size < size) {
            new_size *= 2;
        }
        read_buffer_.reset(new uint8_t[new_size]);
        read_buffer_size_ = new_size;
    }
}

bool SequentialFastPath::decompress_variant(
    const uint8_t* compressed,
    size_t compressed_size,
    size_t uncompressed_size,
    CompressionType type,
    uint8_t* output) {
    
    switch (type) {
        case CompressionType::None:
            // Handle uncompressed data - direct copy
            if (compressed_size != uncompressed_size) {
                snprintf(last_error_, ERROR_MSG_SIZE, 
                        "Uncompressed size mismatch: expected %zu, got %zu", 
                        uncompressed_size, compressed_size);
                return false;
            }
            std::memcpy(output, compressed, compressed_size);
            return true;
            
        case CompressionType::Zlib: {
            // Use raw inflate for speed (no stream reuse for simplicity)
            uLongf dest_len = uncompressed_size;
            int ret = uncompress(reinterpret_cast<Bytef*>(output), &dest_len,
                               reinterpret_cast<const Bytef*>(compressed), compressed_size);
            
            if (ret != Z_OK || dest_len != uncompressed_size) {
                snprintf(last_error_, ERROR_MSG_SIZE, 
                        "Zlib decompression failed: %d", ret);
                return false;
            }
            return true;
        }
        
        case CompressionType::Zstd: {
            // Use pre-created context for better performance
            size_t ret = ZSTD_decompressDCtx(
                static_cast<ZSTD_DCtx*>(zstd_dctx_),
                output, uncompressed_size,
                compressed, compressed_size
            );
            
            if (ZSTD_isError(ret) || ret != uncompressed_size) {
                snprintf(last_error_, ERROR_MSG_SIZE, 
                        "Zstd decompression failed: %s", 
                        ZSTD_isError(ret) ? ZSTD_getErrorName(ret) : "size mismatch");
                return false;
            }
            return true;
        }
        
        default:
            snprintf(last_error_, ERROR_MSG_SIZE, 
                    "Unknown compression type: %d", static_cast<int>(type));
            return false;
    }
}

bool SequentialFastPath::decompress_direct(
    uint64_t file_offset,
    size_t compressed_size,
    size_t uncompressed_size,
    CompressionType compression_type,
    uint8_t* output) {
    
    // Ensure buffer is large enough
    ensure_read_buffer(compressed_size);
    
    // Prefetch output buffer for writing
    if (config_.enable_prefetch) {
        for (size_t i = 0; i < uncompressed_size; i += config_.prefetch_distance) {
            prefetch_write(output + i);
        }
    }
    
    // Read compressed data
    size_t bytes_read = file_reader_->read_at(file_offset, read_buffer_.get(), compressed_size);
    if (bytes_read != compressed_size) {
        snprintf(last_error_, ERROR_MSG_SIZE, 
                "Read failed: expected %zu bytes, got %zu", compressed_size, bytes_read);
        return false;
    }
    
    // Prefetch compressed data for reading
    if (config_.enable_prefetch) {
        for (size_t i = 0; i < compressed_size; i += config_.prefetch_distance) {
            prefetch_read(read_buffer_.get() + i);
        }
    }
    
    // Decompress directly into output
    return decompress_variant(read_buffer_.get(), compressed_size, 
                            uncompressed_size, compression_type, output);
}

size_t SequentialFastPath::decompress_batch_direct(
    const uint64_t* offsets,
    const size_t* compressed_sizes,
    const size_t* uncompressed_sizes,
    const CompressionType* compression_types,
    uint8_t** outputs,
    size_t n_variants) {
    
    size_t successful = 0;
    
    // Process variants sequentially
    for (size_t i = 0; i < n_variants; ++i) {
        // Prefetch next variant's metadata
        if (i + 1 < n_variants && config_.enable_prefetch) {
            prefetch_read(&offsets[i + 1]);
            prefetch_read(&compressed_sizes[i + 1]);
            prefetch_read(&uncompressed_sizes[i + 1]);
            prefetch_read(&compression_types[i + 1]);
        }
        
        if (decompress_direct(offsets[i], compressed_sizes[i], 
                            uncompressed_sizes[i], compression_types[i], outputs[i])) {
            successful++;
        } else {
            // Continue on error, but could also break here
            // depending on error handling policy
        }
    }
    
    return successful;
}

void SequentialFastPath::reset() {
    // Clear error
    last_error_[0] = '\0';
    
    // Reset zstd context for fresh state
    if (zstd_dctx_) {
        ZSTD_DCtx_reset(static_cast<ZSTD_DCtx*>(zstd_dctx_), ZSTD_reset_session_only);
    }
}

} // namespace decompress
} // namespace bgen
} // namespace ldcov