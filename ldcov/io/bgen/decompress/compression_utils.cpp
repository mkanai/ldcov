#include "compression_utils.h"
#include <iostream>
#include "decompressor.h"
#include <cstring>
#include <algorithm>
#include <sstream>

// Include vendored compression libraries
extern "C" {
#include "zlib.h"  // Will be provided by CMake include paths
}
#include "zstd.h"  // Will be provided by CMake include paths

namespace ldcov {
namespace bgen {
namespace decompress {

namespace {
    // Zlib header magic numbers
    constexpr uint8_t ZLIB_DEFLATE_METHOD = 0x78;
    constexpr uint8_t ZLIB_COMPRESSION_LEVELS[] = {0x01, 0x5E, 0x9C, 0xDA};
    
    // Zstd magic number
    constexpr uint32_t ZSTD_MAGIC = 0xFD2FB528;
    
    // Helper to check zlib header
    bool is_zlib_header(const uint8_t* data, size_t size) {
        if (size < 2) return false;
        
        // Check first byte is deflate method
        if (data[0] != ZLIB_DEFLATE_METHOD) return false;
        
        // Check second byte is valid compression level
        for (uint8_t level : ZLIB_COMPRESSION_LEVELS) {
            if (data[1] == level) return true;
        }
        return false;
    }
    
    // Helper to check zstd header
    bool is_zstd_header(const uint8_t* data, size_t size) {
        if (size < 4) return false;
        
        uint32_t magic = *reinterpret_cast<const uint32_t*>(data);
        // Handle endianness
        return magic == ZSTD_MAGIC || magic == __builtin_bswap32(ZSTD_MAGIC);
    }
}

CompressionType detect_compression_type(const uint8_t* data, size_t size) {
    if (!data || size < 4) {
        return CompressionType::Unknown;
    }
    
    // Check for zstd magic number
    if (is_zstd_header(data, size)) {
        return CompressionType::Zstd;
    }
    
    // Check for zlib header
    if (is_zlib_header(data, size)) {
        return CompressionType::Zlib;
    }
    
    // Could be uncompressed or unknown format
    return CompressionType::Unknown;
}

CompressionResult decompress_zlib(const uint8_t* compressed, size_t compressed_size,
                                  uint8_t* output, size_t output_size) {
    if (!compressed || !output || compressed_size == 0 || output_size == 0) {
        return CompressionResult(false, "Invalid input parameters");
    }
    
    // Validate input parameters
    
    
    // Initialize zlib stream
    z_stream stream;
    std::memset(&stream, 0, sizeof(stream));
    
    stream.next_in = const_cast<Bytef*>(compressed);
    stream.avail_in = static_cast<uInt>(compressed_size);
    stream.next_out = output;
    stream.avail_out = static_cast<uInt>(output_size);
    
    // Initialize inflation based on format
    // Check for zlib header to determine format
    // BGEN v1.1 uses standard zlib (with header)
    // BGEN v1.2 uses raw deflate (no header)
    int ret;
    if (compressed_size >= 2 && compressed[0] == 0x78 && 
        (compressed[1] == 0x01 || compressed[1] == 0x5E || 
         compressed[1] == 0x9C || compressed[1] == 0xDA)) {
        // Standard zlib format detected (v1.1)
        ret = inflateInit(&stream);
    } else {
        // Raw deflate format (v1.2)
        ret = inflateInit2(&stream, -15);
    }
    
    if (ret != Z_OK) {
        std::stringstream ss;
        ss << "Failed to initialize zlib decompression: " << ret;
        if (stream.msg) ss << " (" << stream.msg << ")";
        return CompressionResult(false, ss.str());
    }
    
    // Perform decompression
    ret = inflate(&stream, Z_FINISH);
    size_t bytes_written = output_size - stream.avail_out;
    
    
    // Clean up
    inflateEnd(&stream);
    
    if (ret == Z_STREAM_END) {
        return CompressionResult(true, "", bytes_written);
    } else if (ret == Z_OK) {
        // Partial decompression (output buffer too small)
        return CompressionResult(false, "Output buffer too small", bytes_written);
    } else {
        // Error
        std::stringstream ss;
        ss << "Zlib decompression failed: " << ret;
        if (stream.msg) ss << " (" << stream.msg << ")";
        return CompressionResult(false, ss.str(), bytes_written);
    }
}

CompressionResult decompress_zstd(const uint8_t* compressed, size_t compressed_size,
                                  uint8_t* output, size_t output_size) {
    if (!compressed || !output || compressed_size == 0 || output_size == 0) {
        return CompressionResult(false, "Invalid input parameters");
    }
    
    // Perform decompression
    size_t result = ZSTD_decompress(output, output_size, compressed, compressed_size);
    
    if (ZSTD_isError(result)) {
        std::stringstream ss;
        ss << "Zstd decompression failed: " << ZSTD_getErrorName(result);
        return CompressionResult(false, ss.str());
    }
    
    // Check if output buffer was large enough
    if (result > output_size) {
        return CompressionResult(false, "Output buffer too small", 0);
    }
    
    return CompressionResult(true, "", result);
}

std::pair<std::unique_ptr<uint8_t[]>, CompressionResult> 
decompress_with_allocation(const uint8_t* compressed, size_t compressed_size,
                          size_t expected_size, CompressionType compression_type) {
    // Validate inputs
    if (!compressed || compressed_size == 0 || expected_size == 0) {
        return {nullptr, CompressionResult(false, "Invalid input parameters")};
    }
    
    // Allocate output buffer
    auto buffer = std::unique_ptr<uint8_t[]>(new uint8_t[expected_size]);
    if (!buffer) {
        return {nullptr, CompressionResult(false, "Failed to allocate output buffer")};
    }
    
    // Perform decompression based on type
    CompressionResult result;
    switch (compression_type) {
        case CompressionType::None:
            // Uncompressed data is not supported
            return {nullptr, CompressionResult(false, 
                "Uncompressed BGEN data is not supported. "
                "Please use compressed BGEN files (zlib or zstd).")};
            break;
            
        case CompressionType::Zlib:
            result = decompress_zlib(compressed, compressed_size, buffer.get(), expected_size);
            break;
            
        case CompressionType::Zstd:
            result = decompress_zstd(compressed, compressed_size, buffer.get(), expected_size);
            break;
            
        default:
            return {nullptr, CompressionResult(false, "Unknown compression type")};
    }
    
    if (!result.success) {
        return {nullptr, result};
    }
    
    return {std::move(buffer), result};
}

const char* get_compression_type_name(CompressionType type) {
    switch (type) {
        case CompressionType::None: return "none";
        case CompressionType::Zlib: return "zlib";
        case CompressionType::Zstd: return "zstd";
        case CompressionType::Unknown: return "unknown";
        default: return "invalid";
    }
}

bool validate_compressed_header(const uint8_t* data, size_t size, CompressionType type) {
    if (!data || size == 0) return false;
    
    switch (type) {
        case CompressionType::None:
            return true;  // No header to validate
            
        case CompressionType::Zlib:
            return is_zlib_header(data, size);
            
        case CompressionType::Zstd:
            return is_zstd_header(data, size);
            
        default:
            return false;
    }
}

size_t get_decompressed_size_hint(const uint8_t* data, size_t size, CompressionType type) {
    if (!data || size == 0) return 0;
    
    switch (type) {
        case CompressionType::None:
            return size;  // Uncompressed size is same as input
            
        case CompressionType::Zlib:
            // Zlib doesn't store uncompressed size in header
            return 0;
            
        case CompressionType::Zstd:
            // Zstd can store frame content size (minimum frame header is 6 bytes)
            if (size >= 6) {
                unsigned long long content_size = ZSTD_getFrameContentSize(data, size);
                if (content_size != ZSTD_CONTENTSIZE_UNKNOWN && 
                    content_size != ZSTD_CONTENTSIZE_ERROR) {
                    return static_cast<size_t>(content_size);
                }
            }
            return 0;
            
        default:
            return 0;
    }
}

bool is_compression_supported(CompressionType type) {
    switch (type) {
        case CompressionType::None:
            return false;  // Uncompressed data is not supported
            
        case CompressionType::Zlib:
            // Check if zlib is available
            return true;  // We have vendored zlib-ng
            
        case CompressionType::Zstd:
            // Check if zstd is available
            return true;  // We have vendored zstd
            
        default:
            return false;
    }
}

bool initialize_compression_libraries() {
    // Both zlib-ng and zstd are statically linked and don't require
    // explicit initialization in our case
    return true;
}

std::string get_compression_library_version(CompressionType type) {
    switch (type) {
        case CompressionType::None:
            return "N/A";
            
        case CompressionType::Zlib:
            return zlibVersion();
            
        case CompressionType::Zstd:
            return ZSTD_versionString();
            
        default:
            return "Unknown";
    }
}

} // namespace decompress
} // namespace bgen
} // namespace ldcov