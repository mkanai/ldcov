#ifndef LDCOV_BGEN_DECOMPRESS_COMPRESSION_UTILS_H
#define LDCOV_BGEN_DECOMPRESS_COMPRESSION_UTILS_H

#include <cstdint>
#include <cstddef>
#include <memory>
#include <string>

namespace ldcov {
namespace bgen {
namespace decompress {

// Forward declaration
enum class CompressionType : uint8_t;

/**
 * CompressionResult - Result of compression/decompression operations
 */
struct CompressionResult {
    bool success;
    std::string error_message;
    size_t bytes_processed;  // Actual bytes written/read
    
    CompressionResult(bool s = false, const std::string& msg = "", size_t bytes = 0)
        : success(s), error_message(msg), bytes_processed(bytes) {}
};

/**
 * Detect compression type from data
 * 
 * Examines the header bytes of compressed data to determine the compression type.
 * This is useful when the compression type is not explicitly specified.
 * 
 * @param data Pointer to compressed data
 * @param size Size of compressed data (at least 4 bytes needed)
 * @return Detected compression type (Unknown if cannot determine)
 */
CompressionType detect_compression_type(const uint8_t* data, size_t size);

/**
 * Decompress zlib-compressed data
 * 
 * @param compressed Pointer to compressed data
 * @param compressed_size Size of compressed data
 * @param output Output buffer for decompressed data
 * @param output_size Size of output buffer (must be large enough)
 * @return CompressionResult with success status and actual decompressed size
 */
CompressionResult decompress_zlib(const uint8_t* compressed, size_t compressed_size,
                                  uint8_t* output, size_t output_size);

/**
 * Decompress zstd-compressed data
 * 
 * @param compressed Pointer to compressed data
 * @param compressed_size Size of compressed data
 * @param output Output buffer for decompressed data
 * @param output_size Size of output buffer (must be large enough)
 * @return CompressionResult with success status and actual decompressed size
 */
CompressionResult decompress_zstd(const uint8_t* compressed, size_t compressed_size,
                                  uint8_t* output, size_t output_size);

/**
 * Decompress data with automatic buffer allocation
 * 
 * This is a convenience function that allocates the output buffer based on
 * the expected uncompressed size.
 * 
 * @param compressed Pointer to compressed data
 * @param compressed_size Size of compressed data
 * @param expected_size Expected size after decompression
 * @param compression_type Type of compression used
 * @return Pair of (buffer, result) where buffer is the allocated memory
 */
std::pair<std::unique_ptr<uint8_t[]>, CompressionResult> 
decompress_with_allocation(const uint8_t* compressed, size_t compressed_size,
                          size_t expected_size, CompressionType compression_type);

/**
 * Get compression type name
 * 
 * @param type Compression type
 * @return Human-readable name of compression type
 */
const char* get_compression_type_name(CompressionType type);

/**
 * Validate compressed data header
 * 
 * Performs basic validation on compressed data to ensure it has a valid header
 * for the specified compression type.
 * 
 * @param data Pointer to compressed data
 * @param size Size of compressed data
 * @param type Expected compression type
 * @return True if header appears valid
 */
bool validate_compressed_header(const uint8_t* data, size_t size, CompressionType type);

/**
 * Get decompressed size from compressed data (if available)
 * 
 * Some compression formats store the uncompressed size in their headers.
 * This function attempts to extract that information.
 * 
 * @param data Pointer to compressed data
 * @param size Size of compressed data
 * @param type Compression type
 * @return Decompressed size if available, 0 otherwise
 */
size_t get_decompressed_size_hint(const uint8_t* data, size_t size, CompressionType type);

/**
 * Check if compression library is available
 * 
 * @param type Compression type to check
 * @return True if the compression library is available and initialized
 */
bool is_compression_supported(CompressionType type);

/**
 * Initialize compression libraries
 * 
 * This function initializes any required compression libraries.
 * It's called automatically but can be called explicitly for eager initialization.
 * 
 * @return True if all libraries initialized successfully
 */
bool initialize_compression_libraries();

/**
 * Get compression library version
 * 
 * @param type Compression type
 * @return Version string of the compression library
 */
std::string get_compression_library_version(CompressionType type);

} // namespace decompress
} // namespace bgen
} // namespace ldcov

#endif // LDCOV_BGEN_DECOMPRESS_COMPRESSION_UTILS_H