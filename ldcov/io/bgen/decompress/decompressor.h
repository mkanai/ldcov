#ifndef LDCOV_BGEN_DECOMPRESS_DECOMPRESSOR_H
#define LDCOV_BGEN_DECOMPRESS_DECOMPRESSOR_H

#include <cstdint>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#include <stdexcept>

namespace ldcov {
namespace bgen {
namespace decompress {

// Forward declarations
class BufferManager;

/**
 * CompressionType - Supported compression types for BGEN variants
 */
enum class CompressionType : uint8_t {
    None = 0,
    Zlib = 1,
    Zstd = 2,
    Unknown = 255
};

/**
 * CompressedVariant - Input data for decompression
 * 
 * This struct holds the metadata and compressed data for a single variant.
 * The data pointer should remain valid throughout the decompression operation.
 */
struct CompressedVariant {
    // File offset of this variant (for error reporting and caching)
    uint64_t offset;
    
    // Pointer to compressed data (not owned by this struct)
    const uint8_t* data;
    
    // Size of compressed data in bytes
    size_t compressed_size;
    
    // Expected size after decompression (from BGEN metadata)
    size_t uncompressed_size;
    
    // Compression type
    CompressionType compression_type;
    
    // Optional variant identifier for error messages
    std::string variant_id;
    
    // Constructor
    CompressedVariant(uint64_t off, const uint8_t* d, size_t comp_size, 
                      size_t uncomp_size, CompressionType type)
        : offset(off), data(d), compressed_size(comp_size),
          uncompressed_size(uncomp_size), compression_type(type) {}
};

/**
 * DecompressedData - Output data from decompression with owned buffer
 * 
 * This struct owns the decompressed data buffer and provides RAII semantics.
 * The buffer is automatically freed when the struct is destroyed.
 */
struct DecompressedData {
    // Owned buffer containing decompressed data
    std::unique_ptr<uint8_t[]> buffer;
    
    // Actual size of decompressed data
    size_t size;
    
    // File offset of the source variant (for correlation)
    uint64_t offset;
    
    // Success flag
    bool success;
    
    // Error code if not successful
    enum ErrorCode : uint8_t {
        SUCCESS = 0,
        INVALID_INPUT = 1,
        COMPRESSION_ERROR = 2,
        SIZE_MISMATCH = 3,
        MEMORY_ERROR = 4,
        UNSUPPORTED_COMPRESSION = 5
    } error_code;
    
    // Error message for detailed diagnostics
    std::string error_message;
    
    // Default constructor (failed decompression)
    DecompressedData() 
        : buffer(nullptr), size(0), offset(0), success(false), 
          error_code(INVALID_INPUT) {}
    
    // Success constructor
    DecompressedData(std::unique_ptr<uint8_t[]> buf, size_t sz, uint64_t off)
        : buffer(std::move(buf)), size(sz), offset(off), success(true), 
          error_code(SUCCESS) {}
    
    // Error constructor
    DecompressedData(uint64_t off, ErrorCode code, const std::string& msg)
        : buffer(nullptr), size(0), offset(off), success(false), 
          error_code(code), error_message(msg) {}
    
    // Move constructor
    DecompressedData(DecompressedData&&) = default;
    
    // Move assignment
    DecompressedData& operator=(DecompressedData&&) = default;
    
    // Delete copy operations
    DecompressedData(const DecompressedData&) = delete;
    DecompressedData& operator=(const DecompressedData&) = delete;
    
    // Convenience methods
    uint8_t* data() { return buffer.get(); }
    const uint8_t* data() const { return buffer.get(); }
    bool is_valid() const { return success && buffer != nullptr; }
};

/**
 * DecompressionException - Exception type for decompression errors
 */
class DecompressionException : public std::runtime_error {
public:
    DecompressionException(const std::string& msg, DecompressedData::ErrorCode code)
        : std::runtime_error(msg), error_code_(code) {}
    
    DecompressedData::ErrorCode error_code() const { return error_code_; }
    
private:
    DecompressedData::ErrorCode error_code_;
};

/**
 * VariantDecompressor - Abstract base class for variant decompression
 * 
 * This interface defines the contract for all decompressor implementations.
 * Implementations may use different strategies (sequential, batch, adaptive)
 * but must provide the same interface for decompressing variants.
 */
class VariantDecompressor {
public:
    // Configuration for decompressor
    struct Config {
        // Buffer manager to use (optional, will create own if not provided)
        BufferManager* buffer_manager;
        
        // Enable compression type detection (if not specified in variant)
        bool auto_detect_compression;
        
        // Validate decompressed size matches expected size
        bool validate_size;
        
        // Maximum allowed decompressed size (safety limit)
        size_t max_decompressed_size;
        
        // Constructor with default values
        Config() 
            : buffer_manager(nullptr),
              auto_detect_compression(true),
              validate_size(true),
              max_decompressed_size(1024 * 1024 * 1024) // 1GB
        {}
    };
    
    // Constructor
    explicit VariantDecompressor(const Config& config = Config())
        : config_(config) {}
    
    // Virtual destructor
    virtual ~VariantDecompressor() = default;
    
    // Main decompression method
    // Takes a compressed variant and returns decompressed data
    virtual DecompressedData decompress(const CompressedVariant& variant) = 0;
    
    // Batch decompression (optional optimization)
    // Default implementation just calls decompress() for each variant
    virtual std::vector<DecompressedData> decompress_batch(
        const std::vector<CompressedVariant>& variants) {
        std::vector<DecompressedData> results;
        results.reserve(variants.size());
        for (const auto& variant : variants) {
            results.push_back(decompress(variant));
        }
        return results;
    }
    
    // Get statistics about decompression operations (optional)
    struct Statistics {
        uint64_t total_variants = 0;
        uint64_t successful_decompressions = 0;
        uint64_t failed_decompressions = 0;
        uint64_t total_compressed_bytes = 0;
        uint64_t total_decompressed_bytes = 0;
        uint64_t zlib_variants = 0;
        uint64_t zstd_variants = 0;
        uint64_t uncompressed_variants = 0;
    };
    
    virtual Statistics get_statistics() const {
        return Statistics();
    }
    
    // Reset statistics
    virtual void reset_statistics() {}
    
    // Get configuration
    const Config& get_config() const { return config_; }
    
protected:
    Config config_;
};

/**
 * Create a default decompressor
 * This factory function creates an appropriate decompressor based on
 * available implementations and system capabilities.
 */
std::unique_ptr<VariantDecompressor> create_decompressor(
    const VariantDecompressor::Config& config = VariantDecompressor::Config());

// Factory functions are declared in decompressor_factory.h

} // namespace decompress
} // namespace bgen
} // namespace ldcov

#endif // LDCOV_BGEN_DECOMPRESS_DECOMPRESSOR_H