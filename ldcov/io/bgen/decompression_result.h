#ifndef LDCOV_BGEN_DECOMPRESSION_RESULT_H
#define LDCOV_BGEN_DECOMPRESSION_RESULT_H

#include <cstdint>
#include <cstddef>

namespace ldcov {
namespace bgen {

// Unified zero-copy result type for all decompressors
struct DecompressionResult {
    uint64_t offset;           // File offset of this variant
    const uint8_t* data;       // Pointer to decompressed data (owned by decompressor)
    size_t size;              // Actual size of decompressed data
    bool success;             // Whether decompression succeeded
    uint8_t error_code;       // Error code (0=success, 1=read_error, 2=decompress_error)
    
    // Error codes
    static constexpr uint8_t SUCCESS = 0;
    static constexpr uint8_t READ_ERROR = 1;
    static constexpr uint8_t DECOMPRESS_ERROR = 2;
    static constexpr uint8_t INVALID_COMPRESSION = 3;
};

} // namespace bgen
} // namespace ldcov

#endif // LDCOV_BGEN_DECOMPRESSION_RESULT_H