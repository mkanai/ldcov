#ifndef LDCOV_BGEN_FORMAT_BGEN_HEADER_H
#define LDCOV_BGEN_FORMAT_BGEN_HEADER_H

#include <cstdint>
#include <string>
#include <vector>
#include <stdexcept>
#include <cstring>

namespace ldcov {
namespace bgen {

// Utility function to read little-endian values
template<typename T>
inline T readLE(const uint8_t* buffer) {
    T value;
    std::memcpy(&value, buffer, sizeof(T));
    // Assume little-endian system (true for x86/ARM)
    return value;
}

// BGEN file format constants
enum class CompressionType : uint8_t {
    None = 0,
    Zlib = 1,
    Zstd = 2
};

enum class LayoutType : uint8_t {
    V11 = 1,  // Version 1.1
    V12 = 2   // Version 1.2
};

// Structure to hold BGEN header information
struct BgenHeader {
    uint32_t offset;           // Offset to variant data
    uint32_t nvariants;        // Number of variants
    uint32_t nsamples;         // Number of samples
    uint32_t flags;            // Header flags
    CompressionType compression;
    LayoutType layout;
    bool has_sample_ids;
    
    // Constructor
    BgenHeader() : offset(0), nvariants(0), nsamples(0), flags(0),
                   compression(CompressionType::None), 
                   layout(LayoutType::V12), 
                   has_sample_ids(false) {}
};

// BGEN header parser class
class BgenHeaderParser {
public:
    /**
     * Parse BGEN header from buffer
     * @param buffer Pointer to header data
     * @param size Size of buffer
     * @return Parsed header information
     * @throws std::runtime_error if parsing fails
     */
    static BgenHeader parse(const uint8_t* buffer, size_t size);
    
    /**
     * Get the size of the header in bytes
     * @param buffer Pointer to start of file
     * @param size Size of buffer (must be at least 8 bytes)
     * @return Header size including the initial offset and length fields
     * @throws std::runtime_error if buffer is too small
     */
    static size_t getHeaderSize(const uint8_t* buffer, size_t size);
    
    /**
     * Check if the file has valid BGEN magic number
     * @param buffer Pointer to header data
     * @param size Size of buffer
     * @return true if valid BGEN file
     */
    static bool isValidBgen(const uint8_t* buffer, size_t size);
    
private:
    // Helper to read little-endian integers
    template<typename T>
    static T readLE(const uint8_t* ptr) {
        T value = 0;
        for (size_t i = 0; i < sizeof(T); ++i) {
            value |= static_cast<T>(ptr[i]) << (8 * i);
        }
        return value;
    }
};

// Structure to hold sample block information
struct SampleBlock {
    std::vector<std::string> sample_ids;
    uint32_t block_size;  // Total size of sample block
    
    SampleBlock() : block_size(0) {}
};

// Sample block parser class
class SampleBlockParser {
public:
    /**
     * Parse sample block from buffer
     * @param buffer Pointer to sample block data
     * @param size Size of buffer
     * @param expected_samples Expected number of samples
     * @return Parsed sample information
     * @throws std::runtime_error if parsing fails
     */
    static SampleBlock parse(const uint8_t* buffer, size_t size, uint32_t expected_samples);
    
    /**
     * Get the size of the sample block
     * @param buffer Pointer to sample block start
     * @param size Size of buffer (must be at least 4 bytes)
     * @return Sample block size
     */
    static uint32_t getSampleBlockSize(const uint8_t* buffer, size_t size);
};

} // namespace bgen
} // namespace ldcov

#endif // LDCOV_BGEN_FORMAT_BGEN_HEADER_H