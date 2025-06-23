#ifndef LDCOV_IO_BGEN_DECOMPRESS_DECOMPRESSOR_INTERFACE_H
#define LDCOV_IO_BGEN_DECOMPRESS_DECOMPRESSOR_INTERFACE_H

#include <cstdint>
#include <cstddef>
#include <memory>

namespace ldcov {
namespace io {
namespace bgen {

// Forward declarations
class BufferManager;

enum class CompressionType : uint8_t {
    None = 0,
    Zlib = 1,
    Zstd = 2
};

// Simple decompressed data holder
struct DecompressedData {
    std::unique_ptr<uint8_t[]> buffer;
    size_t size;
    
    DecompressedData() : buffer(nullptr), size(0) {}
    
    DecompressedData(std::unique_ptr<uint8_t[]> buf, size_t sz)
        : buffer(std::move(buf)), size(sz) {}
    
    // Move semantics
    DecompressedData(DecompressedData&&) = default;
    DecompressedData& operator=(DecompressedData&&) = default;
    
    // Delete copy
    DecompressedData(const DecompressedData&) = delete;
    DecompressedData& operator=(const DecompressedData&) = delete;
    
    uint8_t* data() { return buffer.get(); }
    const uint8_t* data() const { return buffer.get(); }
};

// Simple decompressor interface
class IDecompressor {
public:
    virtual ~IDecompressor() = default;
    
    virtual DecompressedData decompress(
        const uint8_t* compressed_data,
        size_t compressed_size,
        CompressionType compression,
        size_t expected_size
    ) = 0;
    
    virtual void reset() = 0;
};

} // namespace bgen
} // namespace io
} // namespace ldcov

#endif // LDCOV_IO_BGEN_DECOMPRESS_DECOMPRESSOR_INTERFACE_H