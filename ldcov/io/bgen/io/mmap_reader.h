#ifndef LDCOV_IO_BGEN_IO_MMAP_READER_H
#define LDCOV_IO_BGEN_IO_MMAP_READER_H

#include "reader_interface.h"
#include <string>
#include <vector>

namespace ldcov {
namespace io {
namespace bgen {

/**
 * Memory-mapped file reader implementation.
 * 
 * This class provides efficient file access using memory mapping (mmap),
 * which allows the operating system to map file contents directly into
 * the process's address space for zero-copy access.
 */
class MMapReader : public FileReader {
public:
    MMapReader();
    ~MMapReader() override;

    // Disable copy constructor and assignment operator
    MMapReader(const MMapReader&) = delete;
    MMapReader& operator=(const MMapReader&) = delete;

    // Enable move constructor and assignment operator
    MMapReader(MMapReader&& other) noexcept;
    MMapReader& operator=(MMapReader&& other) noexcept;

    // FileReader interface implementation
    size_t read(uint8_t* buffer, size_t size) override;
    size_t read_at(uint64_t offset, uint8_t* buffer, size_t size) override;
    void seek(uint64_t offset) override;
    uint64_t tell() const override;
    uint64_t size() const override;
    bool is_open() const override;
    void close() override;
    const std::string& filename() const override;

    // MMapReader-specific methods
    bool open(const std::string& filepath);
    std::vector<uint8_t> read_range(uint64_t offset, size_t length);
    const uint8_t* data(size_t offset = 0) const;
    bool supports_direct_access() const { return true; }

private:
    int fd_;                    // File descriptor
    void* mapped_data_;         // Pointer to memory-mapped data
    size_t file_size_;          // Size of the file
    std::string filepath_;      // Path to the currently open file
    mutable uint64_t position_; // Current read position for FileReader interface

    // Helper methods
    void cleanup();
};

} // namespace bgen
} // namespace io
} // namespace ldcov

#endif // LDCOV_IO_BGEN_IO_MMAP_READER_H