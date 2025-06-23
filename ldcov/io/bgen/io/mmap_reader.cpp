#include "mmap_reader.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <sstream>

namespace ldcov {
namespace io {
namespace bgen {

MMapReader::MMapReader()
    : fd_(-1)
    , mapped_data_(nullptr)
    , file_size_(0)
    , filepath_()
    , position_(0)
{
}

MMapReader::~MMapReader() {
    close();
}

MMapReader::MMapReader(MMapReader&& other) noexcept
    : fd_(other.fd_)
    , mapped_data_(other.mapped_data_)
    , file_size_(other.file_size_)
    , filepath_(std::move(other.filepath_))
    , position_(other.position_)
{
    // Reset the other object
    other.fd_ = -1;
    other.mapped_data_ = nullptr;
    other.file_size_ = 0;
    other.position_ = 0;
}

MMapReader& MMapReader::operator=(MMapReader&& other) noexcept {
    if (this != &other) {
        // Clean up current resources
        close();
        
        // Move resources
        fd_ = other.fd_;
        mapped_data_ = other.mapped_data_;
        file_size_ = other.file_size_;
        filepath_ = std::move(other.filepath_);
        position_ = other.position_;
        
        // Reset the other object
        other.fd_ = -1;
        other.mapped_data_ = nullptr;
        other.file_size_ = 0;
        other.position_ = 0;
    }
    return *this;
}

bool MMapReader::open(const std::string& filepath) {
    // Close any currently open file
    close();
    
    // Open the file
    fd_ = ::open(filepath.c_str(), O_RDONLY);
    if (fd_ == -1) {
        std::stringstream ss;
        ss << "Failed to open file '" << filepath << "': " << std::strerror(errno);
        // Log error but return false instead of throwing
        return false;
    }
    
    // Get file size
    struct stat st;
    if (fstat(fd_, &st) == -1) {
        std::stringstream ss;
        ss << "Failed to get file stats for '" << filepath << "': " << std::strerror(errno);
        cleanup();
        return false;
    }
    
    file_size_ = static_cast<size_t>(st.st_size);
    
    // Handle empty files
    if (file_size_ == 0) {
        filepath_ = filepath;
        return true;  // Successfully opened, but nothing to map
    }
    
    // Memory map the file
    mapped_data_ = mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mapped_data_ == MAP_FAILED) {
        std::stringstream ss;
        ss << "Failed to memory map file '" << filepath << "': " << std::strerror(errno);
        mapped_data_ = nullptr;
        cleanup();
        return false;
    }
    
    // Advise the kernel about our access pattern (sequential read)
    madvise(mapped_data_, file_size_, MADV_SEQUENTIAL);
    
    filepath_ = filepath;
    position_ = 0;  // Reset read position
    return true;
}

void MMapReader::close() {
    cleanup();
}

bool MMapReader::is_open() const {
    return fd_ != -1;
}

uint64_t MMapReader::size() const {
    return static_cast<uint64_t>(file_size_);
}

// FileReader interface implementation
size_t MMapReader::read(uint8_t* buffer, size_t size) {
    size_t bytes_read = read_at(position_, buffer, size);
    position_ += bytes_read;
    return bytes_read;
}

size_t MMapReader::read_at(uint64_t offset, uint8_t* buffer, size_t size) {
    if (!is_open()) {
        throw std::runtime_error("No file is open");
    }
    
    if (offset >= file_size_) {
        return 0;  // Nothing to read
    }
    
    // Adjust size if it would read past the end of file
    size_t bytes_to_read = std::min(size, static_cast<size_t>(file_size_ - offset));
    
    if (bytes_to_read == 0) {
        return 0;
    }
    
    if (mapped_data_ == nullptr && file_size_ > 0) {
        throw std::runtime_error("File is open but not memory mapped");
    }
    
    // Copy data from memory-mapped region
    if (mapped_data_ != nullptr) {
        const uint8_t* src = static_cast<const uint8_t*>(mapped_data_) + offset;
        std::memcpy(buffer, src, bytes_to_read);
    }
    
    return bytes_to_read;
}

void MMapReader::seek(uint64_t offset) {
    if (!is_open()) {
        throw std::runtime_error("No file is open");
    }
    position_ = offset;
}

uint64_t MMapReader::tell() const {
    return position_;
}

const std::string& MMapReader::filename() const {
    return filepath_;
}

// MMapReader-specific methods
std::vector<uint8_t> MMapReader::read_range(uint64_t offset, size_t length) {
    if (!is_open()) {
        throw std::runtime_error("No file is open");
    }
    
    if (offset >= file_size_) {
        return std::vector<uint8_t>();  // Empty vector
    }
    
    // Adjust length if it would read past the end of file
    size_t bytes_to_read = std::min(length, static_cast<size_t>(file_size_ - offset));
    
    std::vector<uint8_t> result(bytes_to_read);
    
    if (bytes_to_read > 0) {
        size_t bytes_read = read_at(offset, result.data(), bytes_to_read);
        result.resize(bytes_read);  // Adjust size if less was read
    }
    
    return result;
}

const uint8_t* MMapReader::data(size_t offset) const {
    if (!is_open()) {
        return nullptr;
    }
    
    if (offset >= file_size_) {
        return nullptr;
    }
    
    if (mapped_data_ == nullptr) {
        return nullptr;  // Empty file or not mapped
    }
    
    return static_cast<const uint8_t*>(mapped_data_) + offset;
}

void MMapReader::cleanup() {
    // Unmap the file if mapped
    if (mapped_data_ != nullptr && mapped_data_ != MAP_FAILED) {
        munmap(mapped_data_, file_size_);
        mapped_data_ = nullptr;
    }
    
    // Close the file descriptor
    if (fd_ != -1) {
        ::close(fd_);
        fd_ = -1;
    }
    
    // Reset state
    file_size_ = 0;
    filepath_.clear();
    position_ = 0;
}

} // namespace bgen
} // namespace io
} // namespace ldcov