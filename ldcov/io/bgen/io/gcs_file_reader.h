#ifndef LDCOV_BGEN_GCS_FILE_READER_H
#define LDCOV_BGEN_GCS_FILE_READER_H

#include <Python.h>

#include <memory>
#include <utility>
#include <vector>

#include "reader_interface.h"

namespace ldcov {
namespace io {
namespace bgen {

/**
 * GCSFileReader - File reader implementation for Google Cloud Storage
 *
 * Uses Python's gcsfs library to read BGEN files directly from GCS
 * without downloading the entire file. Supports efficient range requests
 * and buffering for sequential access patterns.
 */
class GCSFileReader : public FileReader {
   public:
    /**
     * Constructor
     * @param filename GCS path (must start with gs://)
     * @param buffer_size Read buffer size in bytes (default 10MB for better GCS performance)
     */
    explicit GCSFileReader(const std::string& filename, size_t buffer_size = 10 * 1024 * 1024);

    ~GCSFileReader() override;

    // FileReader interface implementation
    size_t read(uint8_t* buffer, size_t size) override;
    size_t read_at(uint64_t offset, uint8_t* buffer, size_t size) override;
    void seek(uint64_t offset) override;
    uint64_t tell() const override;
    uint64_t size() const override;
    bool is_open() const override;
    void close() override;
    const std::string& filename() const override {
        return filename_;
    }

   private:
    // Initialize Python and import required modules
    void initialize_python();

    // Create gcsfs filesystem and open file
    void open_file();

    // Read data using gcsfs
    size_t read_internal(uint64_t offset, uint8_t* buffer, size_t size);

    // Buffer management for sequential reads
    void fill_buffer(uint64_t offset);
    size_t read_from_buffer(uint8_t* buffer, size_t size);

    // Python objects (owned references)
    PyObject* gcsfs_module_;
    PyObject* fs_;        // gcsfs.GCSFileSystem instance
    PyObject* file_obj_;  // File handle from fs.open()

    // File information
    std::string filename_;
    uint64_t file_size_;
    uint64_t current_pos_;
    bool is_open_;

    // Buffering for sequential reads
    std::vector<uint8_t> buffer_;
    size_t buffer_size_;
    uint64_t buffer_start_;  // File offset where buffer starts
    size_t buffer_valid_;    // Number of valid bytes in buffer

    // Python GIL state for thread safety
    PyGILState_STATE gil_state_;

    // Error handling
    void check_python_error(const std::string& operation);

    // Disable copy operations
    GCSFileReader(const GCSFileReader&) = delete;
    GCSFileReader& operator=(const GCSFileReader&) = delete;
};

}  // namespace bgen
}  // namespace io
}  // namespace ldcov

#endif  // LDCOV_BGEN_GCS_FILE_READER_H