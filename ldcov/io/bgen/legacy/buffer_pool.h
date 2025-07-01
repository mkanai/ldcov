#ifndef LDCOV_BGEN_BUFFER_POOL_H
#define LDCOV_BGEN_BUFFER_POOL_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace ldcov {
namespace bgen {

// Thread-local buffer pool for zero-allocation decompression
class BufferPool {
   private:
    // Thread-local storage for buffers
    static thread_local struct ThreadLocalBuffers {
        std::unique_ptr<uint8_t[]> decompression_buffer;
        size_t decompression_buffer_size = 0;

        std::unique_ptr<uint8_t[]> work_buffer;
        size_t work_buffer_size = 0;

        ~ThreadLocalBuffers() {
            // Automatic cleanup when thread exits
        }
    } buffers_;

    static constexpr size_t INITIAL_SIZE = 1024 * 1024;  // 1MB initial
    static constexpr size_t GROWTH_FACTOR = 2;           // Double when growing

   public:
    // Get decompression buffer, growing if needed
    static uint8_t* get_decompression_buffer(size_t required_size) {
        if (buffers_.decompression_buffer_size < required_size) {
            size_t new_size = std::max(required_size, INITIAL_SIZE);
            if (buffers_.decompression_buffer_size > 0) {
                new_size = std::max(new_size, buffers_.decompression_buffer_size * GROWTH_FACTOR);
            }
            buffers_.decompression_buffer.reset(new uint8_t[new_size]);
            buffers_.decompression_buffer_size = new_size;
        }
        return buffers_.decompression_buffer.get();
    }

    // Get work buffer for compressed data
    static uint8_t* get_work_buffer(size_t required_size) {
        if (buffers_.work_buffer_size < required_size) {
            size_t new_size = std::max(required_size, INITIAL_SIZE);
            if (buffers_.work_buffer_size > 0) {
                new_size = std::max(new_size, buffers_.work_buffer_size * GROWTH_FACTOR);
            }
            buffers_.work_buffer.reset(new uint8_t[new_size]);
            buffers_.work_buffer_size = new_size;
        }
        return buffers_.work_buffer.get();
    }

    // Get current buffer sizes for statistics
    static size_t decompression_buffer_size() {
        return buffers_.decompression_buffer_size;
    }
    static size_t work_buffer_size() {
        return buffers_.work_buffer_size;
    }
};

}  // namespace bgen
}  // namespace ldcov

#endif  // LDCOV_BGEN_BUFFER_POOL_H