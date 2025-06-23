#include "buffer_pool.h"

namespace ldcov {
namespace bgen {

// Thread-local storage definition
thread_local BufferPool::ThreadLocalBuffers BufferPool::buffers_;

} // namespace bgen
} // namespace ldcov