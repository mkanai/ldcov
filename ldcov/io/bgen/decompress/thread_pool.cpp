#include "thread_pool.h"
#include <algorithm>

namespace ldcov {
namespace io {
namespace bgen {

ThreadPool::ThreadPool(size_t num_threads) {
    // Ensure at least 1 thread, max of hardware concurrency
    num_threads = std::max(size_t(1), std::min(num_threads, size_t(std::thread::hardware_concurrency())));
    
    threads_.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
        threads_.emplace_back(&ThreadPool::worker_thread, this);
    }
}

ThreadPool::~ThreadPool() {
    shutdown();
}

void ThreadPool::shutdown() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }
    
    condition_.notify_all();
    
    for (auto& thread : threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    threads_.clear();
}

void ThreadPool::worker_thread() {
    while (true) {
        std::function<void()> task;
        
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            
            condition_.wait(lock, [this] {
                return stop_ || !tasks_.empty();
            });
            
            if (stop_ && tasks_.empty()) {
                return;
            }
            
            if (!tasks_.empty()) {
                task = std::move(tasks_.front());
                tasks_.pop();
                ++active_tasks_;
            }
        }
        
        if (task) {
            task();
            --active_tasks_;
        }
    }
}

} // namespace bgen
} // namespace io
} // namespace ldcov