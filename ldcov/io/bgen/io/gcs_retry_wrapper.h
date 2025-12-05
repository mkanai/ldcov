#pragma once

#include <chrono>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>

namespace ldcov {
namespace io {
namespace bgen {

/**
 * Retry wrapper for GCS operations with exponential backoff
 */
template <typename Result>
class RetryWrapper {
   public:
    static constexpr size_t DEFAULT_MAX_RETRIES = 3;
    static constexpr size_t DEFAULT_INITIAL_DELAY_MS = 100;
    static constexpr double DEFAULT_BACKOFF_MULTIPLIER = 2.0;

    /**
     * Execute a function with retry logic
     *
     * @param func Function to execute
     * @param operation_name Name of operation for error messages
     * @param max_retries Maximum number of retry attempts
     * @return Result of successful function execution
     */
    static Result execute_with_retry(std::function<Result()> func,
                                     const std::string& operation_name,
                                     size_t max_retries = DEFAULT_MAX_RETRIES) {
        size_t delay_ms = DEFAULT_INITIAL_DELAY_MS;
        std::string last_error;

        for (size_t attempt = 0; attempt <= max_retries; ++attempt) {
            try {
                return func();
            } catch (const std::exception& e) {
                last_error = e.what();

                // Check if error is retryable
                if (!is_retryable_error(last_error)) {
                    throw;  // Don't retry non-network errors
                }

                if (attempt < max_retries) {
                    // Log retry attempt
                    log_retry(operation_name, attempt + 1, delay_ms, last_error);

                    // Sleep before retry
                    std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));

                    // Exponential backoff
                    delay_ms = static_cast<size_t>(delay_ms * DEFAULT_BACKOFF_MULTIPLIER);
                }
            }
        }

        // All retries exhausted
        throw std::runtime_error("Operation '" + operation_name + "' failed after " +
                                 std::to_string(max_retries + 1) +
                                 " attempts. Last error: " + last_error);
    }

   private:
    /**
     * Check if an error is retryable (network-related)
     */
    static bool is_retryable_error(const std::string& error) {
        // Common retryable error patterns
        const char* retryable_patterns[] = {"timeout",
                                            "Timeout",
                                            "timed out",
                                            "Connection reset",
                                            "Connection refused",
                                            "Network is unreachable",
                                            "Name or service not known",
                                            "Temporary failure",
                                            "Service unavailable",
                                            "Too many requests",
                                            "Rate limit",
                                            "socket",
                                            "SSL",
                                            "TLS",
                                            "certificate verify failed",
                                            "DEADLINE_EXCEEDED",
                                            "UNAVAILABLE"};

        for (const auto& pattern : retryable_patterns) {
            if (error.find(pattern) != std::string::npos) {
                return true;
            }
        }

        return false;
    }

    /**
     * Log retry attempt (could be extended to use proper logging)
     */
    static void log_retry(const std::string& operation, size_t attempt, size_t delay_ms,
                          const std::string& error) {
        // In production, this would use a proper logging framework
        // For now, we'll just use stderr to avoid Python GIL issues
        fprintf(stderr,
                "[GCS Retry] Operation '%s' attempt %zu failed: %s. "
                "Retrying in %zu ms...\n",
                operation.c_str(), attempt, error.c_str(), delay_ms);
    }
};

// Specialization for void functions
template <>
class RetryWrapper<void> {
   public:
    static void execute_with_retry(std::function<void()> func, const std::string& operation_name,
                                   size_t max_retries = 3) {
        RetryWrapper<int>::execute_with_retry(
            [func]() -> int {
                func();
                return 0;
            },
            operation_name, max_retries);
    }
};

}  // namespace bgen
}  // namespace io
}  // namespace ldcov