#include <stdexcept>

#include "buffer_manager.h"
#include "compression_utils.h"
#include "decompressor.h"
#include "parallel_decompressor.h"
#include "sequential_decompressor.h"

namespace ldcov {
namespace bgen {
namespace decompress {

// Forward declaration of SimpleDecompressor from simple_decompressor.cpp
class SimpleDecompressor;
std::unique_ptr<VariantDecompressor> create_simple_decompressor(
    const VariantDecompressor::Config& config);

// Forward declaration of create_parallel_decompressor
std::unique_ptr<VariantDecompressor> create_parallel_decompressor(
    ldcov::io::bgen::FileReader* file_reader, size_t num_threads,
    const VariantDecompressor::Config& config);

/**
 * Create a sequential decompressor
 *
 * This factory function creates a SequentialDecompressor with the provided
 * configuration. It requires a FileReader to be specified in the config.
 *
 * @param file_reader FileReader instance to use
 * @param config Base decompressor configuration
 * @param enable_readahead Whether to enable read-ahead optimization
 * @return Unique pointer to SequentialDecompressor
 */
std::unique_ptr<VariantDecompressor> create_sequential_decompressor(
    ldcov::io::bgen::FileReader* file_reader,
    const VariantDecompressor::Config& config = VariantDecompressor::Config(),
    bool enable_readahead = true) {
    if (!file_reader) {
        throw std::invalid_argument("create_sequential_decompressor: file_reader cannot be null");
    }

    // Initialize compression libraries
    static bool initialized = initialize_compression_libraries();
    if (!initialized) {
        throw std::runtime_error("Failed to initialize compression libraries");
    }

    // Create sequential configuration
    SequentialDecompressor::SequentialConfig seq_config;

    // Copy base configuration
    seq_config.buffer_manager = config.buffer_manager;
    seq_config.auto_detect_compression = config.auto_detect_compression;
    seq_config.validate_size = config.validate_size;
    seq_config.max_decompressed_size = config.max_decompressed_size;

    // Set sequential-specific configuration
    seq_config.file_reader = file_reader;
    seq_config.enable_readahead = enable_readahead;

    return std::unique_ptr<SequentialDecompressor>(new SequentialDecompressor(seq_config));
}

/**
 * Create an adaptive decompressor
 *
 * This creates a decompressor that automatically selects the best strategy
 * based on file size. Files > 10MB use parallel decompressor for better
 * performance, while smaller files use sequential decompressor.
 *
 * @param file_reader FileReader instance to use
 * @param config Base decompressor configuration
 * @return Unique pointer to adaptive decompressor
 */
std::unique_ptr<VariantDecompressor> create_adaptive_decompressor(
    ldcov::io::bgen::FileReader* file_reader,
    const VariantDecompressor::Config& config = VariantDecompressor::Config()) {
    // Check file size to determine optimal strategy
    // Note: The Python layer now handles this logic, but we keep this 
    // implementation for C++ API consistency
    if (file_reader && file_reader->size() > 10 * 1024 * 1024) {  // 10MB
        // Use parallel decompressor for larger files
        return create_parallel_decompressor(file_reader, 0, config);  // 0 = auto-detect threads
    } else {
        // Use sequential decompressor for smaller files
        return create_sequential_decompressor(file_reader, config, true);
    }
}

/**
 * Create a parallel decompressor optimized for random access
 *
 * This factory function creates a ParallelDecompressor with the specified
 * number of threads and configuration. It's optimized for random access
 * patterns where multiple variants may be requested in any order.
 *
 * @param file_reader FileReader instance to use
 * @param num_threads Number of worker threads (0 = auto-detect)
 * @param config Base decompressor configuration
 * @return Unique pointer to ParallelDecompressor
 */
std::unique_ptr<VariantDecompressor> create_parallel_decompressor(
    ldcov::io::bgen::FileReader* file_reader, size_t num_threads,
    const VariantDecompressor::Config& config) {
    if (!file_reader) {
        throw std::invalid_argument("create_parallel_decompressor: file_reader cannot be null");
    }

    // Initialize compression libraries
    static bool initialized = initialize_compression_libraries();
    if (!initialized) {
        throw std::runtime_error("Failed to initialize compression libraries");
    }

    // Create parallel configuration
    ParallelDecompressor::ParallelConfig parallel_config;

    // Copy base configuration
    parallel_config.buffer_manager = config.buffer_manager;
    parallel_config.auto_detect_compression = config.auto_detect_compression;
    parallel_config.validate_size = config.validate_size;
    parallel_config.max_decompressed_size = config.max_decompressed_size;

    // Set parallel-specific configuration
    parallel_config.num_threads = num_threads;

    return std::unique_ptr<ParallelDecompressor>(
        new ParallelDecompressor(file_reader, parallel_config));
}

}  // namespace decompress
}  // namespace bgen
}  // namespace ldcov