#ifndef LDCOV_DECOMPRESS_DECOMPRESSOR_FACTORY_H
#define LDCOV_DECOMPRESS_DECOMPRESSOR_FACTORY_H

#include "decompressor.h"
#include "../io/reader_interface.h"
#include <memory>

namespace ldcov {
namespace bgen {
namespace decompress {

/**
 * Factory functions for creating different types of decompressors
 */

/**
 * Create a sequential decompressor optimized for sequential access patterns
 * 
 * @param file_reader FileReader instance to use
 * @param config Base decompressor configuration
 * @param enable_readahead Whether to enable read-ahead optimization
 * @return Unique pointer to SequentialDecompressor
 */
std::unique_ptr<VariantDecompressor> create_sequential_decompressor(
    ldcov::io::bgen::FileReader* file_reader,
    const VariantDecompressor::Config& config = VariantDecompressor::Config(),
    bool enable_readahead = true);

/**
 * Create a parallel decompressor optimized for random access patterns
 * 
 * @param file_reader FileReader instance to use  
 * @param num_threads Number of threads to use for parallel processing
 * @param config Base decompressor configuration
 * @return Unique pointer to ParallelDecompressor
 */
std::unique_ptr<VariantDecompressor> create_parallel_decompressor(
    ldcov::io::bgen::FileReader* file_reader,
    size_t num_threads,
    const VariantDecompressor::Config& config = VariantDecompressor::Config());

/**
 * Create an adaptive decompressor that switches between sequential and parallel
 * based on detected access patterns
 * 
 * @param file_reader FileReader instance to use
 * @param config Base decompressor configuration
 * @return Unique pointer to adaptive decompressor
 */
std::unique_ptr<VariantDecompressor> create_adaptive_decompressor(
    ldcov::io::bgen::FileReader* file_reader,
    const VariantDecompressor::Config& config = VariantDecompressor::Config());

/**
 * Create a simple decompressor for testing/reference
 * 
 * @param config Base decompressor configuration
 * @return Unique pointer to SimpleDecompressor
 */
std::unique_ptr<VariantDecompressor> create_simple_decompressor(
    const VariantDecompressor::Config& config = VariantDecompressor::Config());

} // namespace decompress
} // namespace bgen  
} // namespace ldcov

#endif // LDCOV_DECOMPRESS_DECOMPRESSOR_FACTORY_H