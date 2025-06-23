#ifndef LDCOV_BGEN_READER_IMPL_H
#define LDCOV_BGEN_READER_IMPL_H

#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include "io/reader_interface.h"
#include "index/bgi_reader.h"
#include "format/variant_parser.h"
#include "decompress/decompressor.h"

namespace ldcov {
namespace io {
namespace bgen {

// Import VariantMetadata from the format namespace
using ::ldcov::bgen::VariantMetadata;

// BGEN header structure
struct BgenHeader {
    uint32_t offset;
    uint32_t nvariants;
    uint32_t nsamples;
    uint32_t flags;
    uint8_t compression;
    uint8_t layout;
    bool has_sample_ids;
};

/**
 * BgenReaderImpl - Main implementation of BGEN file reader
 * 
 * This class handles reading BGEN files, managing decompression,
 * and providing efficient access to genetic variants.
 */
class BgenReaderImpl {
public:
    /**
     * Constructor
     * 
     * @param filename Path to BGEN file
     * @param bgi_filename Path to BGI index file
     * @throws std::runtime_error if files cannot be opened
     */
    BgenReaderImpl(const std::string& filename, const std::string& bgi_filename);
    
    /**
     * Destructor
     */
    ~BgenReaderImpl();
    
    /**
     * Get BGEN header information
     * 
     * @return Reference to header structure
     */
    const BgenHeader& header() const;
    
    /**
     * Get sample IDs from BGEN file
     * 
     * @return Vector of sample IDs
     */
    std::vector<std::string> get_sample_ids();
    
    /**
     * Set sample filter for efficient subset extraction
     * 
     * @param indices Sample indices to keep (0-based)
     */
    void set_sample_filter(const std::vector<uint32_t>& indices);
    
    /**
     * Read variant metadata at specific offset
     * 
     * @param offset File offset of variant
     * @return Variant metadata
     */
    VariantMetadata read_variant_metadata(uint64_t offset);
    
    /**
     * Read multiple variant metadata in batch (optimized)
     * 
     * @param offsets Vector of file offsets (will be sorted internally for optimal I/O)
     * @return Vector of variant metadata in the same order as input offsets
     */
    std::vector<VariantMetadata> read_variants_metadata_batch(const std::vector<uint64_t>& offsets);
    
    /**
     * Read and decompress variant genotype data
     * 
     * @param metadata Variant metadata
     * @return Unique pointer to decompressed genotype data
     */
    std::unique_ptr<::ldcov::bgen::decompress::DecompressedData> read_variant_genotypes(const VariantMetadata& metadata);
    
    /**
     * Read multiple variants in batch (optimized)
     * 
     * @param variants Vector of variant metadata
     * @return Vector of unique pointers to decompressed genotype data
     */
    std::vector<std::unique_ptr<::ldcov::bgen::decompress::DecompressedData>> read_variants_batch(
        const std::vector<VariantMetadata>& variants);
    
    /**
     * Set decompressor type
     * 
     * @param type Decompressor type: "adaptive", "sequential", "parallel"
     */
    void set_decompressor_type(const std::string& type);
    
    /**
     * Set number of threads for parallel decompressor
     * 
     * @param n Number of threads
     */
    void set_num_threads(size_t n);
    
    /**
     * Check if file is open
     * 
     * @return true if file is open
     */
    bool is_open() const;
    
    /**
     * Close the file
     */
    void close();
    
private:
    // Implementation details hidden with pimpl
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

} // namespace bgen
} // namespace io
} // namespace ldcov

#endif // LDCOV_BGEN_READER_IMPL_H