#ifndef LDCOV_BGEN_INDEX_BGI_READER_H
#define LDCOV_BGEN_INDEX_BGI_READER_H

#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace ldcov {
namespace io {
namespace bgen {
namespace index {

/**
 * VariantInfo - Information about a variant stored in the BGI index
 */
struct VariantInfo {
    uint64_t file_offset;      // Offset in BGEN file
    uint32_t variant_size;     // Size of variant data in BGEN file
    std::string chromosome;    // Chromosome name
    uint32_t position;         // 1-based position
    std::string rsid;          // RS ID
    std::string varid;         // Variant ID
    uint16_t n_alleles;        // Number of alleles
    std::string allele1;       // First allele (reference)
    std::string allele2;       // Second allele (alternate)
    
    // Constructor
    VariantInfo() : file_offset(0), variant_size(0), position(0), n_alleles(0) {}
};

/**
 * BGIReader - Reader for BGEN index files (.bgi)
 * 
 * The BGI format is a SQLite database with specific tables:
 * - Variant: contains variant metadata and file offsets
 * - Metadata: contains index metadata
 * 
 * This implementation provides thread-safe access to the index.
 */
class BGIReader {
public:
    /**
     * Constructor
     * 
     * @param bgi_path Path to BGI file
     * @throws std::runtime_error if file cannot be opened or is invalid
     */
    explicit BGIReader(const std::string& bgi_path);
    
    /**
     * Destructor
     */
    ~BGIReader();
    
    // Disable copy constructor and assignment
    BGIReader(const BGIReader&) = delete;
    BGIReader& operator=(const BGIReader&) = delete;
    
    // Enable move constructor and assignment
    BGIReader(BGIReader&&) noexcept;
    BGIReader& operator=(BGIReader&&) noexcept;
    
    /**
     * Query variants by genomic region
     * 
     * @param chromosome Chromosome name
     * @param start_pos Start position (1-based, inclusive)
     * @param end_pos End position (1-based, inclusive)
     * @return Vector of variant information
     */
    std::vector<VariantInfo> query_region(
        const std::string& chromosome,
        uint32_t start_pos,
        uint32_t end_pos
    );
    
    /**
     * Query variants by position
     * 
     * @param chromosome Chromosome name
     * @param position Position (1-based)
     * @return Vector of variant information at this position
     */
    std::vector<VariantInfo> query_position(
        const std::string& chromosome,
        uint32_t position
    );
    
    /**
     * Query variant by ID (rsid or varid)
     * 
     * @param variant_id Variant ID to search for
     * @return Vector of variant information (may contain multiple if ID is not unique)
     */
    std::vector<VariantInfo> query_variant_id(const std::string& variant_id);
    
    /**
     * Get variant by index
     * 
     * @param index 0-based variant index
     * @return Variant information
     * @throws std::out_of_range if index is invalid
     */
    VariantInfo get_variant(size_t index);
    
    /**
     * Get file offsets for a range of variant indices
     * 
     * @param start_idx Start index (0-based, inclusive)
     * @param end_idx End index (0-based, exclusive)
     * @return Vector of file offsets
     */
    std::vector<uint64_t> get_file_offsets(size_t start_idx, size_t end_idx);
    
    /**
     * Get all variant info efficiently
     * 
     * @return Vector of all variant information
     */
    std::vector<VariantInfo> get_all_variants();
    
    /**
     * Get total number of variants in the index
     * 
     * @return Number of variants
     */
    size_t get_variant_count() const;
    
    /**
     * Get list of all chromosomes in the index
     * 
     * @return Vector of chromosome names
     */
    std::vector<std::string> get_chromosomes() const;
    
    /**
     * Check if the index is valid and ready to use
     * 
     * @return true if index is valid
     */
    bool is_valid() const;
    
    /**
     * Get index metadata
     * 
     * @return Map of metadata key-value pairs
     */
    std::unordered_map<std::string, std::string> get_metadata() const;
    
    /**
     * Find variants matching chromosome, position, and allele combinations
     * 
     * This method performs exact matching on chromosome, position, allele1, and allele2.
     * It uses batch queries for efficiency when searching for many variants.
     * 
     * @param chromosome Chromosome to filter on
     * @param positions Positions to match
     * @param alleles1 First alleles (must match exactly)
     * @param alleles2 Second alleles (must match exactly)
     * @param batch_size Number of positions to query at once (default: 1000)
     * @return Vector of matched variants in order found
     * @throws std::invalid_argument if input vectors have different sizes
     */
    std::vector<VariantInfo> find_variants_by_filter(
        const std::string& chromosome,
        const std::vector<uint32_t>& positions,
        const std::vector<std::string>& alleles1,
        const std::vector<std::string>& alleles2,
        size_t batch_size = 1000
    );
    
    /**
     * Query multiple positions in a batch
     * 
     * @param chromosome Chromosome name
     * @param positions Vector of positions to query
     * @param batch_size Number of positions to query at once (default: 1000)
     * @return Vector of variant information
     */
    std::vector<VariantInfo> query_positions_batch(
        const std::string& chromosome,
        const std::vector<uint32_t>& positions,
        size_t batch_size = 1000
    );
    
    /**
     * Query multiple variant IDs in a batch
     * 
     * @param variant_ids Vector of variant IDs to search for
     * @param batch_size Number of IDs to query at once (default: 100)
     * @return Vector of variant information
     */
    std::vector<VariantInfo> query_variant_ids_batch(
        const std::vector<std::string>& variant_ids,
        size_t batch_size = 100
    );

private:
    // Forward declaration of implementation
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

} // namespace index
} // namespace bgen
} // namespace io
} // namespace ldcov

#endif // LDCOV_BGEN_INDEX_BGI_READER_H