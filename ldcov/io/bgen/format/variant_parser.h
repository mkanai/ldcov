#ifndef LDCOV_BGEN_FORMAT_VARIANT_PARSER_H
#define LDCOV_BGEN_FORMAT_VARIANT_PARSER_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "bgen_header.h"

namespace ldcov {
namespace bgen {

// Structure to hold variant metadata
struct VariantMetadata {
    uint64_t file_offset;              // Offset in file where variant starts
    std::string varid;                 // Variant ID
    std::string rsid;                  // RS ID
    std::string chrom;                 // Chromosome
    uint32_t pos;                      // Position
    uint16_t n_alleles;                // Number of alleles
    std::vector<std::string> alleles;  // Allele strings
    uint64_t genotype_offset;          // Offset to genotype data
    uint32_t genotype_length;          // Length of genotype data block

    VariantMetadata()
        : file_offset(0), pos(0), n_alleles(0), genotype_offset(0), genotype_length(0) {}
};

// Variant parser class
class VariantParser {
   public:
    /**
     * Parse variant metadata from buffer
     * @param buffer Pointer to variant data
     * @param size Size of buffer
     * @param layout BGEN layout version
     * @param compression Compression type (needed for v1.1 uncompressed size calculation)
     * @param expected_samples Expected number of samples (for v1.1 validation)
     * @return Parsed variant metadata and bytes consumed
     */
    static std::pair<VariantMetadata, size_t> parse(const uint8_t* buffer, size_t size,
                                                    LayoutType layout, CompressionType compression,
                                                    uint32_t expected_samples);

    /**
     * Get the size of variant block (excluding genotype data)
     * This is useful for reading just the metadata
     * @param buffer Pointer to variant data
     * @param size Size of buffer
     * @param layout BGEN layout version
     * @return Size of variant metadata block
     */
    static size_t getVariantMetadataSize(const uint8_t* buffer, size_t size, LayoutType layout);

    /**
     * Skip to next variant in buffer
     * @param buffer Pointer to current variant
     * @param size Size of buffer
     * @param layout BGEN layout version
     * @param compression Compression type
     * @param expected_samples Expected number of samples
     * @return Number of bytes to skip to reach next variant
     */
    static size_t skipVariant(const uint8_t* buffer, size_t size, LayoutType layout,
                              CompressionType compression, uint32_t expected_samples);

   private:
    // Parse v1.1 format variant
    static std::pair<VariantMetadata, size_t> parseV11(const uint8_t* buffer, size_t size,
                                                       CompressionType compression,
                                                       uint32_t expected_samples);

    // Parse v1.2 format variant
    static std::pair<VariantMetadata, size_t> parseV12(const uint8_t* buffer, size_t size);

    // Helper to read little-endian integers
    template <typename T>
    static T readLE(const uint8_t* ptr) {
        T value = 0;
        for (size_t i = 0; i < sizeof(T); ++i) {
            value |= static_cast<T>(ptr[i]) << (8 * i);
        }
        return value;
    }

    // Helper to read a length-prefixed string
    static std::string readLengthPrefixedString(const uint8_t* buffer, size_t& pos, size_t max_size,
                                                bool use_32bit_length = false);
};

// Batch variant parser for efficient reading of multiple variants
class BatchVariantParser {
   public:
    /**
     * Parse multiple variants from a buffer
     * @param buffer Pointer to data containing multiple variants
     * @param size Size of buffer
     * @param layout BGEN layout version
     * @param compression Compression type
     * @param expected_samples Expected number of samples
     * @param max_variants Maximum number of variants to parse (0 = all)
     * @return Vector of parsed variant metadata
     */
    static std::vector<VariantMetadata> parseBatch(const uint8_t* buffer, size_t size,
                                                   LayoutType layout, CompressionType compression,
                                                   uint32_t expected_samples,
                                                   size_t max_variants = 0);

    /**
     * Parse variants at specific offsets
     * @param buffer Pointer to entire file data
     * @param size Size of buffer
     * @param offsets File offsets where variants start
     * @param layout BGEN layout version
     * @param compression Compression type
     * @param expected_samples Expected number of samples
     * @return Vector of parsed variant metadata
     */
    static std::vector<VariantMetadata> parseAtOffsets(const uint8_t* buffer, size_t size,
                                                       const std::vector<uint64_t>& offsets,
                                                       LayoutType layout,
                                                       CompressionType compression,
                                                       uint32_t expected_samples);
};

}  // namespace bgen
}  // namespace ldcov

#endif  // LDCOV_BGEN_FORMAT_VARIANT_PARSER_H