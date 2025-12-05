#ifndef LDCOV_BGEN_FORMAT_GENOTYPE_PARSER_H
#define LDCOV_BGEN_FORMAT_GENOTYPE_PARSER_H

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "bgen_header.h"

namespace ldcov {
namespace bgen {

// Structure to hold genotype data
struct GenotypeData {
    uint32_t n_samples;
    uint16_t n_alleles;
    bool phased;
    std::vector<uint8_t> ploidy;       // Ploidy for each sample
    std::vector<float> probabilities;  // Genotype probabilities
    std::vector<bool> missing;         // Missing data flags
    uint8_t min_ploidy;
    uint8_t max_ploidy;
    bool constant_ploidy;

    GenotypeData()
        : n_samples(0),
          n_alleles(0),
          phased(false),
          min_ploidy(0),
          max_ploidy(0),
          constant_ploidy(true) {}

    // Calculate dosages from probabilities
    void computeDosages(float* output) const;

    // Calculate dosages for specific samples only
    void computeDosagesFiltered(const int* sample_indices, int n_indices, float* output) const;
};

// Genotype parser class
class GenotypeParser {
   public:
    /**
     * Parse genotype data from buffer
     * @param buffer Pointer to genotype data (may be compressed)
     * @param size Size of buffer
     * @param layout BGEN layout version
     * @param compression Compression type
     * @param n_samples Number of samples
     * @param n_alleles Number of alleles
     * @return Parsed genotype data
     */
    static std::unique_ptr<GenotypeData> parse(const uint8_t* buffer, size_t size,
                                               LayoutType layout, CompressionType compression,
                                               uint32_t n_samples, uint16_t n_alleles);

    /**
     * Parse genotype data from already decompressed buffer
     * @param buffer Pointer to decompressed genotype data
     * @param size Size of buffer
     * @param layout BGEN layout version
     * @param n_samples Number of samples
     * @param n_alleles Number of alleles
     * @return Parsed genotype data
     */
    static std::unique_ptr<GenotypeData> parseDecompressed(const uint8_t* buffer, size_t size,
                                                           LayoutType layout, uint32_t n_samples,
                                                           uint16_t n_alleles);

    /**
     * Compute dosages directly without full parsing (for efficiency)
     * @param buffer Pointer to genotype data
     * @param size Size of buffer
     * @param layout BGEN layout version
     * @param compression Compression type
     * @param n_samples Number of samples
     * @param n_alleles Number of alleles
     * @param output Pre-allocated array for dosages (size: n_samples)
     */
    static void computeDosagesDirect(const uint8_t* buffer, size_t size, LayoutType layout,
                                     CompressionType compression, uint32_t n_samples,
                                     uint16_t n_alleles, float* output);

    /**
     * Compute dosages for specific samples only
     * @param buffer Pointer to genotype data
     * @param size Size of buffer
     * @param layout BGEN layout version
     * @param compression Compression type
     * @param n_samples Number of samples in data
     * @param n_alleles Number of alleles
     * @param sample_indices Array of sample indices to extract
     * @param n_indices Number of indices
     * @param output Pre-allocated array for dosages (size: n_indices)
     */
    static void computeDosagesFiltered(const uint8_t* buffer, size_t size, LayoutType layout,
                                       CompressionType compression, uint32_t n_samples,
                                       uint16_t n_alleles, const int* sample_indices, int n_indices,
                                       float* output);

   private:
    // Parse v1.1 format genotypes
    static std::unique_ptr<GenotypeData> parseV11(const uint8_t* buffer, size_t size,
                                                  uint32_t n_samples);

    // Parse v1.2 format genotypes
    static std::unique_ptr<GenotypeData> parseV12(const uint8_t* buffer, size_t size,
                                                  uint32_t n_samples, uint16_t n_alleles);

    // Direct dosage computation for v1.1
    static void computeDosagesV11Direct(const uint8_t* buffer, size_t size, uint32_t n_samples,
                                        float* output);

    // Direct dosage computation for v1.2
    static void computeDosagesV12Direct(const uint8_t* buffer, size_t size, uint32_t n_samples,
                                        uint16_t n_alleles, float* output);

    // Optimized filtered dosage computation for v1.2
    static void computeDosagesV12Filtered(const uint8_t* buffer, size_t size, uint32_t n_samples,
                                          uint16_t n_alleles, const int* sample_indices,
                                          int n_indices, float* output);

    // Helper to read little-endian integers
    template <typename T>
    static T readLE(const uint8_t* ptr) {
        T value = 0;
        for (size_t i = 0; i < sizeof(T); ++i) {
            value |= static_cast<T>(ptr[i]) << (8 * i);
        }
        return value;
    }
};

// Batch genotype parser for efficient processing
class BatchGenotypeParser {
   public:
    /**
     * Parse multiple genotype blocks
     * @param buffers Vector of genotype data buffers
     * @param sizes Sizes of each buffer
     * @param layout BGEN layout version
     * @param compression Compression type
     * @param n_samples Number of samples
     * @param n_alleles_list Number of alleles for each variant
     * @return Vector of parsed genotype data
     */
    static std::vector<std::unique_ptr<GenotypeData>> parseBatch(
        const std::vector<const uint8_t*>& buffers, const std::vector<size_t>& sizes,
        LayoutType layout, CompressionType compression, uint32_t n_samples,
        const std::vector<uint16_t>& n_alleles_list);

    /**
     * Compute dosages for multiple variants directly
     * @param buffers Vector of genotype data buffers
     * @param sizes Sizes of each buffer
     * @param layout BGEN layout version
     * @param compression Compression type
     * @param n_samples Number of samples
     * @param n_alleles_list Number of alleles for each variant
     * @param output Pre-allocated 2D array for dosages (n_samples x n_variants)
     * @param output_stride Stride between variants in output array
     */
    static void computeDosagesBatch(const std::vector<const uint8_t*>& buffers,
                                    const std::vector<size_t>& sizes, LayoutType layout,
                                    CompressionType compression, uint32_t n_samples,
                                    const std::vector<uint16_t>& n_alleles_list, float* output,
                                    size_t output_stride);
};

}  // namespace bgen
}  // namespace ldcov

#endif  // LDCOV_BGEN_FORMAT_GENOTYPE_PARSER_H