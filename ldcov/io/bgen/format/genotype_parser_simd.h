#ifndef LDCOV_BGEN_FORMAT_GENOTYPE_PARSER_SIMD_H
#define LDCOV_BGEN_FORMAT_GENOTYPE_PARSER_SIMD_H

#include <cstddef>
#include <cstdint>

namespace ldcov {
namespace bgen {

/**
 * SIMD-optimized genotype parsing functions.
 *
 * These functions provide vectorized implementations for computing dosages
 * from genotype probabilities, achieving 2-3x speedup over scalar code.
 *
 * Features:
 * - AVX2 support for x86-64 (processes 8-16 samples at once)
 * - NEON support for ARM (processes 4-8 samples at once)
 * - Runtime CPU feature detection
 * - Fallback to scalar code when SIMD not available
 */

// Check if this sample's dosage computation can use SIMD
bool can_use_simd_dosage();

// SIMD implementations for different bit depths
namespace simd {

/**
 * Compute dosages for 8-bit probabilities using SIMD.
 *
 * @param prob_data Pointer to probability data (2 bytes per sample)
 * @param output Output dosages array
 * @param n_samples Number of samples to process
 * @param missing_mask Bit mask for missing samples (can be nullptr)
 */
void compute_dosages_8bit_simd(const uint8_t* prob_data, float* output, size_t n_samples,
                               const uint8_t* missing_mask = nullptr);

/**
 * Compute dosages for 16-bit probabilities using SIMD.
 *
 * @param prob_data Pointer to probability data (4 bytes per sample)
 * @param output Output dosages array
 * @param n_samples Number of samples to process
 * @param missing_mask Bit mask for missing samples (can be nullptr)
 */
void compute_dosages_16bit_simd(const uint8_t* prob_data, float* output, size_t n_samples,
                                const uint8_t* missing_mask = nullptr);

/**
 * Compute dosages for 32-bit probabilities using SIMD.
 *
 * @param prob_data Pointer to probability data (8 bytes per sample)
 * @param output Output dosages array
 * @param n_samples Number of samples to process
 * @param missing_mask Bit mask for missing samples (can be nullptr)
 */
void compute_dosages_32bit_simd(const uint8_t* prob_data, float* output, size_t n_samples,
                                const uint8_t* missing_mask = nullptr);

/**
 * Compute filtered dosages using SIMD (processes only selected samples).
 *
 * @param prob_data Pointer to probability data
 * @param output Output dosages array
 * @param sample_indices Array of sample indices to process
 * @param n_indices Number of indices
 * @param bits_per_prob Bits per probability (8, 16, or 32)
 * @param missing_mask Bit mask for missing samples (can be nullptr)
 */
void compute_dosages_filtered_simd(const uint8_t* prob_data, float* output,
                                   const int* sample_indices, size_t n_indices,
                                   uint8_t bits_per_prob, const uint8_t* missing_mask = nullptr);

}  // namespace simd

}  // namespace bgen
}  // namespace ldcov

#endif  // LDCOV_BGEN_FORMAT_GENOTYPE_PARSER_SIMD_H