#ifndef LDCOV_IO_BGEN_DECOMPRESS_SIMD_UTILS_H
#define LDCOV_IO_BGEN_DECOMPRESS_SIMD_UTILS_H

#include <cstddef>
#include <cstdint>

// Check for SIMD support
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#define LDCOV_HAS_SSE2
#ifdef __AVX2__
#define LDCOV_HAS_AVX2
#endif
#elif defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define LDCOV_HAS_NEON
#endif

namespace ldcov {
namespace io {
namespace bgen {

/**
 * SIMD-optimized utilities for BGEN data processing
 */
class SimdUtils {
   public:
    /**
     * Convert 8-bit probabilities to float dosages using SIMD
     *
     * @param probs Input probabilities (3 per sample: p00, p01, p10)
     * @param dosages Output dosages (1 per sample)
     * @param n_samples Number of samples to process
     * @param ploidy Ploidy (2 for diploid)
     */
    static void convert_probabilities_to_dosages_8bit(const uint8_t* probs, float* dosages,
                                                      size_t n_samples, uint8_t ploidy = 2);

    /**
     * Convert 16-bit probabilities to float dosages using SIMD
     */
    static void convert_probabilities_to_dosages_16bit(const uint16_t* probs, float* dosages,
                                                       size_t n_samples, uint8_t ploidy = 2);

    /**
     * Fast copy with prefetching
     */
    static void fast_copy(uint8_t* dst, const uint8_t* src, size_t size);

    /**
     * Check if SIMD is available
     */
    static bool has_simd_support();

   private:
    // Scalar fallback implementations
    static void convert_probabilities_scalar_8bit(const uint8_t* probs, float* dosages,
                                                  size_t n_samples, uint8_t ploidy);

    static void convert_probabilities_scalar_16bit(const uint16_t* probs, float* dosages,
                                                   size_t n_samples, uint8_t ploidy);

#ifdef LDCOV_HAS_AVX2
    static void convert_probabilities_avx2_8bit(const uint8_t* probs, float* dosages,
                                                size_t n_samples, uint8_t ploidy);
#endif

#ifdef LDCOV_HAS_SSE2
    static void convert_probabilities_sse2_8bit(const uint8_t* probs, float* dosages,
                                                size_t n_samples, uint8_t ploidy);
#endif

#ifdef LDCOV_HAS_NEON
    static void convert_probabilities_neon_8bit(const uint8_t* probs, float* dosages,
                                                size_t n_samples, uint8_t ploidy);
#endif
};

}  // namespace bgen
}  // namespace io
}  // namespace ldcov

#endif  // LDCOV_IO_BGEN_DECOMPRESS_SIMD_UTILS_H