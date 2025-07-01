#include "simd_utils.h"

#include <algorithm>
#include <cstring>

namespace ldcov {
namespace io {
namespace bgen {

bool SimdUtils::has_simd_support() {
#if defined(LDCOV_HAS_AVX2) || defined(LDCOV_HAS_SSE2) || defined(LDCOV_HAS_NEON)
    return true;
#else
    return false;
#endif
}

void SimdUtils::convert_probabilities_to_dosages_8bit(const uint8_t* probs, float* dosages,
                                                      size_t n_samples, uint8_t ploidy) {
#ifdef LDCOV_HAS_AVX2
    convert_probabilities_avx2_8bit(probs, dosages, n_samples, ploidy);
#elif defined(LDCOV_HAS_SSE2)
    convert_probabilities_sse2_8bit(probs, dosages, n_samples, ploidy);
#elif defined(LDCOV_HAS_NEON)
    convert_probabilities_neon_8bit(probs, dosages, n_samples, ploidy);
#else
    convert_probabilities_scalar_8bit(probs, dosages, n_samples, ploidy);
#endif
}

void SimdUtils::convert_probabilities_to_dosages_16bit(const uint16_t* probs, float* dosages,
                                                       size_t n_samples, uint8_t ploidy) {
    // For now, use scalar implementation
    convert_probabilities_scalar_16bit(probs, dosages, n_samples, ploidy);
}

void SimdUtils::convert_probabilities_scalar_8bit(const uint8_t* probs, float* dosages,
                                                  size_t n_samples, uint8_t ploidy) {
    const float scale = 1.0f / 255.0f;

    for (size_t i = 0; i < n_samples; ++i) {
        // For diploid: dosage = p01 + 2*p11
        // probs layout: [p00, p01, p10/p11]
        float p01 = probs[i * 3 + 1] * scale;
        float p11 = probs[i * 3 + 2] * scale;
        dosages[i] = p01 + 2.0f * p11;
    }
}

void SimdUtils::convert_probabilities_scalar_16bit(const uint16_t* probs, float* dosages,
                                                   size_t n_samples, uint8_t ploidy) {
    const float scale = 1.0f / 65535.0f;

    for (size_t i = 0; i < n_samples; ++i) {
        float p01 = probs[i * 3 + 1] * scale;
        float p11 = probs[i * 3 + 2] * scale;
        dosages[i] = p01 + 2.0f * p11;
    }
}

#ifdef LDCOV_HAS_AVX2
void SimdUtils::convert_probabilities_avx2_8bit(const uint8_t* probs, float* dosages,
                                                size_t n_samples, uint8_t ploidy) {
    const __m256 scale = _mm256_set1_ps(1.0f / 255.0f);
    const __m256 two = _mm256_set1_ps(2.0f);

    size_t i = 0;

    // Process 8 samples at a time
    for (; i + 8 <= n_samples; i += 8) {
        // Load 24 bytes (8 samples * 3 probabilities)
        // We need to load p01 and p11 for each sample

        // Extract p01 values (every 3rd byte starting at offset 1)
        __m128i p01_bytes =
            _mm_setr_epi8(probs[i * 3 + 1], probs[(i + 1) * 3 + 1], probs[(i + 2) * 3 + 1],
                          probs[(i + 3) * 3 + 1], probs[(i + 4) * 3 + 1], probs[(i + 5) * 3 + 1],
                          probs[(i + 6) * 3 + 1], probs[(i + 7) * 3 + 1], 0, 0, 0, 0, 0, 0, 0, 0);

        // Extract p11 values (every 3rd byte starting at offset 2)
        __m128i p11_bytes =
            _mm_setr_epi8(probs[i * 3 + 2], probs[(i + 1) * 3 + 2], probs[(i + 2) * 3 + 2],
                          probs[(i + 3) * 3 + 2], probs[(i + 4) * 3 + 2], probs[(i + 5) * 3 + 2],
                          probs[(i + 6) * 3 + 2], probs[(i + 7) * 3 + 2], 0, 0, 0, 0, 0, 0, 0, 0);

        // Convert to 32-bit integers
        __m256i p01_32 = _mm256_cvtepu8_epi32(p01_bytes);
        __m256i p11_32 = _mm256_cvtepu8_epi32(p11_bytes);

        // Convert to float and scale
        __m256 p01_float = _mm256_mul_ps(_mm256_cvtepi32_ps(p01_32), scale);
        __m256 p11_float = _mm256_mul_ps(_mm256_cvtepi32_ps(p11_32), scale);

        // Calculate dosage = p01 + 2*p11
        __m256 dosage = _mm256_add_ps(p01_float, _mm256_mul_ps(two, p11_float));

        // Store result
        _mm256_storeu_ps(&dosages[i], dosage);
    }

    // Handle remaining samples
    for (; i < n_samples; ++i) {
        float p01 = probs[i * 3 + 1] * (1.0f / 255.0f);
        float p11 = probs[i * 3 + 2] * (1.0f / 255.0f);
        dosages[i] = p01 + 2.0f * p11;
    }
}
#endif

#ifdef LDCOV_HAS_SSE2
void SimdUtils::convert_probabilities_sse2_8bit(const uint8_t* probs, float* dosages,
                                                size_t n_samples, uint8_t ploidy) {
    const __m128 scale = _mm_set1_ps(1.0f / 255.0f);
    const __m128 two = _mm_set1_ps(2.0f);

    size_t i = 0;

    // Process 4 samples at a time
    for (; i + 4 <= n_samples; i += 4) {
        // Extract p01 and p11 values
        __m128i p01_bytes =
            _mm_setr_epi8(probs[i * 3 + 1], probs[(i + 1) * 3 + 1], probs[(i + 2) * 3 + 1],
                          probs[(i + 3) * 3 + 1], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

        __m128i p11_bytes =
            _mm_setr_epi8(probs[i * 3 + 2], probs[(i + 1) * 3 + 2], probs[(i + 2) * 3 + 2],
                          probs[(i + 3) * 3 + 2], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

        // Convert to 32-bit integers
        // Note: _mm_cvtepu8_epi32 requires SSE4.1, use manual conversion for SSE2
        __m128i zero = _mm_setzero_si128();
        __m128i p01_16 = _mm_unpacklo_epi8(p01_bytes, zero);
        __m128i p11_16 = _mm_unpacklo_epi8(p11_bytes, zero);
        __m128i p01_32 = _mm_unpacklo_epi16(p01_16, zero);
        __m128i p11_32 = _mm_unpacklo_epi16(p11_16, zero);

        // Convert to float and scale
        __m128 p01_float = _mm_mul_ps(_mm_cvtepi32_ps(p01_32), scale);
        __m128 p11_float = _mm_mul_ps(_mm_cvtepi32_ps(p11_32), scale);

        // Calculate dosage = p01 + 2*p11
        __m128 dosage = _mm_add_ps(p01_float, _mm_mul_ps(two, p11_float));

        // Store result
        _mm_storeu_ps(&dosages[i], dosage);
    }

    // Handle remaining samples
    for (; i < n_samples; ++i) {
        float p01 = probs[i * 3 + 1] * (1.0f / 255.0f);
        float p11 = probs[i * 3 + 2] * (1.0f / 255.0f);
        dosages[i] = p01 + 2.0f * p11;
    }
}
#endif

#ifdef LDCOV_HAS_NEON
void SimdUtils::convert_probabilities_neon_8bit(const uint8_t* probs, float* dosages,
                                                size_t n_samples, uint8_t ploidy) {
    const float32x4_t scale = vdupq_n_f32(1.0f / 255.0f);
    const float32x4_t two = vdupq_n_f32(2.0f);

    size_t i = 0;

    // Process 4 samples at a time
    for (; i + 4 <= n_samples; i += 4) {
        // Load p01 and p11 values
        uint8x8_t p01_p11 = vld1_u8(&probs[i * 3 + 1]);

        // Extract p01 (indices 0, 3, 6, 9)
        uint8x8_t p01_bytes = vext_u8(p01_p11, p01_p11, 0);
        // Extract p11 (indices 1, 4, 7, 10)
        uint8x8_t p11_bytes = vext_u8(p01_p11, p01_p11, 1);

        // Convert to 16-bit
        uint16x4_t p01_16 = vget_low_u16(vmovl_u8(p01_bytes));
        uint16x4_t p11_16 = vget_low_u16(vmovl_u8(p11_bytes));

        // Convert to 32-bit
        uint32x4_t p01_32 = vmovl_u16(p01_16);
        uint32x4_t p11_32 = vmovl_u16(p11_16);

        // Convert to float and scale
        float32x4_t p01_float = vmulq_f32(vcvtq_f32_u32(p01_32), scale);
        float32x4_t p11_float = vmulq_f32(vcvtq_f32_u32(p11_32), scale);

        // Calculate dosage = p01 + 2*p11
        float32x4_t dosage = vaddq_f32(p01_float, vmulq_f32(two, p11_float));

        // Store result
        vst1q_f32(&dosages[i], dosage);
    }

    // Handle remaining samples
    for (; i < n_samples; ++i) {
        float p01 = probs[i * 3 + 1] * (1.0f / 255.0f);
        float p11 = probs[i * 3 + 2] * (1.0f / 255.0f);
        dosages[i] = p01 + 2.0f * p11;
    }
}
#endif

void SimdUtils::fast_copy(uint8_t* dst, const uint8_t* src, size_t size) {
#ifdef LDCOV_HAS_AVX2
    // Use AVX2 for large copies
    if (size >= 256) {
        size_t i = 0;
        for (; i + 32 <= size; i += 32) {
            __m256i data = _mm256_loadu_si256((const __m256i*)(src + i));
            _mm256_storeu_si256((__m256i*)(dst + i), data);
        }

        // Copy remaining bytes
        if (i < size) {
            std::memcpy(dst + i, src + i, size - i);
        }
        return;
    }
#endif

    // Fallback to standard memcpy
    std::memcpy(dst, src, size);
}

}  // namespace bgen
}  // namespace io
}  // namespace ldcov