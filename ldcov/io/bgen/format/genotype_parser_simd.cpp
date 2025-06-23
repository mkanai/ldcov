#include "genotype_parser_simd.h"
#include <cmath>
#include <cstring>
#include <algorithm>

#ifdef __x86_64__
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace ldcov {
namespace bgen {

// CPU feature detection
static bool g_simd_initialized = false;
static bool g_has_avx2 = false;
static bool g_has_neon = false;

static void detect_cpu_features() {
    if (g_simd_initialized) return;
    
#ifdef __x86_64__
    // Check for AVX2 support
    __builtin_cpu_init();
    g_has_avx2 = __builtin_cpu_supports("avx2");
#elif defined(__aarch64__)
    // ARM NEON is always available on AArch64
    g_has_neon = true;
#endif
    
    g_simd_initialized = true;
}

bool can_use_simd_dosage() {
    detect_cpu_features();
    return g_has_avx2 || g_has_neon;
}

namespace simd {

// Helper to read little-endian 16-bit value
static inline uint16_t read_le16(const uint8_t* ptr) {
    return ptr[0] | (static_cast<uint16_t>(ptr[1]) << 8);
}

// Helper to read little-endian 32-bit value
static inline uint32_t read_le32(const uint8_t* ptr) {
    return ptr[0] | (static_cast<uint32_t>(ptr[1]) << 8) |
           (static_cast<uint32_t>(ptr[2]) << 16) | (static_cast<uint32_t>(ptr[3]) << 24);
}

// 8-bit dosage computation
void compute_dosages_8bit_simd(
    const uint8_t* prob_data,
    float* output,
    size_t n_samples,
    const uint8_t* missing_mask
) {
    size_t i = 0;
    
#ifdef __x86_64__
    if (g_has_avx2) {
        // Process 16 samples at a time with AVX2
        const __m256 scale = _mm256_set1_ps(1.0f / 255.0f);
        const __m256 two = _mm256_set1_ps(2.0f);
        
        for (; i + 15 < n_samples; i += 16) {
            // Load 32 bytes (16 samples × 2 bytes)
            __m256i data = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(prob_data + i * 2));
            
            // Extract P(AA) and P(AB) for each sample
            __m256i mask_even = _mm256_set_epi8(
                30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0,
                30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0
            );
            __m256i mask_odd = _mm256_set_epi8(
                31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1,
                31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1
            );
            
            __m256i prob_aa_packed = _mm256_shuffle_epi8(data, mask_even);
            __m256i prob_ab_packed = _mm256_shuffle_epi8(data, mask_odd);
            
            // Convert to 16-bit for processing
            __m256i prob_aa_lo = _mm256_unpacklo_epi8(prob_aa_packed, _mm256_setzero_si256());
            __m256i prob_aa_hi = _mm256_unpackhi_epi8(prob_aa_packed, _mm256_setzero_si256());
            __m256i prob_ab_lo = _mm256_unpacklo_epi8(prob_ab_packed, _mm256_setzero_si256());
            __m256i prob_ab_hi = _mm256_unpackhi_epi8(prob_ab_packed, _mm256_setzero_si256());
            
            // Calculate P(BB) = 255 - P(AA) - P(AB)
            __m256i sum_lo = _mm256_add_epi16(prob_aa_lo, prob_ab_lo);
            __m256i sum_hi = _mm256_add_epi16(prob_aa_hi, prob_ab_hi);
            __m256i max_val = _mm256_set1_epi16(255);
            __m256i prob_bb_lo = _mm256_sub_epi16(max_val, sum_lo);
            __m256i prob_bb_hi = _mm256_sub_epi16(max_val, sum_hi);
            
            // Convert to float and calculate dosage = (P(AB) + 2*P(BB)) / 255
            // Process lower 8 samples
            __m256 prob_ab_f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(prob_ab_lo)));
            __m256 prob_bb_f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(prob_bb_lo)));
            __m256 dosage_lo = _mm256_mul_ps(_mm256_fmadd_ps(two, prob_bb_f, prob_ab_f), scale);
            _mm256_storeu_ps(output + i, dosage_lo);
            
            // Process upper 8 samples
            prob_ab_f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(prob_ab_lo, 1)));
            prob_bb_f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(prob_bb_lo, 1)));
            __m256 dosage_hi = _mm256_mul_ps(_mm256_fmadd_ps(two, prob_bb_f, prob_ab_f), scale);
            _mm256_storeu_ps(output + i + 8, dosage_hi);
        }
    }
#elif defined(__aarch64__)
    if (g_has_neon) {
        // Process 8 samples at a time with NEON
        const float32x4_t scale = vdupq_n_f32(1.0f / 255.0f);
        const float32x4_t two = vdupq_n_f32(2.0f);
        
        for (; i + 7 < n_samples; i += 8) {
            // Load 16 bytes (8 samples × 2 bytes)
            uint8x16_t data = vld1q_u8(prob_data + i * 2);
            
            // Extract P(AA) and P(AB) - even and odd bytes
            uint8x8_t prob_aa_8 = vuzp1_u8(vget_low_u8(data), vget_high_u8(data));
            uint8x8_t prob_ab_8 = vuzp2_u8(vget_low_u8(data), vget_high_u8(data));
            
            // Convert to 16-bit
            uint16x8_t prob_aa_16 = vmovl_u8(prob_aa_8);
            uint16x8_t prob_ab_16 = vmovl_u8(prob_ab_8);
            
            // Calculate P(BB) = 255 - P(AA) - P(AB)
            uint16x8_t sum = vaddq_u16(prob_aa_16, prob_ab_16);
            uint16x8_t prob_bb_16 = vsubq_u16(vdupq_n_u16(255), sum);
            
            // Convert to float and calculate dosage
            // Process lower 4 samples
            uint32x4_t prob_ab_32_lo = vmovl_u16(vget_low_u16(prob_ab_16));
            uint32x4_t prob_bb_32_lo = vmovl_u16(vget_low_u16(prob_bb_16));
            float32x4_t prob_ab_f = vcvtq_f32_u32(prob_ab_32_lo);
            float32x4_t prob_bb_f = vcvtq_f32_u32(prob_bb_32_lo);
            float32x4_t dosage_lo = vmulq_f32(vmlaq_f32(prob_ab_f, two, prob_bb_f), scale);
            vst1q_f32(output + i, dosage_lo);
            
            // Process upper 4 samples
            uint32x4_t prob_ab_32_hi = vmovl_u16(vget_high_u16(prob_ab_16));
            uint32x4_t prob_bb_32_hi = vmovl_u16(vget_high_u16(prob_bb_16));
            prob_ab_f = vcvtq_f32_u32(prob_ab_32_hi);
            prob_bb_f = vcvtq_f32_u32(prob_bb_32_hi);
            float32x4_t dosage_hi = vmulq_f32(vmlaq_f32(prob_ab_f, two, prob_bb_f), scale);
            vst1q_f32(output + i + 4, dosage_hi);
        }
    }
#endif
    
    // Scalar fallback for remaining samples
    for (; i < n_samples; ++i) {
        uint8_t prob_aa = prob_data[i * 2];
        uint8_t prob_ab = prob_data[i * 2 + 1];
        
        if (prob_aa + prob_ab > 255) {
            output[i] = std::nanf("");
        } else {
            uint8_t prob_bb = 255 - prob_aa - prob_ab;
            output[i] = (prob_ab + 2.0f * prob_bb) / 255.0f;
        }
    }
    
    // Apply missing mask if provided
    if (missing_mask) {
        for (size_t j = 0; j < n_samples; ++j) {
            int byte_idx = j / 8;
            int bit_idx = j % 8;
            if (missing_mask[byte_idx] & (1 << bit_idx)) {
                output[j] = std::nanf("");
            }
        }
    }
}

// 16-bit dosage computation
void compute_dosages_16bit_simd(
    const uint8_t* prob_data,
    float* output,
    size_t n_samples,
    const uint8_t* missing_mask
) {
    size_t i = 0;
    
#ifdef __x86_64__
    if (g_has_avx2) {
        // Process 8 samples at a time with AVX2
        const __m256 scale = _mm256_set1_ps(1.0f / 65535.0f);
        const __m256 two = _mm256_set1_ps(2.0f);
        const __m256i max_val = _mm256_set1_epi32(65535);
        
        for (; i + 7 < n_samples; i += 8) {
            // Load 32 bytes (8 samples × 4 bytes)
            __m256i data_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(prob_data + i * 4));
            __m256i data_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(prob_data + i * 4 + 16));
            
            // Extract P(AA) and P(AB) as 32-bit values
            __m256i indices_aa = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);
            __m256i indices_ab = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);
            
            // Gather P(AA) values (every other 16-bit value starting at offset 0)
            __m256i prob_aa = _mm256_set_epi32(
                read_le16(prob_data + (i + 7) * 4),
                read_le16(prob_data + (i + 6) * 4),
                read_le16(prob_data + (i + 5) * 4),
                read_le16(prob_data + (i + 4) * 4),
                read_le16(prob_data + (i + 3) * 4),
                read_le16(prob_data + (i + 2) * 4),
                read_le16(prob_data + (i + 1) * 4),
                read_le16(prob_data + (i + 0) * 4)
            );
            
            // Gather P(AB) values (every other 16-bit value starting at offset 2)
            __m256i prob_ab = _mm256_set_epi32(
                read_le16(prob_data + (i + 7) * 4 + 2),
                read_le16(prob_data + (i + 6) * 4 + 2),
                read_le16(prob_data + (i + 5) * 4 + 2),
                read_le16(prob_data + (i + 4) * 4 + 2),
                read_le16(prob_data + (i + 3) * 4 + 2),
                read_le16(prob_data + (i + 2) * 4 + 2),
                read_le16(prob_data + (i + 1) * 4 + 2),
                read_le16(prob_data + (i + 0) * 4 + 2)
            );
            
            // Calculate P(BB) = 65535 - P(AA) - P(AB)
            __m256i sum = _mm256_add_epi32(prob_aa, prob_ab);
            __m256i prob_bb = _mm256_sub_epi32(max_val, sum);
            
            // Convert to float and calculate dosage
            __m256 prob_ab_f = _mm256_cvtepi32_ps(prob_ab);
            __m256 prob_bb_f = _mm256_cvtepi32_ps(prob_bb);
            __m256 dosage = _mm256_mul_ps(_mm256_fmadd_ps(two, prob_bb_f, prob_ab_f), scale);
            
            _mm256_storeu_ps(output + i, dosage);
        }
    }
#elif defined(__aarch64__)
    if (g_has_neon) {
        // Process 4 samples at a time with NEON
        const float32x4_t scale = vdupq_n_f32(1.0f / 65535.0f);
        const float32x4_t two = vdupq_n_f32(2.0f);
        
        for (; i + 3 < n_samples; i += 4) {
            // Load P(AA) and P(AB) for 4 samples
            uint16x4_t prob_aa = {
                read_le16(prob_data + (i + 0) * 4),
                read_le16(prob_data + (i + 1) * 4),
                read_le16(prob_data + (i + 2) * 4),
                read_le16(prob_data + (i + 3) * 4)
            };
            
            uint16x4_t prob_ab = {
                read_le16(prob_data + (i + 0) * 4 + 2),
                read_le16(prob_data + (i + 1) * 4 + 2),
                read_le16(prob_data + (i + 2) * 4 + 2),
                read_le16(prob_data + (i + 3) * 4 + 2)
            };
            
            // Calculate P(BB) = 65535 - P(AA) - P(AB)
            uint32x4_t prob_aa_32 = vmovl_u16(prob_aa);
            uint32x4_t prob_ab_32 = vmovl_u16(prob_ab);
            uint32x4_t sum = vaddq_u32(prob_aa_32, prob_ab_32);
            uint32x4_t prob_bb_32 = vsubq_u32(vdupq_n_u32(65535), sum);
            
            // Convert to float and calculate dosage
            float32x4_t prob_ab_f = vcvtq_f32_u32(prob_ab_32);
            float32x4_t prob_bb_f = vcvtq_f32_u32(prob_bb_32);
            float32x4_t dosage = vmulq_f32(vmlaq_f32(prob_ab_f, two, prob_bb_f), scale);
            
            vst1q_f32(output + i, dosage);
        }
    }
#endif
    
    // Scalar fallback for remaining samples
    for (; i < n_samples; ++i) {
        uint16_t prob_aa = read_le16(prob_data + i * 4);
        uint16_t prob_ab = read_le16(prob_data + i * 4 + 2);
        
        if (prob_aa + prob_ab > 65535) {
            output[i] = std::nanf("");
        } else {
            uint16_t prob_bb = 65535 - prob_aa - prob_ab;
            output[i] = (prob_ab + 2.0f * prob_bb) / 65535.0f;
        }
    }
    
    // Apply missing mask if provided
    if (missing_mask) {
        for (size_t j = 0; j < n_samples; ++j) {
            int byte_idx = j / 8;
            int bit_idx = j % 8;
            if (missing_mask[byte_idx] & (1 << bit_idx)) {
                output[j] = std::nanf("");
            }
        }
    }
}

// 32-bit dosage computation
void compute_dosages_32bit_simd(
    const uint8_t* prob_data,
    float* output,
    size_t n_samples,
    const uint8_t* missing_mask
) {
    size_t i = 0;
    
#ifdef __x86_64__
    if (g_has_avx2) {
        // Process 4 samples at a time with AVX2
        const __m256d scale = _mm256_set1_pd(1.0 / 4294967295.0);
        const __m256d two = _mm256_set1_pd(2.0);
        
        for (; i + 3 < n_samples; i += 4) {
            // Load P(AA) and P(AB) for 4 samples
            uint64_t prob_aa[4], prob_ab[4];
            for (int j = 0; j < 4; ++j) {
                prob_aa[j] = read_le32(prob_data + (i + j) * 8);
                prob_ab[j] = read_le32(prob_data + (i + j) * 8 + 4);
            }
            
            // Calculate P(BB) = 4294967295 - P(AA) - P(AB)
            uint64_t prob_bb[4];
            for (int j = 0; j < 4; ++j) {
                if (prob_aa[j] + prob_ab[j] > 4294967295UL) {
                    output[i + j] = std::nanf("");
                    prob_bb[j] = 0;  // Dummy value
                } else {
                    prob_bb[j] = 4294967295UL - prob_aa[j] - prob_ab[j];
                }
            }
            
            // Convert to double for precision
            __m256d prob_ab_d = _mm256_set_pd(
                static_cast<double>(prob_ab[3]),
                static_cast<double>(prob_ab[2]),
                static_cast<double>(prob_ab[1]),
                static_cast<double>(prob_ab[0])
            );
            
            __m256d prob_bb_d = _mm256_set_pd(
                static_cast<double>(prob_bb[3]),
                static_cast<double>(prob_bb[2]),
                static_cast<double>(prob_bb[1]),
                static_cast<double>(prob_bb[0])
            );
            
            // Calculate dosage
            __m256d dosage_d = _mm256_mul_pd(_mm256_fmadd_pd(two, prob_bb_d, prob_ab_d), scale);
            
            // Convert to float and store
            __m128 dosage_f = _mm256_cvtpd_ps(dosage_d);
            _mm_storeu_ps(output + i, dosage_f);
        }
    }
#endif
    
    // Scalar fallback (also used for ARM)
    for (; i < n_samples; ++i) {
        uint32_t prob_aa = read_le32(prob_data + i * 8);
        uint32_t prob_ab = read_le32(prob_data + i * 8 + 4);
        
        if (static_cast<uint64_t>(prob_aa) + prob_ab > 4294967295UL) {
            output[i] = std::nanf("");
        } else {
            uint32_t prob_bb = 4294967295UL - prob_aa - prob_ab;
            double dosage = (static_cast<double>(prob_ab) + 2.0 * prob_bb) / 4294967295.0;
            output[i] = static_cast<float>(dosage);
        }
    }
    
    // Apply missing mask if provided
    if (missing_mask) {
        for (size_t j = 0; j < n_samples; ++j) {
            int byte_idx = j / 8;
            int bit_idx = j % 8;
            if (missing_mask[byte_idx] & (1 << bit_idx)) {
                output[j] = std::nanf("");
            }
        }
    }
}

// Forward declarations of helper functions
static void compute_dosages_filtered_8bit_simd(
    const uint8_t* prob_data,
    float* output,
    const int* sample_indices,
    size_t n_indices,
    const uint8_t* missing_mask
);

static void compute_dosages_filtered_16bit_simd(
    const uint8_t* prob_data,
    float* output,
    const int* sample_indices,
    size_t n_indices,
    const uint8_t* missing_mask
);

static void compute_dosages_filtered_32bit_simd(
    const uint8_t* prob_data,
    float* output,
    const int* sample_indices,
    size_t n_indices,
    const uint8_t* missing_mask
);

// Optimized filtered dosage computation
void compute_dosages_filtered_simd(
    const uint8_t* prob_data,
    float* output,
    const int* sample_indices,
    size_t n_indices,
    uint8_t bits_per_prob,
    const uint8_t* missing_mask
) {
    if (bits_per_prob == 8) {
        compute_dosages_filtered_8bit_simd(prob_data, output, sample_indices, n_indices, missing_mask);
    } else if (bits_per_prob == 16) {
        compute_dosages_filtered_16bit_simd(prob_data, output, sample_indices, n_indices, missing_mask);
    } else if (bits_per_prob == 32) {
        compute_dosages_filtered_32bit_simd(prob_data, output, sample_indices, n_indices, missing_mask);
    }
}

// Helper functions for filtered computation
static void compute_dosages_filtered_8bit_simd(
    const uint8_t* prob_data,
    float* output,
    const int* sample_indices,
    size_t n_indices,
    const uint8_t* missing_mask
) {
    size_t i = 0;
    
#ifdef __x86_64__
    if (g_has_avx2) {
        // Process 8 samples at a time with AVX2
        const __m256 scale = _mm256_set1_ps(1.0f / 255.0f);
        const __m256 two = _mm256_set1_ps(2.0f);
        
        for (; i + 7 < n_indices; i += 8) {
            // Gather data for 8 selected samples
            uint8_t prob_aa[8], prob_ab[8];
            bool is_missing[8] = {false};
            
            for (int j = 0; j < 8; ++j) {
                int idx = sample_indices[i + j];
                prob_aa[j] = prob_data[idx * 2];
                prob_ab[j] = prob_data[idx * 2 + 1];
                
                if (missing_mask) {
                    int byte_idx = idx / 8;
                    int bit_idx = idx % 8;
                    is_missing[j] = (missing_mask[byte_idx] & (1 << bit_idx)) != 0;
                }
            }
            
            // Pack into vectors
            __m256i prob_aa_vec = _mm256_set_epi32(
                prob_aa[7], prob_aa[6], prob_aa[5], prob_aa[4],
                prob_aa[3], prob_aa[2], prob_aa[1], prob_aa[0]
            );
            
            __m256i prob_ab_vec = _mm256_set_epi32(
                prob_ab[7], prob_ab[6], prob_ab[5], prob_ab[4],
                prob_ab[3], prob_ab[2], prob_ab[1], prob_ab[0]
            );
            
            // Calculate P(BB) = 255 - P(AA) - P(AB)
            __m256i sum = _mm256_add_epi32(prob_aa_vec, prob_ab_vec);
            __m256i max_val = _mm256_set1_epi32(255);
            __m256i prob_bb = _mm256_sub_epi32(max_val, sum);
            
            // Convert to float and calculate dosage
            __m256 prob_ab_f = _mm256_cvtepi32_ps(prob_ab_vec);
            __m256 prob_bb_f = _mm256_cvtepi32_ps(prob_bb);
            __m256 dosage = _mm256_mul_ps(_mm256_fmadd_ps(two, prob_bb_f, prob_ab_f), scale);
            
            // Handle missing values
            __m256 missing_mask_vec = _mm256_set_ps(
                is_missing[7] ? std::nanf("") : 0.0f,
                is_missing[6] ? std::nanf("") : 0.0f,
                is_missing[5] ? std::nanf("") : 0.0f,
                is_missing[4] ? std::nanf("") : 0.0f,
                is_missing[3] ? std::nanf("") : 0.0f,
                is_missing[2] ? std::nanf("") : 0.0f,
                is_missing[1] ? std::nanf("") : 0.0f,
                is_missing[0] ? std::nanf("") : 0.0f
            );
            
            dosage = _mm256_blendv_ps(dosage, missing_mask_vec, missing_mask_vec);
            
            _mm256_storeu_ps(output + i, dosage);
        }
    }
#elif defined(__aarch64__)
    if (g_has_neon) {
        // Process 4 samples at a time with NEON
        const float32x4_t scale = vdupq_n_f32(1.0f / 255.0f);
        const float32x4_t two = vdupq_n_f32(2.0f);
        
        for (; i + 3 < n_indices; i += 4) {
            // Gather data for 4 selected samples
            uint8_t prob_aa[4], prob_ab[4];
            bool is_missing[4] = {false};
            
            for (int j = 0; j < 4; ++j) {
                int idx = sample_indices[i + j];
                prob_aa[j] = prob_data[idx * 2];
                prob_ab[j] = prob_data[idx * 2 + 1];
                
                if (missing_mask) {
                    int byte_idx = idx / 8;
                    int bit_idx = idx % 8;
                    is_missing[j] = (missing_mask[byte_idx] & (1 << bit_idx)) != 0;
                }
            }
            
            // Convert to vectors
            uint32x4_t prob_aa_vec = {prob_aa[0], prob_aa[1], prob_aa[2], prob_aa[3]};
            uint32x4_t prob_ab_vec = {prob_ab[0], prob_ab[1], prob_ab[2], prob_ab[3]};
            
            // Calculate P(BB) = 255 - P(AA) - P(AB)
            uint32x4_t sum = vaddq_u32(prob_aa_vec, prob_ab_vec);
            uint32x4_t prob_bb = vsubq_u32(vdupq_n_u32(255), sum);
            
            // Convert to float and calculate dosage
            float32x4_t prob_ab_f = vcvtq_f32_u32(prob_ab_vec);
            float32x4_t prob_bb_f = vcvtq_f32_u32(prob_bb);
            float32x4_t dosage = vmulq_f32(vmlaq_f32(prob_ab_f, two, prob_bb_f), scale);
            
            // Store results with missing handling
            float result[4];
            vst1q_f32(result, dosage);
            
            for (int j = 0; j < 4; ++j) {
                output[i + j] = is_missing[j] ? std::nanf("") : result[j];
            }
        }
    }
#endif
    
    // Scalar fallback for remaining samples
    for (; i < n_indices; ++i) {
        int idx = sample_indices[i];
        
        // Check if missing
        bool is_missing = false;
        if (missing_mask) {
            int byte_idx = idx / 8;
            int bit_idx = idx % 8;
            is_missing = (missing_mask[byte_idx] & (1 << bit_idx)) != 0;
        }
        
        if (is_missing) {
            output[i] = std::nanf("");
            continue;
        }
        
        uint8_t prob_aa = prob_data[idx * 2];
        uint8_t prob_ab = prob_data[idx * 2 + 1];
        
        if (prob_aa + prob_ab > 255) {
            output[i] = std::nanf("");
        } else {
            uint8_t prob_bb = 255 - prob_aa - prob_ab;
            output[i] = (prob_ab + 2.0f * prob_bb) / 255.0f;
        }
    }
}

static void compute_dosages_filtered_16bit_simd(
    const uint8_t* prob_data,
    float* output,
    const int* sample_indices,
    size_t n_indices,
    const uint8_t* missing_mask
) {
    size_t i = 0;
    
#ifdef __x86_64__
    if (g_has_avx2) {
        // Process 8 samples at a time with AVX2
        const __m256 scale = _mm256_set1_ps(1.0f / 65535.0f);
        const __m256 two = _mm256_set1_ps(2.0f);
        
        for (; i + 7 < n_indices; i += 8) {
            // Gather data for 8 selected samples
            __m256i prob_aa = _mm256_set_epi32(
                read_le16(prob_data + sample_indices[i + 7] * 4),
                read_le16(prob_data + sample_indices[i + 6] * 4),
                read_le16(prob_data + sample_indices[i + 5] * 4),
                read_le16(prob_data + sample_indices[i + 4] * 4),
                read_le16(prob_data + sample_indices[i + 3] * 4),
                read_le16(prob_data + sample_indices[i + 2] * 4),
                read_le16(prob_data + sample_indices[i + 1] * 4),
                read_le16(prob_data + sample_indices[i + 0] * 4)
            );
            
            __m256i prob_ab = _mm256_set_epi32(
                read_le16(prob_data + sample_indices[i + 7] * 4 + 2),
                read_le16(prob_data + sample_indices[i + 6] * 4 + 2),
                read_le16(prob_data + sample_indices[i + 5] * 4 + 2),
                read_le16(prob_data + sample_indices[i + 4] * 4 + 2),
                read_le16(prob_data + sample_indices[i + 3] * 4 + 2),
                read_le16(prob_data + sample_indices[i + 2] * 4 + 2),
                read_le16(prob_data + sample_indices[i + 1] * 4 + 2),
                read_le16(prob_data + sample_indices[i + 0] * 4 + 2)
            );
            
            // Calculate P(BB) = 65535 - P(AA) - P(AB)
            __m256i sum = _mm256_add_epi32(prob_aa, prob_ab);
            __m256i max_val = _mm256_set1_epi32(65535);
            __m256i prob_bb = _mm256_sub_epi32(max_val, sum);
            
            // Convert to float and calculate dosage
            __m256 prob_ab_f = _mm256_cvtepi32_ps(prob_ab);
            __m256 prob_bb_f = _mm256_cvtepi32_ps(prob_bb);
            __m256 dosage = _mm256_mul_ps(_mm256_fmadd_ps(two, prob_bb_f, prob_ab_f), scale);
            
            _mm256_storeu_ps(output + i, dosage);
            
            // Handle missing values
            if (missing_mask) {
                for (int j = 0; j < 8; ++j) {
                    int idx = sample_indices[i + j];
                    int byte_idx = idx / 8;
                    int bit_idx = idx % 8;
                    if (missing_mask[byte_idx] & (1 << bit_idx)) {
                        output[i + j] = std::nanf("");
                    }
                }
            }
        }
    }
#elif defined(__aarch64__)
    if (g_has_neon) {
        // Process 4 samples at a time with NEON
        const float32x4_t scale = vdupq_n_f32(1.0f / 65535.0f);
        const float32x4_t two = vdupq_n_f32(2.0f);
        
        for (; i + 3 < n_indices; i += 4) {
            // Gather data for 4 selected samples
            uint16x4_t prob_aa = {
                read_le16(prob_data + sample_indices[i + 0] * 4),
                read_le16(prob_data + sample_indices[i + 1] * 4),
                read_le16(prob_data + sample_indices[i + 2] * 4),
                read_le16(prob_data + sample_indices[i + 3] * 4)
            };
            
            uint16x4_t prob_ab = {
                read_le16(prob_data + sample_indices[i + 0] * 4 + 2),
                read_le16(prob_data + sample_indices[i + 1] * 4 + 2),
                read_le16(prob_data + sample_indices[i + 2] * 4 + 2),
                read_le16(prob_data + sample_indices[i + 3] * 4 + 2)
            };
            
            // Calculate P(BB) = 65535 - P(AA) - P(AB)
            uint32x4_t prob_aa_32 = vmovl_u16(prob_aa);
            uint32x4_t prob_ab_32 = vmovl_u16(prob_ab);
            uint32x4_t sum = vaddq_u32(prob_aa_32, prob_ab_32);
            uint32x4_t prob_bb_32 = vsubq_u32(vdupq_n_u32(65535), sum);
            
            // Convert to float and calculate dosage
            float32x4_t prob_ab_f = vcvtq_f32_u32(prob_ab_32);
            float32x4_t prob_bb_f = vcvtq_f32_u32(prob_bb_32);
            float32x4_t dosage = vmulq_f32(vmlaq_f32(prob_ab_f, two, prob_bb_f), scale);
            
            vst1q_f32(output + i, dosage);
            
            // Handle missing values
            if (missing_mask) {
                for (int j = 0; j < 4; ++j) {
                    int idx = sample_indices[i + j];
                    int byte_idx = idx / 8;
                    int bit_idx = idx % 8;
                    if (missing_mask[byte_idx] & (1 << bit_idx)) {
                        output[i + j] = std::nanf("");
                    }
                }
            }
        }
    }
#endif
    
    // Scalar fallback for remaining samples
    for (; i < n_indices; ++i) {
        int idx = sample_indices[i];
        
        // Check if missing
        bool is_missing = false;
        if (missing_mask) {
            int byte_idx = idx / 8;
            int bit_idx = idx % 8;
            is_missing = (missing_mask[byte_idx] & (1 << bit_idx)) != 0;
        }
        
        if (is_missing) {
            output[i] = std::nanf("");
            continue;
        }
        
        uint16_t prob_aa = read_le16(prob_data + idx * 4);
        uint16_t prob_ab = read_le16(prob_data + idx * 4 + 2);
        
        if (prob_aa + prob_ab > 65535) {
            output[i] = std::nanf("");
        } else {
            uint16_t prob_bb = 65535 - prob_aa - prob_ab;
            output[i] = (prob_ab + 2.0f * prob_bb) / 65535.0f;
        }
    }
}

static void compute_dosages_filtered_32bit_simd(
    const uint8_t* prob_data,
    float* output,
    const int* sample_indices,
    size_t n_indices,
    const uint8_t* missing_mask
) {
    size_t i = 0;
    
#ifdef __x86_64__
    if (g_has_avx2) {
        // Process 4 samples at a time with AVX2
        const __m256d scale = _mm256_set1_pd(1.0 / 4294967295.0);
        const __m256d two = _mm256_set1_pd(2.0);
        
        for (; i + 3 < n_indices; i += 4) {
            // Gather data for 4 selected samples
            uint64_t prob_aa[4], prob_ab[4];
            for (int j = 0; j < 4; ++j) {
                int idx = sample_indices[i + j];
                prob_aa[j] = read_le32(prob_data + idx * 8);
                prob_ab[j] = read_le32(prob_data + idx * 8 + 4);
            }
            
            // Calculate P(BB) and dosage
            uint64_t prob_bb[4];
            double dosages[4];
            for (int j = 0; j < 4; ++j) {
                if (prob_aa[j] + prob_ab[j] > 4294967295UL) {
                    dosages[j] = std::nan("");
                    prob_bb[j] = 0;
                } else {
                    prob_bb[j] = 4294967295UL - prob_aa[j] - prob_ab[j];
                    dosages[j] = (static_cast<double>(prob_ab[j]) + 2.0 * prob_bb[j]) / 4294967295.0;
                }
            }
            
            // Store results
            for (int j = 0; j < 4; ++j) {
                output[i + j] = static_cast<float>(dosages[j]);
            }
            
            // Handle missing values
            if (missing_mask) {
                for (int j = 0; j < 4; ++j) {
                    int idx = sample_indices[i + j];
                    int byte_idx = idx / 8;
                    int bit_idx = idx % 8;
                    if (missing_mask[byte_idx] & (1 << bit_idx)) {
                        output[i + j] = std::nanf("");
                    }
                }
            }
        }
    }
#endif
    
    // Scalar fallback (also used for ARM)
    for (; i < n_indices; ++i) {
        int idx = sample_indices[i];
        
        // Check if missing
        bool is_missing = false;
        if (missing_mask) {
            int byte_idx = idx / 8;
            int bit_idx = idx % 8;
            is_missing = (missing_mask[byte_idx] & (1 << bit_idx)) != 0;
        }
        
        if (is_missing) {
            output[i] = std::nanf("");
            continue;
        }
        
        uint32_t prob_aa = read_le32(prob_data + idx * 8);
        uint32_t prob_ab = read_le32(prob_data + idx * 8 + 4);
        
        if (static_cast<uint64_t>(prob_aa) + prob_ab > 4294967295UL) {
            output[i] = std::nanf("");
        } else {
            uint32_t prob_bb = 4294967295UL - prob_aa - prob_ab;
            double dosage = (static_cast<double>(prob_ab) + 2.0 * prob_bb) / 4294967295.0;
            output[i] = static_cast<float>(dosage);
        }
    }
}

} // namespace simd
} // namespace bgen
} // namespace ldcov