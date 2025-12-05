#include "genotype_parser.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "genotype_parser_simd.h"

namespace ldcov {
namespace bgen {

// GenotypeData member implementations
void GenotypeData::computeDosages(float* output) const {
    if (n_alleles != 2) {
        throw std::runtime_error("Dosage computation only supported for biallelic variants");
    }

    // For biallelic variants, dosage = 0*P(AA) + 1*P(Aa) + 2*P(aa)
    // where probabilities are stored as [P(AA), P(Aa), P(aa)] for each sample
    for (uint32_t i = 0; i < n_samples; ++i) {
        if (missing[i]) {
            output[i] = std::nanf("");
        } else {
            size_t offset = i * 3;
            output[i] = probabilities[offset + 1] + 2.0f * probabilities[offset + 2];
        }
    }
}

void GenotypeData::computeDosagesFiltered(const int* sample_indices, int n_indices,
                                          float* output) const {
    if (n_alleles != 2) {
        throw std::runtime_error("Dosage computation only supported for biallelic variants");
    }

    for (int i = 0; i < n_indices; ++i) {
        int idx = sample_indices[i];
        if (idx < 0 || idx >= static_cast<int>(n_samples)) {
            throw std::runtime_error("Sample index out of bounds");
        }

        if (missing[idx]) {
            output[i] = std::nanf("");
        } else {
            size_t offset = idx * 3;
            output[i] = probabilities[offset + 1] + 2.0f * probabilities[offset + 2];
        }
    }
}

// GenotypeParser implementations
std::unique_ptr<GenotypeData> GenotypeParser::parseV11(const uint8_t* buffer, size_t size,
                                                       uint32_t n_samples) {
    auto data = std::unique_ptr<GenotypeData>(new GenotypeData());
    data->n_samples = n_samples;
    data->n_alleles = 2;  // v1.1 is always biallelic
    data->phased = false;
    data->constant_ploidy = true;
    data->min_ploidy = 2;
    data->max_ploidy = 2;

    // v1.1 format: 6 bytes per sample
    if (size < n_samples * 6) {
        throw std::runtime_error("Buffer too small for v1.1 genotype data");
    }

    data->ploidy.resize(n_samples, 2);
    data->probabilities.resize(n_samples * 3);
    data->missing.resize(n_samples);

    const uint8_t* ptr = buffer;

    for (uint32_t i = 0; i < n_samples; ++i) {
        // Read 3 probabilities (2 bytes each)
        uint16_t prob_aa = readLE<uint16_t>(ptr);
        ptr += 2;
        uint16_t prob_ab = readLE<uint16_t>(ptr);
        ptr += 2;
        uint16_t prob_bb = readLE<uint16_t>(ptr);
        ptr += 2;

        // Check for missing data (all probs = 0)
        if (prob_aa == 0 && prob_ab == 0 && prob_bb == 0) {
            data->missing[i] = true;
            data->probabilities[i * 3] = 0.0f;
            data->probabilities[i * 3 + 1] = 0.0f;
            data->probabilities[i * 3 + 2] = 0.0f;
        } else {
            data->missing[i] = false;
            // Convert from 16-bit to float probabilities
            float sum = static_cast<float>(prob_aa + prob_ab + prob_bb);
            data->probabilities[i * 3] = prob_aa / sum;
            data->probabilities[i * 3 + 1] = prob_ab / sum;
            data->probabilities[i * 3 + 2] = prob_bb / sum;
        }
    }

    return data;
}

std::unique_ptr<GenotypeData> GenotypeParser::parseV12(const uint8_t* buffer, size_t size,
                                                       uint32_t n_samples, uint16_t n_alleles) {
    if (size < 10) {  // Minimum size for header
        throw std::runtime_error("Buffer too small for v1.2 genotype data");
    }

    auto data = std::unique_ptr<GenotypeData>(new GenotypeData());
    data->n_samples = n_samples;
    data->n_alleles = n_alleles;

    const uint8_t* ptr = buffer;

    // Read header
    uint32_t n_samples_check = readLE<uint32_t>(ptr);
    ptr += 4;

    if (n_samples_check != n_samples) {
        throw std::runtime_error("Sample count mismatch in genotype data");
    }

    uint16_t n_alleles_check = readLE<uint16_t>(ptr);
    ptr += 2;

    if (n_alleles_check != n_alleles) {
        throw std::runtime_error("Allele count mismatch in genotype data");
    }

    // Read ploidy and phase info
    uint8_t min_ploidy = *ptr++;
    uint8_t max_ploidy = *ptr++;

    data->min_ploidy = min_ploidy & 0x3F;
    data->max_ploidy = max_ploidy & 0x3F;
    data->constant_ploidy = (data->min_ploidy == data->max_ploidy);

    // Read ploidy for each sample
    data->ploidy.resize(n_samples);
    data->missing.resize(n_samples);

    if (data->constant_ploidy) {
        // All samples have same ploidy
        uint8_t ploidy = data->min_ploidy;
        std::fill(data->ploidy.begin(), data->ploidy.end(), ploidy);

        // Read missing flags - bit array of (n_samples + 7) / 8 bytes
        size_t missing_bytes = (n_samples + 7) / 8;
        for (uint32_t i = 0; i < n_samples; ++i) {
            size_t byte_idx = i / 8;
            size_t bit_idx = i % 8;
            if (byte_idx < missing_bytes) {
                data->missing[i] = (ptr[byte_idx] & (1 << bit_idx)) != 0;
            } else {
                data->missing[i] = false;
            }
        }
        ptr += missing_bytes;
    } else {
        // Variable ploidy
        for (uint32_t i = 0; i < n_samples; ++i) {
            uint8_t ploidy_missing = *ptr++;
            data->ploidy[i] = ploidy_missing & 0x3F;
            data->missing[i] = (ploidy_missing & 0x80) != 0;
        }
    }

    // Check phase info
    data->phased = (*ptr++ & 1) != 0;

    // Read bits per probability
    uint8_t bits_per_prob = *ptr++;
    if (bits_per_prob != 8 && bits_per_prob != 16 && bits_per_prob != 32) {
        throw std::runtime_error("Unsupported bits per probability: " +
                                 std::to_string(bits_per_prob));
    }

    // Calculate number of probabilities per sample
    size_t probs_per_sample;
    if (data->phased) {
        probs_per_sample = n_alleles;
    } else {
        // Unphased: need probabilities for all possible genotypes
        probs_per_sample = n_alleles * (n_alleles + 1) / 2;
    }

    data->probabilities.resize(n_samples * probs_per_sample);

    // Read probability data
    for (uint32_t i = 0; i < n_samples; ++i) {
        if (data->missing[i]) {
            // Skip missing samples
            for (size_t j = 0; j < probs_per_sample; ++j) {
                data->probabilities[i * probs_per_sample + j] = 0.0f;
            }
        } else {
            // Read probabilities based on bit width
            if (bits_per_prob == 8) {
                uint32_t sum = 0;
                std::vector<uint8_t> probs(probs_per_sample);
                for (size_t j = 0; j < probs_per_sample - 1; ++j) {
                    probs[j] = *ptr++;
                    sum += probs[j];
                }
                probs[probs_per_sample - 1] = 255 - sum;

                // Convert to float
                for (size_t j = 0; j < probs_per_sample; ++j) {
                    data->probabilities[i * probs_per_sample + j] = probs[j] / 255.0f;
                }
            } else if (bits_per_prob == 16) {
                uint32_t sum = 0;
                std::vector<uint16_t> probs(probs_per_sample);
                for (size_t j = 0; j < probs_per_sample - 1; ++j) {
                    probs[j] = readLE<uint16_t>(ptr);
                    ptr += 2;
                    sum += probs[j];
                }
                probs[probs_per_sample - 1] = 65535 - sum;

                // Convert to float
                for (size_t j = 0; j < probs_per_sample; ++j) {
                    data->probabilities[i * probs_per_sample + j] = probs[j] / 65535.0f;
                }
            } else {  // 32 bits
                uint64_t sum = 0;
                std::vector<uint32_t> probs(probs_per_sample);
                for (size_t j = 0; j < probs_per_sample - 1; ++j) {
                    probs[j] = readLE<uint32_t>(ptr);
                    ptr += 4;
                    sum += probs[j];
                }
                probs[probs_per_sample - 1] = 4294967295UL - sum;

                // Convert to float
                for (size_t j = 0; j < probs_per_sample; ++j) {
                    data->probabilities[i * probs_per_sample + j] = probs[j] / 4294967295.0f;
                }
            }
        }
    }

    return data;
}

std::unique_ptr<GenotypeData> GenotypeParser::parse(const uint8_t* buffer, size_t size,
                                                    LayoutType layout, CompressionType compression,
                                                    uint32_t n_samples, uint16_t n_alleles) {
    // Handle decompression if needed
    std::vector<uint8_t> decompressed;
    const uint8_t* data_ptr = buffer;
    size_t data_size = size;

    if (compression != CompressionType::None) {
        // For now, throw an error - decompression should be handled externally
        throw std::runtime_error("Compressed genotype data should be decompressed before parsing");
    }

    return parseDecompressed(data_ptr, data_size, layout, n_samples, n_alleles);
}

std::unique_ptr<GenotypeData> GenotypeParser::parseDecompressed(const uint8_t* buffer, size_t size,
                                                                LayoutType layout,
                                                                uint32_t n_samples,
                                                                uint16_t n_alleles) {
    if (layout == LayoutType::V11) {
        return parseV11(buffer, size, n_samples);
    } else if (layout == LayoutType::V12) {
        try {
            return parseV12(buffer, size, n_samples, n_alleles);
        } catch (const std::runtime_error& e) {
            // Some BGEN files have V1.2 headers but V1.1 genotype blocks
            // If V1.2 parsing fails, try V1.1 format
            std::string error_msg = e.what();
            if (error_msg.find("Sample count mismatch") != std::string::npos ||
                error_msg.find("Allele count mismatch") != std::string::npos) {
                // Some BGEN files have malformed genotype data

                // Check if this could be V1.1 format (6 bytes per sample)
                if (size == n_samples * 6) {
                    return parseV11(buffer, size, n_samples);
                }
            }
            // Re-throw the original error if fallback doesn't apply
            throw;
        }
    } else {
        throw std::runtime_error("Unsupported BGEN layout version");
    }
}

void GenotypeParser::computeDosagesV11Direct(const uint8_t* buffer, size_t size, uint32_t n_samples,
                                             float* output) {
    if (size < n_samples * 6) {
        throw std::runtime_error("Buffer too small for v1.1 genotype data");
    }

    // Check if we can use SIMD optimization
    if (can_use_simd_dosage()) {
        // Use SIMD-optimized implementation
        simd::compute_dosages_v11_simd(buffer, n_samples, output);
        return;
    }

    // Fallback to scalar implementation
    const uint8_t* ptr = buffer;

    for (uint32_t i = 0; i < n_samples; ++i) {
        // Read 3 probabilities (2 bytes each)
        uint16_t prob_aa = readLE<uint16_t>(ptr);
        ptr += 2;
        uint16_t prob_ab = readLE<uint16_t>(ptr);
        ptr += 2;
        uint16_t prob_bb = readLE<uint16_t>(ptr);
        ptr += 2;

        // Check for missing data
        if (prob_aa == 0 && prob_ab == 0 && prob_bb == 0) {
            output[i] = std::nanf("");
        } else {
            // Compute dosage directly
            float sum = static_cast<float>(prob_aa + prob_ab + prob_bb);
            output[i] = (prob_ab + 2.0f * prob_bb) / sum;
        }
    }
}

void GenotypeParser::computeDosagesV12Direct(const uint8_t* buffer, size_t size, uint32_t n_samples,
                                             uint16_t n_alleles, float* output) {
    if (n_alleles != 2) {
        throw std::runtime_error("Direct dosage computation only supported for biallelic variants");
    }

    // Uncompressed data should not reach this point as it's blocked at the reader level
    // If we somehow get here with what looks like uncompressed data, reject it

    if (size < 10) {  // Minimum size for header
        throw std::runtime_error("Buffer too small for v1.2 genotype data");
    }

    // Parse header to get to probability data
    const uint8_t* ptr = buffer;

    // Verify n_samples
    uint32_t n_samples_check = readLE<uint32_t>(ptr);
    ptr += 4;
    if (n_samples_check != n_samples) {
        throw std::runtime_error("Sample count mismatch in genotype data");
    }

    // Verify n_alleles
    uint16_t n_alleles_check = readLE<uint16_t>(ptr);
    ptr += 2;
    if (n_alleles_check != n_alleles) {
        throw std::runtime_error("Allele count mismatch in genotype data");
    }

    uint8_t min_ploidy = *ptr++;
    uint8_t max_ploidy = *ptr++;

    min_ploidy &= 0x3F;
    max_ploidy &= 0x3F;
    // bool constant_ploidy = (min_ploidy == max_ploidy);  // Unused in this function

    // Save pointer to missing data for single-pass processing
    const uint8_t* missing_data_ptr = ptr;

    // Skip missing data section
    ptr += n_samples;

    // Skip phase info
    ptr++;

    // Read bits per probability
    uint8_t bits_per_prob = *ptr++;

    if (bits_per_prob != 8 && bits_per_prob != 16 && bits_per_prob != 32) {
        throw std::runtime_error("Unsupported bits per probability: " +
                                 std::to_string(bits_per_prob));
    }

    // Single-pass processing: read missing status and probabilities together
    const uint8_t* prob_ptr = ptr;

    for (uint32_t i = 0; i < n_samples; ++i) {
        // Read missing status for this sample
        uint8_t ploidy_missing = missing_data_ptr[i];
        bool is_missing = (ploidy_missing & 0x80) != 0;

        if (is_missing) {
            // Sample is missing - output NaN
            output[i] = std::nanf("");

            // Skip probability data for this sample
            if (bits_per_prob == 8) {
                prob_ptr += 2;  // Skip 2 bytes (prob_aa, prob_ab)
            } else if (bits_per_prob == 16) {
                prob_ptr += 4;  // Skip 4 bytes (2 x uint16_t)
            } else {            // 32 bits
                prob_ptr += 8;  // Skip 8 bytes (2 x uint32_t)
            }
        } else {
            // Read and compute dosage
            if (bits_per_prob == 8) {
                uint8_t prob_aa = *prob_ptr++;
                uint8_t prob_ab = *prob_ptr++;
                // prob_bb is implicit = 255 - prob_aa - prob_ab

                // Additional check for invalid data
                if (prob_aa + prob_ab > 255) {
                    output[i] = std::nanf("");
                } else {
                    uint8_t prob_bb = 255 - prob_aa - prob_ab;
                    output[i] = (prob_ab + 2.0f * prob_bb) / 255.0f;
                }
            } else if (bits_per_prob == 16) {
                uint16_t prob_aa = readLE<uint16_t>(prob_ptr);
                prob_ptr += 2;
                uint16_t prob_ab = readLE<uint16_t>(prob_ptr);
                prob_ptr += 2;

                // Additional check for invalid data
                if (prob_aa + prob_ab > 65535) {
                    output[i] = std::nanf("");
                } else {
                    uint16_t prob_bb = 65535 - prob_aa - prob_ab;
                    output[i] = (prob_ab + 2.0f * prob_bb) / 65535.0f;
                }
            } else {  // 32 bits
                uint32_t prob_aa = readLE<uint32_t>(prob_ptr);
                prob_ptr += 4;
                uint32_t prob_ab = readLE<uint32_t>(prob_ptr);
                prob_ptr += 4;

                // Additional check for invalid data
                if (static_cast<uint64_t>(prob_aa) + prob_ab > 4294967295UL) {
                    output[i] = std::nanf("");
                } else {
                    uint32_t prob_bb = 4294967295UL - prob_aa - prob_ab;
                    // Use double precision to avoid overflow
                    double dosage = (static_cast<double>(prob_ab) + 2.0 * prob_bb) / 4294967295.0;
                    output[i] = static_cast<float>(dosage);
                }
            }
        }
    }

    // Verify we consumed the expected amount of data
    size_t expected_ptr_offset = prob_ptr - buffer;
    if (expected_ptr_offset > size) {
        throw std::runtime_error("Buffer overrun while parsing genotype data");
    }
}

void GenotypeParser::computeDosagesV12Filtered(const uint8_t* buffer, size_t size,
                                               uint32_t n_samples, uint16_t n_alleles,
                                               const int* sample_indices, int n_indices,
                                               float* output) {
    if (n_alleles != 2) {
        throw std::runtime_error(
            "Filtered dosage computation only supported for biallelic variants");
    }

    if (size < 10) {  // Minimum size for header
        throw std::runtime_error("Buffer too small for v1.2 genotype data");
    }

    // Parse header
    const uint8_t* ptr = buffer;

    // Verify n_samples
    uint32_t n_samples_check = readLE<uint32_t>(ptr);
    ptr += 4;
    if (n_samples_check != n_samples) {
        throw std::runtime_error("Sample count mismatch in genotype data");
    }

    // Verify n_alleles
    uint16_t n_alleles_check = readLE<uint16_t>(ptr);
    ptr += 2;
    if (n_alleles_check != n_alleles) {
        throw std::runtime_error("Allele count mismatch in genotype data");
    }

    uint8_t min_ploidy = *ptr++;
    uint8_t max_ploidy = *ptr++;

    min_ploidy &= 0x3F;
    max_ploidy &= 0x3F;
    bool constant_ploidy = (min_ploidy == max_ploidy);

    // Save pointer to missing data start
    const uint8_t* missing_data_ptr = ptr;

    // Skip missing data section - consistent with legacy reader behavior
    // For both constant and variable ploidy, BGEN v1.2 uses n_samples bytes
    ptr += n_samples;

    // Skip phase info
    ptr++;

    // Read bits per probability
    uint8_t bits_per_prob = *ptr++;

    if (bits_per_prob != 8 && bits_per_prob != 16 && bits_per_prob != 32) {
        throw std::runtime_error("Unsupported bits per probability: " +
                                 std::to_string(bits_per_prob));
    }

    // Save pointer to probability data start
    const uint8_t* prob_data_ptr = ptr;

    // Check if we can use SIMD for filtered computation
    if (can_use_simd_dosage()) {
        // Use SIMD-optimized filtered computation
        simd::compute_dosages_filtered_simd(prob_data_ptr, output, sample_indices, n_indices,
                                            bits_per_prob,
                                            constant_ploidy ? missing_data_ptr : nullptr);
        return;
    }

    // Fallback to scalar processing
    // Process only requested samples
    for (int idx = 0; idx < n_indices; ++idx) {
        int sample_idx = sample_indices[idx];

        // Validate sample index
        if (sample_idx < 0 || sample_idx >= static_cast<int>(n_samples)) {
            output[idx] = std::nanf("");
            continue;
        }

        // Check if sample is missing
        bool is_missing = false;
        if (constant_ploidy) {
            // Read from bit array
            int byte_idx = sample_idx / 8;
            int bit_idx = sample_idx % 8;
            uint8_t missing_byte = missing_data_ptr[byte_idx];
            is_missing = (missing_byte & (1 << bit_idx)) != 0;
        } else {
            // Read from byte array
            uint8_t ploidy_missing = missing_data_ptr[sample_idx];
            is_missing = (ploidy_missing & 0x80) != 0;
        }

        if (is_missing) {
            output[idx] = std::nanf("");
            continue;
        }

        // Calculate offset to this sample's probability data
        const uint8_t* sample_prob_ptr = prob_data_ptr;

        if (bits_per_prob == 8) {
            // 2 bytes per sample
            sample_prob_ptr += sample_idx * 2;

            uint8_t prob_aa = sample_prob_ptr[0];
            uint8_t prob_ab = sample_prob_ptr[1];

            if (prob_aa + prob_ab > 255) {
                output[idx] = std::nanf("");
            } else {
                uint8_t prob_bb = 255 - prob_aa - prob_ab;
                output[idx] = (prob_ab + 2.0f * prob_bb) / 255.0f;
            }
        } else if (bits_per_prob == 16) {
            // 4 bytes per sample
            sample_prob_ptr += sample_idx * 4;

            uint16_t prob_aa = readLE<uint16_t>(sample_prob_ptr);
            uint16_t prob_ab = readLE<uint16_t>(sample_prob_ptr + 2);

            if (prob_aa + prob_ab > 65535) {
                output[idx] = std::nanf("");
            } else {
                uint16_t prob_bb = 65535 - prob_aa - prob_ab;
                output[idx] = (prob_ab + 2.0f * prob_bb) / 65535.0f;
            }
        } else {  // 32 bits
            // 8 bytes per sample
            sample_prob_ptr += sample_idx * 8;

            uint32_t prob_aa = readLE<uint32_t>(sample_prob_ptr);
            uint32_t prob_ab = readLE<uint32_t>(sample_prob_ptr + 4);

            if (static_cast<uint64_t>(prob_aa) + prob_ab > 4294967295UL) {
                output[idx] = std::nanf("");
            } else {
                uint32_t prob_bb = 4294967295UL - prob_aa - prob_ab;
                double dosage = (static_cast<double>(prob_ab) + 2.0 * prob_bb) / 4294967295.0;
                output[idx] = static_cast<float>(dosage);
            }
        }
    }
}

void GenotypeParser::computeDosagesDirect(const uint8_t* buffer, size_t size, LayoutType layout,
                                          CompressionType compression, uint32_t n_samples,
                                          uint16_t n_alleles, float* output) {
    if (compression != CompressionType::None) {
        throw std::runtime_error(
            "Compressed genotype data should be decompressed before computing dosages");
    }

    if (layout == LayoutType::V11) {
        computeDosagesV11Direct(buffer, size, n_samples, output);
    } else if (layout == LayoutType::V12) {
        try {
            computeDosagesV12Direct(buffer, size, n_samples, n_alleles, output);
        } catch (const std::runtime_error& e) {
            // Some BGEN files have V1.2 headers but V1.1 genotype blocks
            // If V1.2 parsing fails, try V1.1 format
            std::string error_msg = e.what();
            if (error_msg.find("Sample count mismatch") != std::string::npos ||
                error_msg.find("Allele count mismatch") != std::string::npos) {
                // Some BGEN files have malformed genotype data

                // Check if this could be V1.1 format (6 bytes per sample)
                if (size == n_samples * 6) {
                    computeDosagesV11Direct(buffer, size, n_samples, output);
                    return;
                }
            }
            // Re-throw the original error if fallback doesn't apply
            throw;
        }
    } else {
        throw std::runtime_error("Unsupported BGEN layout version");
    }
}

void GenotypeParser::computeDosagesFiltered(const uint8_t* buffer, size_t size, LayoutType layout,
                                            CompressionType compression, uint32_t n_samples,
                                            uint16_t n_alleles, const int* sample_indices,
                                            int n_indices, float* output) {
    // Use optimized implementation for v1.2 that only processes requested samples
    if (layout == LayoutType::V12 && compression == CompressionType::None) {
        computeDosagesV12Filtered(buffer, size, n_samples, n_alleles, sample_indices, n_indices,
                                  output);
        return;
    }

    // For v1.1 or compressed data, fall back to full parsing
    // (compressed data should already be decompressed before reaching here)
    auto data = parse(buffer, size, layout, compression, n_samples, n_alleles);
    data->computeDosagesFiltered(sample_indices, n_indices, output);
}

// BatchGenotypeParser implementations
std::vector<std::unique_ptr<GenotypeData>> BatchGenotypeParser::parseBatch(
    const std::vector<const uint8_t*>& buffers, const std::vector<size_t>& sizes, LayoutType layout,
    CompressionType compression, uint32_t n_samples, const std::vector<uint16_t>& n_alleles_list) {
    if (buffers.size() != sizes.size() || buffers.size() != n_alleles_list.size()) {
        throw std::runtime_error("Mismatched input sizes for batch parsing");
    }

    std::vector<std::unique_ptr<GenotypeData>> results;
    results.reserve(buffers.size());

    for (size_t i = 0; i < buffers.size(); ++i) {
        results.push_back(GenotypeParser::parse(buffers[i], sizes[i], layout, compression,
                                                n_samples, n_alleles_list[i]));
    }

    return results;
}

void BatchGenotypeParser::computeDosagesBatch(const std::vector<const uint8_t*>& buffers,
                                              const std::vector<size_t>& sizes, LayoutType layout,
                                              CompressionType compression, uint32_t n_samples,
                                              const std::vector<uint16_t>& n_alleles_list,
                                              float* output, size_t output_stride) {
    if (buffers.size() != sizes.size() || buffers.size() != n_alleles_list.size()) {
        throw std::runtime_error("Mismatched input sizes for batch dosage computation");
    }

    for (size_t i = 0; i < buffers.size(); ++i) {
        GenotypeParser::computeDosagesDirect(buffers[i], sizes[i], layout, compression, n_samples,
                                             n_alleles_list[i], output + i * output_stride);
    }
}

// Explicit template instantiation
template uint16_t GenotypeParser::readLE<uint16_t>(const uint8_t*);
template uint32_t GenotypeParser::readLE<uint32_t>(const uint8_t*);

}  // namespace bgen
}  // namespace ldcov