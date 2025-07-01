#include "variant_parser.h"

#include <cstring>
#include <sstream>

namespace ldcov {
namespace bgen {

std::string VariantParser::readLengthPrefixedString(const uint8_t* buffer, size_t& pos,
                                                    size_t max_size, bool use_32bit_length) {
    size_t length_size = use_32bit_length ? 4 : 2;

    if (pos + length_size > max_size) {
        throw std::runtime_error("Buffer too small for string length");
    }

    uint32_t length;
    if (use_32bit_length) {
        length = readLE<uint32_t>(buffer + pos);
    } else {
        length = readLE<uint16_t>(buffer + pos);
    }
    pos += length_size;

    if (pos + length > max_size) {
        throw std::runtime_error("Buffer too small for string data");
    }

    std::string result;
    if (length > 0) {
        result.assign(reinterpret_cast<const char*>(buffer + pos), length);
    }
    pos += length;

    return result;
}

std::pair<VariantMetadata, size_t> VariantParser::parseV11(const uint8_t* buffer, size_t size,
                                                           CompressionType compression,
                                                           uint32_t expected_samples) {
    VariantMetadata variant;
    size_t pos = 0;

    // Read number of samples (4 bytes)
    if (pos + 4 > size) {
        throw std::runtime_error("Buffer too small for v1.1 sample count");
    }
    uint32_t n_samples = readLE<uint32_t>(buffer + pos);
    pos += 4;

    if (n_samples != expected_samples) {
        throw std::runtime_error("Sample count mismatch in variant");
    }

    // Read variant ID
    variant.varid = readLengthPrefixedString(buffer, pos, size, false);

    // Read rsID
    variant.rsid = readLengthPrefixedString(buffer, pos, size, false);

    // Read chromosome
    variant.chrom = readLengthPrefixedString(buffer, pos, size, false);

    // Read position (4 bytes)
    if (pos + 4 > size) {
        throw std::runtime_error("Buffer too small for position");
    }
    variant.pos = readLE<uint32_t>(buffer + pos);
    pos += 4;

    // v1.1 is always biallelic
    variant.n_alleles = 2;
    variant.alleles.reserve(2);

    // Read alleles (each with 32-bit length prefix)
    for (int i = 0; i < 2; ++i) {
        variant.alleles.push_back(readLengthPrefixedString(buffer, pos, size, true));
    }

    // Calculate genotype data length
    if (compression == CompressionType::None) {
        // Uncompressed: 6 bytes per sample (v1.1 format only)
        // For v1.1, there's no length prefix
        variant.genotype_length = n_samples * 6;
    } else {
        // Compressed: read the length
        if (pos + 4 > size) {
            throw std::runtime_error("Buffer too small for compressed genotype length");
        }
        variant.genotype_length = readLE<uint32_t>(buffer + pos);
        pos += 4;
    }

    // Store genotype offset (relative to start of variant)
    variant.genotype_offset = pos;

    // Total size includes genotype data
    size_t total_size = pos + variant.genotype_length;

    return {std::move(variant), total_size};
}

std::pair<VariantMetadata, size_t> VariantParser::parseV12(const uint8_t* buffer, size_t size) {
    VariantMetadata variant;
    size_t pos = 0;

    // v1.2 format goes straight to variant ID (no variant data block length)

    // Read variant ID
    variant.varid = readLengthPrefixedString(buffer, pos, size, false);

    // Read rsID
    variant.rsid = readLengthPrefixedString(buffer, pos, size, false);

    // Read chromosome
    variant.chrom = readLengthPrefixedString(buffer, pos, size, false);

    // Read position (4 bytes)
    if (pos + 4 > size) {
        throw std::runtime_error("Buffer too small for position");
    }
    variant.pos = readLE<uint32_t>(buffer + pos);
    pos += 4;

    // Read number of alleles (2 bytes)
    if (pos + 2 > size) {
        throw std::runtime_error("Buffer too small for allele count");
    }
    variant.n_alleles = readLE<uint16_t>(buffer + pos);
    pos += 2;

    // Read each allele
    variant.alleles.reserve(variant.n_alleles);
    for (uint16_t i = 0; i < variant.n_alleles; ++i) {
        variant.alleles.push_back(readLengthPrefixedString(buffer, pos, size, true));
    }

    // Read genotype data block length (4 bytes)
    if (pos + 4 > size) {
        throw std::runtime_error("Buffer too small for genotype block length");
    }
    variant.genotype_length = readLE<uint32_t>(buffer + pos);
    pos += 4;

    // Store genotype offset (relative to start of variant)
    variant.genotype_offset = pos;

    // Total size includes genotype data
    size_t total_size = pos + variant.genotype_length;

    return {std::move(variant), total_size};
}

std::pair<VariantMetadata, size_t> VariantParser::parse(const uint8_t* buffer, size_t size,
                                                        LayoutType layout,
                                                        CompressionType compression,
                                                        uint32_t expected_samples) {
    if (layout == LayoutType::V11) {
        return parseV11(buffer, size, compression, expected_samples);
    } else if (layout == LayoutType::V12) {
        return parseV12(buffer, size);
    } else {
        throw std::runtime_error("Unsupported BGEN layout version");
    }
}

size_t VariantParser::getVariantMetadataSize(const uint8_t* buffer, size_t size,
                                             LayoutType layout) {
    size_t pos = 0;

    if (layout == LayoutType::V11) {
        // Skip sample count (4 bytes)
        pos += 4;

        // Skip variant ID
        if (pos + 2 > size)
            return 0;
        uint16_t len = readLE<uint16_t>(buffer + pos);
        pos += 2 + len;

        // Skip rsID
        if (pos + 2 > size)
            return 0;
        len = readLE<uint16_t>(buffer + pos);
        pos += 2 + len;

        // Skip chromosome
        if (pos + 2 > size)
            return 0;
        len = readLE<uint16_t>(buffer + pos);
        pos += 2 + len;

        // Skip position (4 bytes)
        pos += 4;

        // Skip 2 alleles (each with 32-bit length)
        for (int i = 0; i < 2; ++i) {
            if (pos + 4 > size)
                return 0;
            uint32_t allele_len = readLE<uint32_t>(buffer + pos);
            pos += 4 + allele_len;
        }

        return pos;
    } else {  // V12
        // Skip variant ID
        if (pos + 2 > size)
            return 0;
        uint16_t len = readLE<uint16_t>(buffer + pos);
        pos += 2 + len;

        // Skip rsID
        if (pos + 2 > size)
            return 0;
        len = readLE<uint16_t>(buffer + pos);
        pos += 2 + len;

        // Skip chromosome
        if (pos + 2 > size)
            return 0;
        len = readLE<uint16_t>(buffer + pos);
        pos += 2 + len;

        // Skip position (4 bytes)
        pos += 4;

        // Read number of alleles (2 bytes)
        if (pos + 2 > size)
            return 0;
        uint16_t n_alleles = readLE<uint16_t>(buffer + pos);
        pos += 2;

        // Skip alleles
        for (uint16_t i = 0; i < n_alleles; ++i) {
            if (pos + 4 > size)
                return 0;
            uint32_t allele_len = readLE<uint32_t>(buffer + pos);
            pos += 4 + allele_len;
        }

        // Skip genotype block length (4 bytes)
        pos += 4;

        return pos;
    }
}

size_t VariantParser::skipVariant(const uint8_t* buffer, size_t size, LayoutType layout,
                                  CompressionType compression, uint32_t expected_samples) {
    // Parse the variant to get its total size
    auto parse_result = parse(buffer, size, layout, compression, expected_samples);
    return parse_result.second;
}

std::vector<VariantMetadata> BatchVariantParser::parseBatch(const uint8_t* buffer, size_t size,
                                                            LayoutType layout,
                                                            CompressionType compression,
                                                            uint32_t expected_samples,
                                                            size_t max_variants) {
    std::vector<VariantMetadata> variants;
    size_t pos = 0;
    size_t count = 0;

    while (pos < size && (max_variants == 0 || count < max_variants)) {
        try {
            auto parse_result = VariantParser::parse(buffer + pos, size - pos, layout, compression,
                                                     expected_samples);
            VariantMetadata variant = parse_result.first;
            size_t variant_size = parse_result.second;

            // Set file offset
            variant.file_offset = pos;

            variants.push_back(std::move(variant));
            pos += variant_size;
            count++;
        } catch (const std::exception& e) {
            // Stop parsing on error
            break;
        }
    }

    return variants;
}

std::vector<VariantMetadata> BatchVariantParser::parseAtOffsets(
    const uint8_t* buffer, size_t size, const std::vector<uint64_t>& offsets, LayoutType layout,
    CompressionType compression, uint32_t expected_samples) {
    std::vector<VariantMetadata> variants;
    variants.reserve(offsets.size());

    for (uint64_t offset : offsets) {
        if (offset >= size) {
            throw std::runtime_error("Variant offset " + std::to_string(offset) +
                                     " exceeds buffer size " + std::to_string(size));
        }

        auto parse_result = VariantParser::parse(buffer + offset, size - offset, layout,
                                                 compression, expected_samples);
        VariantMetadata variant = parse_result.first;
        size_t variant_size = parse_result.second;

        // Set file offset
        variant.file_offset = offset;

        variants.push_back(std::move(variant));
    }

    return variants;
}

// Explicit template instantiation
template uint16_t VariantParser::readLE<uint16_t>(const uint8_t*);
template uint32_t VariantParser::readLE<uint32_t>(const uint8_t*);

}  // namespace bgen
}  // namespace ldcov