#include "bgen_header.h"
#include <cstring>
#include <algorithm>

namespace ldcov {
namespace bgen {

BgenHeader BgenHeaderParser::parse(const uint8_t* buffer, size_t size) {
    if (size < 20) {
        throw std::runtime_error("Buffer too small for BGEN header");
    }
    
    BgenHeader header;
    size_t pos = 0;
    
    // Read offset to variant data (4 bytes)
    header.offset = readLE<uint32_t>(buffer + pos);
    pos += 4;
    
    // Read header length (4 bytes)
    uint32_t header_length = readLE<uint32_t>(buffer + pos);
    pos += 4;
    
    if (size < header_length) {
        throw std::runtime_error("Buffer too small for complete header");
    }
    
    // Read number of variants (4 bytes)
    header.nvariants = readLE<uint32_t>(buffer + pos);
    pos += 4;
    
    // Read number of samples (4 bytes)
    header.nsamples = readLE<uint32_t>(buffer + pos);
    pos += 4;
    
    // Check for magic number to determine version
    bool is_v12 = false;
    if (header_length >= 20 && pos < size - 4) {
        // Check if next 4 bytes are "bgen"
        if (std::memcmp(buffer + pos, "bgen", 4) == 0) {
            is_v12 = true;
            pos += 4;  // Skip magic number
        }
    }
    
    // Read flags (4 bytes) - after the header
    if (size < header_length + 4) {
        throw std::runtime_error("Buffer too small for flags");
    }
    header.flags = readLE<uint32_t>(buffer + header_length);
    
    // Decode flags
    uint32_t compression_flag = header.flags & 3;
    switch (compression_flag) {
        case 0:
            header.compression = CompressionType::None;
            break;
        case 1:
            header.compression = CompressionType::Zlib;
            break;
        case 2:
            header.compression = CompressionType::Zstd;
            break;
        default:
            throw std::runtime_error("Unknown compression type: " + std::to_string(compression_flag));
    }
    
    uint32_t layout_flag = (header.flags >> 2) & 15;
    switch (layout_flag) {
        case 1:
            header.layout = LayoutType::V11;
            break;
        case 2:
            header.layout = LayoutType::V12;
            break;
        default:
            throw std::runtime_error("Unsupported layout version: " + std::to_string(layout_flag));
    }
    
    // Check if sample IDs are present
    header.has_sample_ids = (header.flags >> 31) & 1;
    
    return header;
}

size_t BgenHeaderParser::getHeaderSize(const uint8_t* buffer, size_t size) {
    if (size < 8) {
        throw std::runtime_error("Buffer too small to read header size");
    }
    
    // Skip offset (4 bytes) and read header length (4 bytes)
    uint32_t header_length = readLE<uint32_t>(buffer + 4);
    
    // Total size includes header length + flags (4 bytes)
    return header_length + 4;
}

bool BgenHeaderParser::isValidBgen(const uint8_t* buffer, size_t size) {
    try {
        if (size < 20) {
            return false;
        }
        
        // Read header length
        uint32_t header_length = readLE<uint32_t>(buffer + 4);
        if (header_length < 12) {  // Minimum valid header length
            return false;
        }
        
        // For v1.2, check magic number
        if (header_length >= 20 && size >= 20) {
            if (std::memcmp(buffer + 16, "bgen", 4) == 0) {
                return true;
            }
        }
        
        // For v1.1, check basic structure
        if (header_length >= 16 && size >= header_length + 4) {
            // Read flags and check layout version
            uint32_t flags = readLE<uint32_t>(buffer + header_length);
            uint32_t layout = (flags >> 2) & 15;
            return (layout == 1 || layout == 2);
        }
        
        return false;
    } catch (...) {
        return false;
    }
}

SampleBlock SampleBlockParser::parse(const uint8_t* buffer, size_t size, uint32_t expected_samples) {
    if (size < 8) {
        throw std::runtime_error("Buffer too small for sample block");
    }
    
    SampleBlock block;
    size_t pos = 0;
    
    // Read sample block length (4 bytes)
    block.block_size = readLE<uint32_t>(buffer + pos);
    pos += 4;
    
    if (size < block.block_size + 4) {
        throw std::runtime_error("Buffer too small for complete sample block");
    }
    
    // Read number of samples (4 bytes)
    uint32_t n_samples = readLE<uint32_t>(buffer + pos);
    pos += 4;
    
    if (n_samples != expected_samples) {
        throw std::runtime_error("Sample count mismatch: expected " + 
                                 std::to_string(expected_samples) + 
                                 ", got " + std::to_string(n_samples));
    }
    
    // Read each sample ID
    block.sample_ids.reserve(n_samples);
    for (uint32_t i = 0; i < n_samples; ++i) {
        if (pos + 2 > size) {
            throw std::runtime_error("Buffer too small for sample ID length");
        }
        
        // Read ID length (2 bytes)
        uint16_t id_length = readLE<uint16_t>(buffer + pos);
        pos += 2;
        
        if (pos + id_length > size) {
            throw std::runtime_error("Buffer too small for sample ID");
        }
        
        // Read ID string
        std::string sample_id(reinterpret_cast<const char*>(buffer + pos), id_length);
        block.sample_ids.push_back(std::move(sample_id));
        pos += id_length;
    }
    
    return block;
}

uint32_t SampleBlockParser::getSampleBlockSize(const uint8_t* buffer, size_t size) {
    if (size < 4) {
        throw std::runtime_error("Buffer too small to read sample block size");
    }
    
    // Read sample block length
    return readLE<uint32_t>(buffer) + 4;  // Include the length field itself
}

// Explicit template instantiation
template uint16_t BgenHeaderParser::readLE<uint16_t>(const uint8_t*);
template uint32_t BgenHeaderParser::readLE<uint32_t>(const uint8_t*);

} // namespace bgen
} // namespace ldcov