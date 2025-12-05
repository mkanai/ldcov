#include "bgen_reader_impl.h"

#include <iostream>

#include "file_reader_wrapper.h"
#include "format/bgen_header.h"
#include "format/genotype_parser.h"
#include "format/variant_parser.h"
#include "index/bgi_reader.h"
#include "io/gcs_file_reader.h"
#include "io/mmap_reader.h"
// #include "index/simple_bgi_cache.h"  // BGI download now handled by Python
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>

#include "decompress/decompressor_factory.h"

namespace ldcov {
namespace io {
namespace bgen {

// Using directives for nested namespaces
using index::BGIReader;
using ::ldcov::bgen::BgenHeaderParser;
using ::ldcov::bgen::CompressionType;
using ::ldcov::bgen::LayoutType;
using ::ldcov::bgen::SampleBlockParser;
using ::ldcov::bgen::VariantMetadata;
using ::ldcov::bgen::VariantParser;

// Decompress namespace
namespace decompress = ::ldcov::bgen::decompress;

// Regular file reader implementation
class RegularFileReader : public FileReader {
   public:
    explicit RegularFileReader(const std::string& filename) : filename_(filename), file_size_(0) {
        file_.open(filename, std::ios::binary);
        if (!file_.is_open()) {
            throw std::runtime_error("Failed to open file: " + filename);
        }

        // Get file size
        file_.seekg(0, std::ios::end);
        file_size_ = file_.tellg();
        file_.seekg(0, std::ios::beg);
    }

    ~RegularFileReader() override {
        close();
    }

    size_t read(uint8_t* buffer, size_t size) override {
        if (!file_.is_open()) {
            return 0;
        }
        file_.read(reinterpret_cast<char*>(buffer), size);
        return file_.gcount();
    }

    size_t read_at(uint64_t offset, uint8_t* buffer, size_t size) override {
        if (!file_.is_open()) {
            return 0;
        }
        auto current_pos = file_.tellg();
        file_.seekg(offset);
        size_t bytes_read = read(buffer, size);
        file_.seekg(current_pos);
        return bytes_read;
    }

    void seek(uint64_t offset) override {
        if (file_.is_open()) {
            file_.seekg(offset);
        }
    }

    uint64_t tell() const override {
        if (!file_.is_open()) {
            return 0;
        }
        return file_.tellg();
    }

    uint64_t size() const override {
        return file_size_;
    }

    bool is_open() const override {
        return file_.is_open();
    }

    void close() override {
        if (file_.is_open()) {
            file_.close();
        }
    }

    const std::string& filename() const override {
        return filename_;
    }

   private:
    std::string filename_;
    mutable std::ifstream file_;
    uint64_t file_size_;
};

// Memory-mapped file reader implementation
class MMapFileReader : public FileReader {
   public:
    explicit MMapFileReader(const std::string& filename)
        : filename_(filename), data_(nullptr), file_size_(0), current_pos_(0), fd_(-1) {
        // Open file
        fd_ = ::open(filename.c_str(), O_RDONLY);
        if (fd_ < 0) {
            throw std::runtime_error("Failed to open file: " + filename);
        }

        // Get file size
        struct stat st;
        if (fstat(fd_, &st) < 0) {
            ::close(fd_);
            throw std::runtime_error("Failed to stat file: " + filename);
        }
        file_size_ = st.st_size;

        // Memory map the file
        data_ = static_cast<uint8_t*>(mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0));
        if (data_ == MAP_FAILED) {
            ::close(fd_);
            throw std::runtime_error("Failed to mmap file: " + filename);
        }

        // Advise kernel about access pattern - default to random access
        // since ldcov primarily performs indexed lookups
        madvise(data_, file_size_, MADV_RANDOM);
    }

    ~MMapFileReader() override {
        close();
    }

    size_t read(uint8_t* buffer, size_t size) override {
        if (!data_ || current_pos_ >= file_size_) {
            return 0;
        }

        size_t bytes_to_read = std::min(size, static_cast<size_t>(file_size_ - current_pos_));
        memcpy(buffer, data_ + current_pos_, bytes_to_read);
        current_pos_ += bytes_to_read;
        return bytes_to_read;
    }

    size_t read_at(uint64_t offset, uint8_t* buffer, size_t size) override {
        if (!data_ || offset >= file_size_) {
            return 0;
        }

        size_t bytes_to_read = std::min(size, static_cast<size_t>(file_size_ - offset));
        memcpy(buffer, data_ + offset, bytes_to_read);
        return bytes_to_read;
    }

    void seek(uint64_t offset) override {
        current_pos_ = std::min(offset, file_size_);
    }

    uint64_t tell() const override {
        return current_pos_;
    }

    uint64_t size() const override {
        return file_size_;
    }

    bool is_open() const override {
        return data_ != nullptr;
    }

    void close() override {
        if (data_ && data_ != MAP_FAILED) {
            munmap(data_, file_size_);
            data_ = nullptr;
        }
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
    }

    const std::string& filename() const override {
        return filename_;
    }

    void advise(uint64_t offset, size_t length, int advice) override {
        if (!data_ || offset >= file_size_) {
            return;  // No-op if file not mapped or offset out of bounds
        }

        // Ensure we don't go past the end of the file
        size_t actual_length = std::min(length, static_cast<size_t>(file_size_ - offset));

        // Call madvise on the specified region
        // Note: madvise requires page-aligned addresses, but it's OK to pass
        // unaligned addresses - the kernel will handle the alignment
        if (madvise(data_ + offset, actual_length, advice) != 0) {
            // Log warning but don't throw - madvise is just a hint
            // In production, you might want to use a proper logging framework
            // For now, we'll silently ignore failures
        }
    }

   private:
    std::string filename_;
    uint8_t* data_;
    uint64_t file_size_;
    uint64_t current_pos_;
    int fd_;
};

// Implementation class (pimpl idiom)
class BgenReaderImpl::Impl {
   public:
    // Constructor
    Impl(const std::string& filename, const std::string& bgi_filename)
        : filename_(filename),
          bgi_filename_(bgi_filename),
          file_reader_(),
          bgi_reader_(),
          decompressor_(),
          header_(),
          sample_ids_(),
          sample_filter_(),
          is_open_(false),
          num_threads_(0),
          decompressor_type_("adaptive") {
        try {
            // Open the BGEN file
            openFile();

            // Read and parse header
            parseHeader();

            // Open BGI index
            openIndex();

            // Read sample IDs if present
            readSampleIds();

            // Create default decompressor (adaptive)
            createDefaultDecompressor();

            is_open_ = true;
        } catch (const std::exception& e) {
            // Clean up on error
            close();
            throw;
        }
    }

    // Destructor
    ~Impl() {
        close();
    }

    // Get header
    const BgenHeader& header() const {
        return header_;
    }

    // Get sample IDs
    std::vector<std::string> get_sample_ids() {
        return sample_ids_;
    }

    // Set sample filter
    void set_sample_filter(const std::vector<uint32_t>& indices) {
        sample_filter_ = indices;
        // Validate indices
        for (auto idx : sample_filter_) {
            if (idx >= header_.nsamples) {
                throw std::runtime_error(
                    "Sample index " + std::to_string(idx) +
                    " out of range (max: " + std::to_string(header_.nsamples - 1) + ")");
            }
        }
    }

    // Read variant metadata
    VariantMetadata read_variant_metadata(uint64_t offset) {
        if (!is_open_) {
            throw std::runtime_error("BGEN file is not open");
        }

        // Read variant metadata at offset
        const size_t buffer_size = 65536;  // 64KB should be enough for metadata
        std::vector<uint8_t> buffer(buffer_size);

        size_t bytes_read = file_reader_->read_at(offset, buffer.data(), buffer_size);
        if (bytes_read == 0) {
            throw std::runtime_error("Failed to read variant at offset " + std::to_string(offset));
        }

        // Parse variant metadata
        auto layout_type = static_cast<LayoutType>(header_.layout);
        auto compression_type = static_cast<CompressionType>(header_.compression);

        auto parse_result = VariantParser::parse(buffer.data(), bytes_read, layout_type,
                                                 compression_type, header_.nsamples);
        VariantMetadata metadata = parse_result.first;
        // size_t bytes_consumed = parse_result.second;  // Unused for now

        // Set the file offset
        metadata.file_offset = offset;

        // Convert relative genotype offset to absolute file offset
        // genotype_offset from parser is relative to start of variant
        uint64_t relative_offset = metadata.genotype_offset;
        metadata.genotype_offset = offset + metadata.genotype_offset;

        return metadata;
    }

    // Read multiple variant metadata in batch (optimized)
    std::vector<VariantMetadata> read_variants_metadata_batch(
        const std::vector<uint64_t>& offsets) {
        if (!is_open_) {
            throw std::runtime_error("BGEN file is not open");
        }

        if (offsets.empty()) {
            return std::vector<VariantMetadata>();
        }

        // Create pairs of (offset, original_index) and sort by offset
        std::vector<std::pair<uint64_t, size_t>> sorted_offsets;
        sorted_offsets.reserve(offsets.size());
        for (size_t i = 0; i < offsets.size(); ++i) {
            sorted_offsets.emplace_back(offsets[i], i);
        }
        std::sort(sorted_offsets.begin(), sorted_offsets.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        // Allocate single large buffer for reading (4MB)
        const size_t buffer_size = 4 * 1024 * 1024;  // 4MB buffer
        std::vector<uint8_t> buffer(buffer_size);

        // Results vector (in sorted order initially)
        std::vector<VariantMetadata> sorted_results;
        sorted_results.reserve(offsets.size());

        // Track current buffer state
        uint64_t buffer_start_offset = std::numeric_limits<uint64_t>::max();
        uint64_t buffer_end_offset = 0;
        size_t buffer_valid_bytes = 0;

        // Parse configuration
        auto layout_type = static_cast<LayoutType>(header_.layout);
        auto compression_type = static_cast<CompressionType>(header_.compression);

        // Process variants in sorted order
        for (size_t idx = 0; idx < sorted_offsets.size(); ++idx) {
            const auto& [offset, original_idx] = sorted_offsets[idx];

            // Check if we need to read more data
            if (offset < buffer_start_offset || offset >= buffer_end_offset) {
                // Advise kernel to prefetch the data we're about to read
                file_reader_->advise(offset, buffer_size, MADV_WILLNEED);

                // Also prefetch the next chunk if there are more variants
                if (idx + 1 < sorted_offsets.size()) {
                    uint64_t next_offset = sorted_offsets[idx + 1].first;
                    // Only prefetch if the next offset is beyond our current buffer
                    if (next_offset >= offset + buffer_size) {
                        file_reader_->advise(next_offset, buffer_size, MADV_WILLNEED);
                    }
                }

                // Read new chunk starting at this offset
                buffer_start_offset = offset;
                file_reader_->seek(offset);
                buffer_valid_bytes = file_reader_->read(buffer.data(), buffer_size);
                if (buffer_valid_bytes == 0) {
                    throw std::runtime_error("Failed to read variant at offset " +
                                             std::to_string(offset));
                }
                buffer_end_offset = buffer_start_offset + buffer_valid_bytes;
            }

            // Calculate position within buffer
            size_t buffer_pos = offset - buffer_start_offset;
            if (buffer_pos >= buffer_valid_bytes) {
                throw std::runtime_error("Variant offset outside valid buffer range");
            }

            // Calculate available bytes for parsing
            size_t available_bytes = buffer_valid_bytes - buffer_pos;

            // Parse variant metadata from buffer
            auto parse_result =
                VariantParser::parse(buffer.data() + buffer_pos, available_bytes, layout_type,
                                     compression_type, header_.nsamples);
            VariantMetadata metadata = parse_result.first;
            size_t bytes_consumed = parse_result.second;

            // Set the file offset
            metadata.file_offset = offset;

            // Convert relative genotype offset to absolute file offset
            metadata.genotype_offset = offset + metadata.genotype_offset;

            // Store result in sorted order
            sorted_results.push_back(std::move(metadata));

            // Check if we need to handle a variant that spans buffer boundary
            if (buffer_pos + bytes_consumed > buffer_valid_bytes &&
                buffer_end_offset < file_reader_->size()) {
                // This variant spans the buffer boundary, need to re-read with larger context
                // This is rare but can happen with very large variant metadata
                std::vector<uint8_t> temp_buffer(bytes_consumed + 1024);  // Add some extra space
                size_t temp_bytes =
                    file_reader_->read_at(offset, temp_buffer.data(), temp_buffer.size());
                if (temp_bytes < bytes_consumed) {
                    throw std::runtime_error("Failed to read complete variant metadata at offset " +
                                             std::to_string(offset));
                }

                // Re-parse with complete data
                auto reparse_result =
                    VariantParser::parse(temp_buffer.data(), temp_bytes, layout_type,
                                         compression_type, header_.nsamples);
                sorted_results.back() = reparse_result.first;
                sorted_results.back().file_offset = offset;
                sorted_results.back().genotype_offset =
                    offset + reparse_result.first.genotype_offset;
            }
        }

        // Reorder results to match original input order
        std::vector<VariantMetadata> results(offsets.size());
        for (size_t i = 0; i < sorted_offsets.size(); ++i) {
            results[sorted_offsets[i].second] = std::move(sorted_results[i]);
        }

        return results;
    }

    // Read variant genotypes
    std::unique_ptr<decompress::DecompressedData> read_variant_genotypes(
        const VariantMetadata& metadata) {
        if (!is_open_) {
            throw std::runtime_error("BGEN file is not open");
        }

        // Read compressed genotype data
        std::vector<uint8_t> compressed_data(metadata.genotype_length);

        size_t bytes_read = file_reader_->read_at(metadata.genotype_offset, compressed_data.data(),
                                                  metadata.genotype_length);

        if (bytes_read != metadata.genotype_length) {
            throw std::runtime_error("Failed to read complete genotype data for variant " +
                                     metadata.varid);
        }

        const uint8_t* data_ptr = compressed_data.data();
        size_t compressed_size = metadata.genotype_length;
        size_t uncompressed_size = 0;

        // Create compressed variant struct (need this early for debug output)
        auto comp_type = static_cast<decompress::CompressionType>(header_.compression);

        if (header_.layout == 2) {  // v1.2
            if (comp_type != decompress::CompressionType::None) {
                // For v1.2, genotype_offset points to after the C field
                // So the data starts with D field (uncompressed length) followed by compressed data
                if (compressed_size < 4) {
                    throw std::runtime_error(
                        "Invalid compressed genotype data size for v1.2 variant");
                }
                // Read uncompressed size (little-endian) - this is the D field
                uncompressed_size =
                    data_ptr[0] | (data_ptr[1] << 8) | (data_ptr[2] << 16) | (data_ptr[3] << 24);

                // Sanity check: uncompressed size should be reasonable
                // For biallelic variants with 8-bit precision, expect ~3 bytes per sample
                size_t max_expected_size =
                    header_.nsamples * 10 + 1000;  // Very generous upper bound
                if (uncompressed_size > max_expected_size || uncompressed_size == 0) {
                    throw std::runtime_error("Invalid uncompressed size in BGEN file: " +
                                             std::to_string(uncompressed_size) +
                                             " bytes. "
                                             "Expected at most " +
                                             std::to_string(max_expected_size) + " bytes for " +
                                             std::to_string(header_.nsamples) + " samples.");
                }

                data_ptr += 4;
                compressed_size -= 4;
            } else {
                // Uncompressed data is not supported
                throw std::runtime_error(
                    "Uncompressed BGEN files are not supported. "
                    "Please use compressed BGEN files (zlib or zstd). "
                    "You can compress your BGEN file using bgenix or qctool2.");
            }
        } else {  // v1.1
            if (comp_type == decompress::CompressionType::None) {
                // Uncompressed data is not supported
                throw std::runtime_error(
                    "Uncompressed BGEN files are not supported. "
                    "Please use compressed BGEN files (zlib or zstd). "
                    "You can compress your BGEN file using bgenix or qctool2.");
            }
            // For v1.1, estimate uncompressed size
            uncompressed_size = header_.nsamples * 6;  // 3 * 2 bytes per sample
        }

        // Create compressed variant for decompression

        decompress::CompressedVariant comp_variant(metadata.file_offset, data_ptr, compressed_size,
                                                   uncompressed_size, comp_type);
        comp_variant.variant_id = metadata.varid;

        // Decompress

        auto result = decompressor_->decompress(comp_variant);
        return std::unique_ptr<decompress::DecompressedData>(
            new decompress::DecompressedData(std::move(result)));
    }

    // Read multiple variants in batch
    std::vector<std::unique_ptr<decompress::DecompressedData>> read_variants_batch(
        const std::vector<VariantMetadata>& variants) {
        if (!is_open_) {
            throw std::runtime_error("BGEN file is not open");
        }

        std::vector<decompress::CompressedVariant> compressed_variants;
        compressed_variants.reserve(variants.size());

        // Phase 1: Calculate total buffer size needed
        size_t total_buffer_size = 0;
        for (const auto& metadata : variants) {
            total_buffer_size += metadata.genotype_length;
        }

        // Phase 2: Allocate single large buffer for all compressed data
        auto consolidated_buffer =
            std::unique_ptr<std::vector<uint8_t>>(new std::vector<uint8_t>(total_buffer_size));

        // Phase 2.5: Advise kernel to prefetch all the regions we're about to read
        // This is done in batches to avoid too many madvise calls
        const size_t prefetch_batch_size = 32;  // Prefetch up to 32 regions at a time
        for (size_t i = 0; i < variants.size(); i += prefetch_batch_size) {
            size_t batch_end = std::min(i + prefetch_batch_size, variants.size());

            // Calculate the range for this batch of variants
            uint64_t batch_start_offset = variants[i].genotype_offset;
            uint64_t batch_end_offset =
                variants[batch_end - 1].genotype_offset + variants[batch_end - 1].genotype_length;
            size_t batch_size = batch_end_offset - batch_start_offset;

            // Advise kernel to prefetch this batch
            file_reader_->advise(batch_start_offset, batch_size, MADV_WILLNEED);
        }

        // Phase 3: Read all variant data into the consolidated buffer
        size_t current_offset = 0;
        for (const auto& metadata : variants) {
            // Read compressed genotype data directly into the consolidated buffer
            uint8_t* dest_ptr = consolidated_buffer->data() + current_offset;
            size_t bytes_read =
                file_reader_->read_at(metadata.genotype_offset, dest_ptr, metadata.genotype_length);

            if (bytes_read != metadata.genotype_length) {
                throw std::runtime_error("Failed to read genotype data for variant " +
                                         metadata.varid);
            }

            // Handle v1.2 format
            const uint8_t* data_ptr = dest_ptr;
            size_t compressed_size = metadata.genotype_length;
            size_t uncompressed_size = 0;

            if (header_.layout == 2) {  // v1.2
                // For v1.2, genotype_offset points to after the C field
                // So the data starts with D field (uncompressed length) followed by compressed data
                if (compressed_size < 4) {
                    throw std::runtime_error("Invalid genotype data size for v1.2 variant");
                }

                // Read uncompressed size (D field) - data already starts after C field
                uncompressed_size =
                    data_ptr[0] | (data_ptr[1] << 8) | (data_ptr[2] << 16) | (data_ptr[3] << 24);

                // Sanity check
                size_t max_expected_size = header_.nsamples * 10 + 1000;
                if (uncompressed_size > max_expected_size || uncompressed_size == 0) {
                    throw std::runtime_error(
                        "Invalid uncompressed size: " + std::to_string(uncompressed_size) +
                        " bytes. "
                        "Expected at most " +
                        std::to_string(max_expected_size) + " bytes for " +
                        std::to_string(header_.nsamples) + " samples.");
                }

                data_ptr += 4;
                compressed_size -= 4;
            } else {  // v1.1
                uncompressed_size = header_.nsamples * 6;
            }

            // Create compressed variant
            auto comp_type = static_cast<decompress::CompressionType>(header_.compression);

            // Check for unsupported uncompressed data
            if (comp_type == decompress::CompressionType::None) {
                throw std::runtime_error(
                    "Uncompressed BGEN files are not supported. "
                    "Please use compressed BGEN files (zlib or zstd). "
                    "You can compress your BGEN file using bgenix or qctool2.");
            }

            compressed_variants.emplace_back(metadata.file_offset, data_ptr, compressed_size,
                                             uncompressed_size, comp_type);
            compressed_variants.back().variant_id = metadata.varid;

            // Update offset for next variant
            current_offset += metadata.genotype_length;
        }

        // Store the consolidated buffer (replace the old vector approach)
        compressed_buffers_.clear();
        compressed_buffers_.push_back(std::move(consolidated_buffer));

        // Decompress in batch
        auto results = decompressor_->decompress_batch(compressed_variants);

        // Clear temporary buffers
        compressed_buffers_.clear();

        // Convert to unique_ptr vector
        std::vector<std::unique_ptr<decompress::DecompressedData>> ptr_results;
        ptr_results.reserve(results.size());
        for (auto& result : results) {
            ptr_results.push_back(std::unique_ptr<decompress::DecompressedData>(
                new decompress::DecompressedData(std::move(result))));
        }
        return ptr_results;
    }

    // Set decompressor type
    void set_decompressor_type(const std::string& type) {
        decompress::VariantDecompressor::Config config;
        config.auto_detect_compression = true;
        config.validate_size = true;

        if (type == "sequential") {
            decompressor_ =
                decompress::create_sequential_decompressor(file_reader_.get(), config, true);
        } else if (type == "parallel") {
            decompressor_ =
                decompress::create_parallel_decompressor(file_reader_.get(), num_threads_, config);
        } else if (type == "adaptive") {
            decompressor_ = decompress::create_adaptive_decompressor(file_reader_.get(), config);
        } else {
            throw std::runtime_error("Unknown decompressor type: " + type);
        }
    }

    // Set number of threads
    void set_num_threads(size_t n) {
        num_threads_ = n;
        // If using parallel decompressor, recreate it
        if (decompressor_type_ == "parallel") {
            set_decompressor_type("parallel");
        }
    }

    // Check if open
    bool is_open() const {
        return is_open_;
    }

    // Close file
    void close() {
        if (file_reader_) {
            file_reader_->close();
            file_reader_.reset();
        }
        if (bgi_reader_) {
            bgi_reader_.reset();
        }
        decompressor_.reset();
        compressed_buffers_.clear();
        is_open_ = false;
    }

   private:
    // Open the BGEN file
    void openFile() {
        // Create appropriate reader based on filename
        if (filename_.substr(0, 5) == "gs://") {
            // Use GCS file reader
            file_reader_ = std::unique_ptr<GCSFileReader>(new GCSFileReader(filename_));
        } else {
            // Try memory-mapped file first for local files
            try {
                file_reader_ = std::unique_ptr<MMapFileReader>(new MMapFileReader(filename_));
                if (file_reader_->is_open()) {
                    return;
                }
            } catch (...) {
                // Fall back to regular file reader
            }

            // Use regular file reader
            file_reader_ = std::unique_ptr<RegularFileReader>(new RegularFileReader(filename_));
        }

        if (!file_reader_ || !file_reader_->is_open()) {
            throw std::runtime_error("Failed to open BGEN file: " + filename_);
        }
    }

    // Parse BGEN header
    void parseHeader() {
        // Read header size first (need at least 20 bytes for initial fields)
        std::vector<uint8_t> initial_buffer(20);
        size_t bytes_read = file_reader_->read(initial_buffer.data(), 20);
        if (bytes_read < 20) {
            throw std::runtime_error("BGEN file too small to contain valid header");
        }

        // Get full header size
        size_t header_size = BgenHeaderParser::getHeaderSize(initial_buffer.data(), bytes_read);

        // Read full header
        std::vector<uint8_t> header_buffer(header_size);
        file_reader_->seek(0);
        bytes_read = file_reader_->read(header_buffer.data(), header_size);
        if (bytes_read < header_size) {
            throw std::runtime_error("Failed to read complete BGEN header");
        }

        // Parse header
        auto parsed_header = BgenHeaderParser::parse(header_buffer.data(), bytes_read);

        // Convert to our header structure
        header_.offset = parsed_header.offset;
        header_.nvariants = parsed_header.nvariants;
        header_.nsamples = parsed_header.nsamples;
        header_.flags = parsed_header.flags;
        header_.compression = static_cast<uint8_t>(parsed_header.compression);
        header_.layout = static_cast<uint8_t>(parsed_header.layout);
        header_.has_sample_ids = parsed_header.has_sample_ids;
    }

    // Open BGI index
    void openIndex() {
        try {
            // BGI path should already be local (Python handles GCS download)
            bgi_reader_ = std::unique_ptr<BGIReader>(new BGIReader(bgi_filename_));
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to open BGI index: " + std::string(e.what()));
        }
    }

    // Read sample IDs
    void readSampleIds() {
        if (header_.has_sample_ids) {
            // Current position should be right after header
            uint64_t sample_block_pos = file_reader_->tell();

            // Read sample block size
            uint8_t size_buffer[4];
            size_t bytes_read = file_reader_->read(size_buffer, 4);
            if (bytes_read < 4) {
                throw std::runtime_error("Failed to read sample block size");
            }

            // Convert from little-endian
            uint32_t block_size_with_n = size_buffer[0] | (size_buffer[1] << 8) |
                                         (size_buffer[2] << 16) | (size_buffer[3] << 24);

            // Sanity check the block size (should include sample count)
            if (block_size_with_n < 4 || block_size_with_n > 100000000) {  // 100MB max
                throw std::runtime_error("Invalid sample block size: " +
                                         std::to_string(block_size_with_n));
            }

            // Read full sample block (including the 4-byte size prefix)
            std::vector<uint8_t> sample_buffer(block_size_with_n + 4);
            file_reader_->seek(sample_block_pos);
            bytes_read = file_reader_->read(sample_buffer.data(), block_size_with_n + 4);
            if (bytes_read < block_size_with_n + 4) {
                throw std::runtime_error("Failed to read sample block");
            }

            // Parse sample block
            auto sample_block =
                SampleBlockParser::parse(sample_buffer.data(), bytes_read, header_.nsamples);
            sample_ids_ = std::move(sample_block.sample_ids);

            // Update offset to skip sample block
            header_.offset = sample_block_pos + block_size_with_n + 4;
        } else {
            // Generate default sample IDs
            sample_ids_.reserve(header_.nsamples);
            for (uint32_t i = 0; i < header_.nsamples; ++i) {
                sample_ids_.push_back("sample_" + std::to_string(i));
            }
        }
    }

    // Create default decompressor
    void createDefaultDecompressor() {
        decompress::VariantDecompressor::Config config;
        config.auto_detect_compression = true;
        config.validate_size = true;

        // Use adaptive decompressor by default
        decompressor_ = decompress::create_adaptive_decompressor(file_reader_.get(), config);
        decompressor_type_ = "adaptive";
    }

   private:
    std::string filename_;
    std::string bgi_filename_;
    std::unique_ptr<FileReader> file_reader_;
    std::unique_ptr<BGIReader> bgi_reader_;
    std::unique_ptr<decompress::VariantDecompressor> decompressor_;
    BgenHeader header_;
    std::vector<std::string> sample_ids_;
    std::vector<uint32_t> sample_filter_;
    bool is_open_;
    size_t num_threads_ = 0;
    std::string decompressor_type_;

    // Temporary storage for compressed data during batch operations
    std::vector<std::unique_ptr<std::vector<uint8_t>>> compressed_buffers_;
};

// BgenReaderImpl public methods (forwarding to pimpl)

BgenReaderImpl::BgenReaderImpl(const std::string& filename, const std::string& bgi_filename)
    : pimpl_(std::unique_ptr<Impl>(new Impl(filename, bgi_filename))) {}

BgenReaderImpl::~BgenReaderImpl() = default;

const BgenHeader& BgenReaderImpl::header() const {
    return pimpl_->header();
}

std::vector<std::string> BgenReaderImpl::get_sample_ids() {
    return pimpl_->get_sample_ids();
}

void BgenReaderImpl::set_sample_filter(const std::vector<uint32_t>& indices) {
    pimpl_->set_sample_filter(indices);
}

VariantMetadata BgenReaderImpl::read_variant_metadata(uint64_t offset) {
    return pimpl_->read_variant_metadata(offset);
}

std::vector<VariantMetadata> BgenReaderImpl::read_variants_metadata_batch(
    const std::vector<uint64_t>& offsets) {
    return pimpl_->read_variants_metadata_batch(offsets);
}

std::unique_ptr<decompress::DecompressedData> BgenReaderImpl::read_variant_genotypes(
    const VariantMetadata& metadata) {
    return pimpl_->read_variant_genotypes(metadata);
}

std::vector<std::unique_ptr<decompress::DecompressedData>> BgenReaderImpl::read_variants_batch(
    const std::vector<VariantMetadata>& variants) {
    return pimpl_->read_variants_batch(variants);
}

void BgenReaderImpl::set_decompressor_type(const std::string& type) {
    pimpl_->set_decompressor_type(type);
}

void BgenReaderImpl::set_num_threads(size_t n) {
    pimpl_->set_num_threads(n);
}

bool BgenReaderImpl::is_open() const {
    return pimpl_->is_open();
}

void BgenReaderImpl::close() {
    pimpl_->close();
}

}  // namespace bgen
}  // namespace io
}  // namespace ldcov