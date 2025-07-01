#include "bgi_reader.h"

#include <sqlite3.h>

#include <algorithm>
#include <cstring>
#include <list>
#include <set>
#include <sstream>
#include <stdexcept>

// C++17 feature detection for shared_mutex
#if __cplusplus >= 201703L
#include <shared_mutex>
#endif

namespace ldcov {
namespace io {
namespace bgen {
namespace index {

// Implementation class
class BGIReader::Impl {
   private:
// C++17 feature detection for lock types
#if __cplusplus >= 201703L
    using mutex_type = std::shared_mutex;
    using read_lock = std::shared_lock<std::shared_mutex>;
    using write_lock = std::unique_lock<std::shared_mutex>;
#else
    using mutex_type = std::mutex;
    using read_lock = std::lock_guard<std::mutex>;
    using write_lock = std::lock_guard<std::mutex>;
#endif

   public:
    Impl(const std::string& bgi_path) : db_(nullptr), variant_count_(0) {
        // Open SQLite database (removed NOMUTEX for better internal optimization)
        int rc = sqlite3_open_v2(bgi_path.c_str(), &db_, SQLITE_OPEN_READONLY, nullptr);
        if (rc != SQLITE_OK) {
            throw std::runtime_error("Failed to open BGI file: " + bgi_path + " - " +
                                     sqlite3_errmsg(db_));
        }

        // Configure SQLite for optimal performance
        sqlite3_exec(db_, "PRAGMA cache_size = 10000", nullptr, nullptr, nullptr);
        sqlite3_exec(db_, "PRAGMA mmap_size = 268435456", nullptr, nullptr, nullptr);  // 256MB
        sqlite3_exec(db_, "PRAGMA temp_store = MEMORY", nullptr, nullptr, nullptr);

        // Verify database structure and load metadata
        verify_database();
        load_metadata();

        // Prepare frequently used statements
        prepare_statements();
    }

    ~Impl() {
        // Finalize prepared statements
        if (stmt_by_region_)
            sqlite3_finalize(stmt_by_region_);
        if (stmt_by_position_)
            sqlite3_finalize(stmt_by_position_);
        if (stmt_by_rsid_)
            sqlite3_finalize(stmt_by_rsid_);
        if (stmt_by_varid_)
            sqlite3_finalize(stmt_by_varid_);
        if (stmt_by_index_)
            sqlite3_finalize(stmt_by_index_);
        if (stmt_offsets_range_)
            sqlite3_finalize(stmt_offsets_range_);
        if (stmt_chromosomes_)
            sqlite3_finalize(stmt_chromosomes_);

        // Close database
        if (db_)
            sqlite3_close(db_);
    }

    std::vector<VariantInfo> query_region(const std::string& chromosome, uint32_t start_pos,
                                          uint32_t end_pos) {
        read_lock lock(mutex_);

        // Query database directly without cache
        sqlite3_reset(stmt_by_region_);
        sqlite3_bind_text(stmt_by_region_, 1, chromosome.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_int(stmt_by_region_, 2, start_pos);
        sqlite3_bind_int(stmt_by_region_, 3, end_pos);

        return execute_query(stmt_by_region_);
    }

    std::vector<VariantInfo> query_position(const std::string& chromosome, uint32_t position) {
        read_lock lock(mutex_);

        sqlite3_reset(stmt_by_position_);
        sqlite3_bind_text(stmt_by_position_, 1, chromosome.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_int(stmt_by_position_, 2, position);

        return execute_query(stmt_by_position_);
    }

    std::vector<VariantInfo> query_variant_id(const std::string& variant_id) {
        read_lock lock(mutex_);

        // Try rsid first
        sqlite3_reset(stmt_by_rsid_);
        sqlite3_bind_text(stmt_by_rsid_, 1, variant_id.c_str(), -1, SQLITE_STATIC);
        auto results = execute_query(stmt_by_rsid_);

        // If not found, try varid
        if (results.empty()) {
            sqlite3_reset(stmt_by_varid_);
            sqlite3_bind_text(stmt_by_varid_, 1, variant_id.c_str(), -1, SQLITE_STATIC);
            results = execute_query(stmt_by_varid_);
        }

        return std::move(results);
    }

    VariantInfo get_variant(size_t index) {
        read_lock lock(mutex_);

        if (index >= variant_count_) {
            throw std::out_of_range("Variant index out of range: " + std::to_string(index));
        }

        sqlite3_reset(stmt_by_index_);
        sqlite3_bind_int64(stmt_by_index_, 1, static_cast<sqlite3_int64>(index));

        auto results = execute_query(stmt_by_index_);
        if (results.empty()) {
            throw std::runtime_error("Failed to get variant at index: " + std::to_string(index));
        }

        return results[0];
    }

    std::vector<uint64_t> get_file_offsets(size_t start_idx, size_t end_idx) {
        read_lock lock(mutex_);

        if (start_idx >= variant_count_ || end_idx > variant_count_ || start_idx >= end_idx) {
            throw std::out_of_range("Invalid index range");
        }

        std::vector<uint64_t> offsets;
        offsets.reserve(end_idx - start_idx);

        sqlite3_reset(stmt_offsets_range_);

        if (use_offset_for_range_) {
            // Using LIMIT/OFFSET approach
            sqlite3_bind_int64(stmt_offsets_range_, 1,
                               static_cast<sqlite3_int64>(end_idx - start_idx));
            sqlite3_bind_int64(stmt_offsets_range_, 2, static_cast<sqlite3_int64>(start_idx));
        } else {
            // Using ROWID approach (1-based)
            sqlite3_bind_int64(stmt_offsets_range_, 1, static_cast<sqlite3_int64>(start_idx));
            sqlite3_bind_int64(stmt_offsets_range_, 2, static_cast<sqlite3_int64>(end_idx));
        }

        int rc;
        while ((rc = sqlite3_step(stmt_offsets_range_)) == SQLITE_ROW) {
            offsets.push_back(static_cast<uint64_t>(sqlite3_column_int64(stmt_offsets_range_, 0)));
        }

        if (rc != SQLITE_DONE) {
            throw std::runtime_error("Failed to query file offsets: " +
                                     std::string(sqlite3_errmsg(db_)));
        }

        return std::move(offsets);
    }

    size_t get_variant_count() const {
        return variant_count_;
    }

    std::vector<std::string> get_chromosomes() const {
        read_lock lock(mutex_);

        std::vector<std::string> chromosomes;
        sqlite3_reset(stmt_chromosomes_);

        int rc;
        while ((rc = sqlite3_step(stmt_chromosomes_)) == SQLITE_ROW) {
            const char* chr =
                reinterpret_cast<const char*>(sqlite3_column_text(stmt_chromosomes_, 0));
            if (chr) {
                chromosomes.push_back(chr);
            }
        }

        if (rc != SQLITE_DONE) {
            throw std::runtime_error("Failed to query chromosomes: " +
                                     std::string(sqlite3_errmsg(db_)));
        }

        return std::move(chromosomes);
    }

    bool is_valid() const {
        return db_ != nullptr && variant_count_ > 0;
    }

    std::unordered_map<std::string, std::string> get_metadata() const {
        return metadata_;
    }

    std::vector<VariantInfo> get_all_variants() {
        read_lock lock(mutex_);

        std::vector<VariantInfo> results;
        results.reserve(variant_count_);

        // Query all variants in one go
        const char* query =
            "SELECT file_start_position, size_in_bytes, chromosome, position, "
            "rsid, number_of_alleles, allele1, allele2 "
            "FROM Variant "
            "ORDER BY file_start_position";

        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_, query, -1, &stmt, nullptr) != SQLITE_OK) {
            throw std::runtime_error("Failed to prepare all variants query");
        }

        results = execute_query(stmt);
        sqlite3_finalize(stmt);

        return std::move(results);
    }

    std::vector<VariantInfo> find_variants_by_filter(const std::string& chromosome,
                                                     const std::vector<uint32_t>& positions,
                                                     const std::vector<std::string>& alleles1,
                                                     const std::vector<std::string>& alleles2,
                                                     size_t batch_size) {
        read_lock lock(mutex_);

        if (positions.size() != alleles1.size() || positions.size() != alleles2.size()) {
            throw std::invalid_argument("Input vectors must have same size");
        }

        if (positions.empty()) {
            return std::vector<VariantInfo>();
        }

        // Build lookup table for allele matching with original indices
        struct AlleleMatch {
            std::string allele1;
            std::string allele2;
            size_t original_index;
        };
        std::unordered_map<uint32_t, std::vector<AlleleMatch>> allele_map;
        for (size_t i = 0; i < positions.size(); ++i) {
            allele_map[positions[i]].push_back({alleles1[i], alleles2[i], i});
        }

        // Result map to maintain original order
        std::unordered_map<size_t, VariantInfo> result_map;

        // Get unique positions for efficient querying
        std::set<uint32_t> unique_positions(positions.begin(), positions.end());
        std::vector<uint32_t> unique_pos_vec(unique_positions.begin(), unique_positions.end());

        // Process in batches
        for (size_t i = 0; i < unique_pos_vec.size(); i += batch_size) {
            size_t batch_end = std::min(i + batch_size, unique_pos_vec.size());
            size_t batch_size_actual = batch_end - i;

            // Build query with IN clause
            std::stringstream query;
            query << "SELECT file_start_position, size_in_bytes, chromosome, position, "
                  << "rsid, number_of_alleles, allele1, allele2 "
                  << "FROM Variant WHERE chromosome = ? AND position IN (";

            for (size_t j = 0; j < batch_size_actual; ++j) {
                if (j > 0)
                    query << ",";
                query << "?";
            }
            query << ") ORDER BY file_start_position";

            // Prepare statement
            sqlite3_stmt* stmt;
            if (sqlite3_prepare_v2(db_, query.str().c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
                throw std::runtime_error("Failed to prepare batch query: " +
                                         std::string(sqlite3_errmsg(db_)));
            }

            // Bind parameters
            sqlite3_bind_text(stmt, 1, chromosome.c_str(), -1, SQLITE_STATIC);
            for (size_t j = 0; j < batch_size_actual; ++j) {
                sqlite3_bind_int(stmt, 2 + j, unique_pos_vec[i + j]);
            }

            // Execute and collect results
            int rc;
            while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
                VariantInfo info;

                // Extract data from columns
                info.file_offset = static_cast<uint64_t>(sqlite3_column_int64(stmt, 0));
                info.variant_size = static_cast<uint32_t>(sqlite3_column_int(stmt, 1));

                const char* chr = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
                if (chr)
                    info.chromosome = chr;

                info.position = static_cast<uint32_t>(sqlite3_column_int(stmt, 3));

                const char* rsid = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
                if (rsid)
                    info.rsid = rsid;

                info.n_alleles = static_cast<uint16_t>(sqlite3_column_int(stmt, 5));

                const char* a1 = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 6));
                if (a1)
                    info.allele1 = a1;

                const char* a2 = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 7));
                if (a2)
                    info.allele2 = a2;
                else
                    info.allele2 = "";  // Handle NULL as empty string

                // Check if this variant matches any of our allele combinations
                auto it = allele_map.find(info.position);
                if (it != allele_map.end()) {
                    for (const auto& match : it->second) {
                        if (info.allele1 == match.allele1 && info.allele2 == match.allele2) {
                            result_map[match.original_index] = info;
                        }
                    }
                }
            }

            if (rc != SQLITE_DONE) {
                sqlite3_finalize(stmt);
                throw std::runtime_error("Batch query execution failed: " +
                                         std::string(sqlite3_errmsg(db_)));
            }

            sqlite3_finalize(stmt);
        }

        // Build results in original order
        std::vector<VariantInfo> results;
        results.reserve(result_map.size());

        for (size_t i = 0; i < positions.size(); ++i) {
            auto it = result_map.find(i);
            if (it != result_map.end()) {
                results.push_back(it->second);
            }
        }

        return std::move(results);
    }

    std::vector<VariantInfo> query_positions_batch(const std::string& chromosome,
                                                   const std::vector<uint32_t>& positions,
                                                   size_t batch_size) {
        read_lock lock(mutex_);

        if (positions.empty()) {
            return std::vector<VariantInfo>();
        }

        std::vector<VariantInfo> all_results;
        all_results.reserve(positions.size() *
                            2);  // Reserve extra space for multiple variants per position

        // Get unique positions
        std::set<uint32_t> unique_positions(positions.begin(), positions.end());
        std::vector<uint32_t> unique_pos_vec(unique_positions.begin(), unique_positions.end());

        // Process in batches
        for (size_t i = 0; i < unique_pos_vec.size(); i += batch_size) {
            size_t batch_end = std::min(i + batch_size, unique_pos_vec.size());
            size_t batch_size_actual = batch_end - i;

            // Build query
            std::stringstream query;
            query << "SELECT file_start_position, size_in_bytes, chromosome, position, "
                  << "rsid, number_of_alleles, allele1, allele2 "
                  << "FROM Variant WHERE chromosome = ? AND position IN (";

            for (size_t j = 0; j < batch_size_actual; ++j) {
                if (j > 0)
                    query << ",";
                query << "?";
            }
            query << ") ORDER BY position, file_start_position";

            // Prepare and execute
            sqlite3_stmt* stmt;
            if (sqlite3_prepare_v2(db_, query.str().c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
                throw std::runtime_error("Failed to prepare batch position query");
            }

            // Bind parameters
            sqlite3_bind_text(stmt, 1, chromosome.c_str(), -1, SQLITE_STATIC);
            for (size_t j = 0; j < batch_size_actual; ++j) {
                sqlite3_bind_int(stmt, 2 + j, unique_pos_vec[i + j]);
            }

            // Execute
            auto batch_results = execute_query(stmt);
            sqlite3_finalize(stmt);

            all_results.insert(all_results.end(), batch_results.begin(), batch_results.end());
        }

        return std::move(all_results);
    }

    std::vector<VariantInfo> query_variant_ids_batch(const std::vector<std::string>& variant_ids,
                                                     size_t batch_size) {
        read_lock lock(mutex_);

        if (variant_ids.empty()) {
            return std::vector<VariantInfo>();
        }

        std::vector<VariantInfo> all_results;
        all_results.reserve(variant_ids.size());

        // Get unique IDs
        std::set<std::string> unique_ids(variant_ids.begin(), variant_ids.end());
        std::vector<std::string> unique_ids_vec(unique_ids.begin(), unique_ids.end());

        // Process in batches
        for (size_t i = 0; i < unique_ids_vec.size(); i += batch_size) {
            size_t batch_end = std::min(i + batch_size, unique_ids_vec.size());
            size_t batch_size_actual = batch_end - i;

            // Try rsid first
            std::stringstream query;
            query << "SELECT file_start_position, size_in_bytes, chromosome, position, "
                  << "rsid, number_of_alleles, allele1, allele2 " << "FROM Variant WHERE rsid IN (";

            for (size_t j = 0; j < batch_size_actual; ++j) {
                if (j > 0)
                    query << ",";
                query << "?";
            }
            query << ")";

            // Also check varid if column exists
            if (stmt_by_varid_ != nullptr) {
                query << " UNION "
                      << "SELECT file_start_position, size_in_bytes, chromosome, position, "
                      << "rsid, number_of_alleles, allele1, allele2 "
                      << "FROM Variant WHERE variant_id IN (";

                for (size_t j = 0; j < batch_size_actual; ++j) {
                    if (j > 0)
                        query << ",";
                    query << "?";
                }
                query << ")";
            }

            query << " ORDER BY file_start_position";

            // Prepare and execute
            sqlite3_stmt* stmt;
            if (sqlite3_prepare_v2(db_, query.str().c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
                throw std::runtime_error("Failed to prepare batch ID query");
            }

            // Bind parameters
            int param_idx = 1;
            for (size_t j = 0; j < batch_size_actual; ++j) {
                sqlite3_bind_text(stmt, param_idx++, unique_ids_vec[i + j].c_str(), -1,
                                  SQLITE_STATIC);
            }

            // Bind again for varid if needed
            if (stmt_by_varid_ != nullptr) {
                for (size_t j = 0; j < batch_size_actual; ++j) {
                    sqlite3_bind_text(stmt, param_idx++, unique_ids_vec[i + j].c_str(), -1,
                                      SQLITE_STATIC);
                }
            }

            // Execute
            auto batch_results = execute_query(stmt);
            sqlite3_finalize(stmt);

            all_results.insert(all_results.end(), batch_results.begin(), batch_results.end());
        }

        return std::move(all_results);
    }

   private:
    void verify_database() {
        // Check if required tables exist
        const char* check_tables =
            "SELECT COUNT(*) FROM sqlite_master "
            "WHERE type='table' AND name IN ('Variant', 'Metadata')";

        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_, check_tables, -1, &stmt, nullptr) != SQLITE_OK) {
            throw std::runtime_error("Failed to prepare table check query");
        }

        int table_count = 0;
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            table_count = sqlite3_column_int(stmt, 0);
        }
        sqlite3_finalize(stmt);

        if (table_count < 1) {  // At least Variant table must exist
            throw std::runtime_error("Invalid BGI file: missing required tables");
        }

        // Get variant count
        const char* count_query = "SELECT COUNT(*) FROM Variant";
        if (sqlite3_prepare_v2(db_, count_query, -1, &stmt, nullptr) != SQLITE_OK) {
            throw std::runtime_error("Failed to prepare count query");
        }

        if (sqlite3_step(stmt) == SQLITE_ROW) {
            variant_count_ = static_cast<size_t>(sqlite3_column_int64(stmt, 0));
        }
        sqlite3_finalize(stmt);
    }

    void load_metadata() {
        // Check if Metadata table exists
        const char* check_metadata =
            "SELECT COUNT(*) FROM sqlite_master "
            "WHERE type='table' AND name='Metadata'";

        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_, check_metadata, -1, &stmt, nullptr) != SQLITE_OK) {
            return;  // No metadata table
        }

        int exists = 0;
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            exists = sqlite3_column_int(stmt, 0);
        }
        sqlite3_finalize(stmt);

        if (!exists)
            return;

        // Load metadata
        const char* metadata_query = "SELECT key, value FROM Metadata";
        if (sqlite3_prepare_v2(db_, metadata_query, -1, &stmt, nullptr) != SQLITE_OK) {
            return;
        }

        while (sqlite3_step(stmt) == SQLITE_ROW) {
            const char* key = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
            const char* value = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            if (key && value) {
                metadata_[key] = value;
            }
        }
        sqlite3_finalize(stmt);
    }

    void prepare_statements() {
        // Query by region
        const char* region_query =
            "SELECT file_start_position, size_in_bytes, chromosome, position, "
            "rsid, number_of_alleles, allele1, allele2 "
            "FROM Variant "
            "WHERE chromosome = ? AND position >= ? AND position <= ? "
            "ORDER BY position";

        if (sqlite3_prepare_v2(db_, region_query, -1, &stmt_by_region_, nullptr) != SQLITE_OK) {
            throw std::runtime_error("Failed to prepare region query");
        }

        // Query by position
        const char* position_query =
            "SELECT file_start_position, size_in_bytes, chromosome, position, "
            "rsid, number_of_alleles, allele1, allele2 "
            "FROM Variant "
            "WHERE chromosome = ? AND position = ?";

        if (sqlite3_prepare_v2(db_, position_query, -1, &stmt_by_position_, nullptr) != SQLITE_OK) {
            throw std::runtime_error("Failed to prepare position query");
        }

        // Query by rsid
        const char* rsid_query =
            "SELECT file_start_position, size_in_bytes, chromosome, position, "
            "rsid, number_of_alleles, allele1, allele2 "
            "FROM Variant "
            "WHERE rsid = ?";

        if (sqlite3_prepare_v2(db_, rsid_query, -1, &stmt_by_rsid_, nullptr) != SQLITE_OK) {
            throw std::runtime_error("Failed to prepare rsid query");
        }

        // Query by varid (if column exists)
        const char* varid_query =
            "SELECT file_start_position, size_in_bytes, chromosome, position, "
            "rsid, number_of_alleles, allele1, allele2 "
            "FROM Variant "
            "WHERE variant_id = ?";

        // This might fail if variant_id column doesn't exist, which is OK
        sqlite3_prepare_v2(db_, varid_query, -1, &stmt_by_varid_, nullptr);

        // Query by index - use ROWID for efficiency instead of OFFSET
        const char* index_query =
            "SELECT file_start_position, size_in_bytes, chromosome, position, "
            "rsid, number_of_alleles, allele1, allele2 "
            "FROM Variant "
            "WHERE ROWID = ? + 1";  // SQLite ROWID is 1-based

        if (sqlite3_prepare_v2(db_, index_query, -1, &stmt_by_index_, nullptr) != SQLITE_OK) {
            // Fallback to OFFSET if ROWID approach fails
            const char* offset_query =
                "SELECT file_start_position, size_in_bytes, chromosome, position, "
                "rsid, number_of_alleles, allele1, allele2 "
                "FROM Variant "
                "ORDER BY file_start_position "
                "LIMIT 1 OFFSET ?";

            if (sqlite3_prepare_v2(db_, offset_query, -1, &stmt_by_index_, nullptr) != SQLITE_OK) {
                throw std::runtime_error("Failed to prepare index query");
            }
        }

        // Query file offsets range - use ROWID for efficiency
        const char* offsets_query =
            "SELECT file_start_position "
            "FROM Variant "
            "WHERE ROWID > ? AND ROWID <= ?";

        if (sqlite3_prepare_v2(db_, offsets_query, -1, &stmt_offsets_range_, nullptr) !=
            SQLITE_OK) {
            // Fallback to OFFSET approach
            const char* offset_query =
                "SELECT file_start_position "
                "FROM Variant "
                "ORDER BY file_start_position "
                "LIMIT ? OFFSET ?";

            if (sqlite3_prepare_v2(db_, offset_query, -1, &stmt_offsets_range_, nullptr) !=
                SQLITE_OK) {
                throw std::runtime_error("Failed to prepare offsets query");
            }

            // Set flag to indicate we're using the offset approach
            use_offset_for_range_ = true;
        }

        // Query distinct chromosomes
        const char* chr_query =
            "SELECT DISTINCT chromosome "
            "FROM Variant "
            "ORDER BY chromosome";

        if (sqlite3_prepare_v2(db_, chr_query, -1, &stmt_chromosomes_, nullptr) != SQLITE_OK) {
            throw std::runtime_error("Failed to prepare chromosomes query");
        }
    }

    std::vector<VariantInfo> execute_query(sqlite3_stmt* stmt) {
        std::vector<VariantInfo> results;

        int rc;
        while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
            VariantInfo info;

            // Extract data from columns
            info.file_offset = static_cast<uint64_t>(sqlite3_column_int64(stmt, 0));
            info.variant_size = static_cast<uint32_t>(sqlite3_column_int(stmt, 1));

            const char* chr = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
            if (chr)
                info.chromosome = chr;

            info.position = static_cast<uint32_t>(sqlite3_column_int(stmt, 3));

            const char* rsid = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
            if (rsid)
                info.rsid = rsid;

            info.n_alleles = static_cast<uint16_t>(sqlite3_column_int(stmt, 5));

            const char* allele1 = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 6));
            if (allele1)
                info.allele1 = allele1;

            const char* allele2 = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 7));
            if (allele2)
                info.allele2 = allele2;

            results.push_back(info);
        }

        if (rc != SQLITE_DONE) {
            throw std::runtime_error("Query execution failed: " + std::string(sqlite3_errmsg(db_)));
        }

        return std::move(results);
    }

   private:
    sqlite3* db_;
    size_t variant_count_;
    std::unordered_map<std::string, std::string> metadata_;

    // Prepared statements
    sqlite3_stmt* stmt_by_region_ = nullptr;
    sqlite3_stmt* stmt_by_position_ = nullptr;
    sqlite3_stmt* stmt_by_rsid_ = nullptr;
    sqlite3_stmt* stmt_by_varid_ = nullptr;
    sqlite3_stmt* stmt_by_index_ = nullptr;
    sqlite3_stmt* stmt_offsets_range_ = nullptr;
    sqlite3_stmt* stmt_chromosomes_ = nullptr;

    // Thread safety
    mutable mutex_type mutex_;

    // Flag for query approach
    bool use_offset_for_range_ = false;
};

// BGIReader implementation

BGIReader::BGIReader(const std::string& bgi_path)
    : pimpl_(std::unique_ptr<Impl>(new Impl(bgi_path))) {}

BGIReader::~BGIReader() = default;

BGIReader::BGIReader(BGIReader&&) noexcept = default;

BGIReader& BGIReader::operator=(BGIReader&&) noexcept = default;

std::vector<VariantInfo> BGIReader::query_region(const std::string& chromosome, uint32_t start_pos,
                                                 uint32_t end_pos) {
    return pimpl_->query_region(chromosome, start_pos, end_pos);
}

std::vector<VariantInfo> BGIReader::query_position(const std::string& chromosome,
                                                   uint32_t position) {
    return pimpl_->query_position(chromosome, position);
}

std::vector<VariantInfo> BGIReader::query_variant_id(const std::string& variant_id) {
    return pimpl_->query_variant_id(variant_id);
}

VariantInfo BGIReader::get_variant(size_t index) {
    return pimpl_->get_variant(index);
}

std::vector<uint64_t> BGIReader::get_file_offsets(size_t start_idx, size_t end_idx) {
    return pimpl_->get_file_offsets(start_idx, end_idx);
}

std::vector<VariantInfo> BGIReader::get_all_variants() {
    return pimpl_->get_all_variants();
}

size_t BGIReader::get_variant_count() const {
    return pimpl_->get_variant_count();
}

std::vector<std::string> BGIReader::get_chromosomes() const {
    return pimpl_->get_chromosomes();
}

bool BGIReader::is_valid() const {
    return pimpl_->is_valid();
}

std::unordered_map<std::string, std::string> BGIReader::get_metadata() const {
    return pimpl_->get_metadata();
}

std::vector<VariantInfo> BGIReader::find_variants_by_filter(
    const std::string& chromosome, const std::vector<uint32_t>& positions,
    const std::vector<std::string>& alleles1, const std::vector<std::string>& alleles2,
    size_t batch_size) {
    return pimpl_->find_variants_by_filter(chromosome, positions, alleles1, alleles2, batch_size);
}

std::vector<VariantInfo> BGIReader::query_positions_batch(const std::string& chromosome,
                                                          const std::vector<uint32_t>& positions,
                                                          size_t batch_size) {
    return pimpl_->query_positions_batch(chromosome, positions, batch_size);
}

std::vector<VariantInfo> BGIReader::query_variant_ids_batch(
    const std::vector<std::string>& variant_ids, size_t batch_size) {
    return pimpl_->query_variant_ids_batch(variant_ids, batch_size);
}

}  // namespace index
}  // namespace bgen
}  // namespace io
}  // namespace ldcov