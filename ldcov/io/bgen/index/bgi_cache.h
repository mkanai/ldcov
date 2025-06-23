#ifndef LDCOV_BGEN_BGI_CACHE_H
#define LDCOV_BGEN_BGI_CACHE_H

#include <string>

namespace ldcov {
namespace io {
namespace bgen {
namespace index {

/**
 * BGICache - BGI file management for remote BGEN files
 * 
 * When accessing BGEN files from remote storage (e.g., GCS), downloads
 * the BGI index file to the current working directory if needed.
 * Similar to bcftools approach - just checks local file existence.
 */
class BGICache {
public:
    /**
     * Get local BGI path, downloading if necessary
     * 
     * @param bgi_path Remote BGI path (e.g., gs://bucket/file.bgen.bgi)
     * @return Local path to BGI file (in current directory)
     */
    static std::string ensureLocalBGI(const std::string& bgi_path);
    
private:
    /**
     * Extract filename from path
     */
    static std::string getFilename(const std::string& path);
    
    /**
     * Check if file exists
     */
    static bool fileExists(const std::string& path);
    
    /**
     * Download file from GCS
     */
    static void downloadFromGCS(const std::string& gcs_path, const std::string& local_path);
};

} // namespace index
} // namespace bgen
} // namespace io
} // namespace ldcov

#endif // LDCOV_BGEN_SIMPLE_BGI_CACHE_H