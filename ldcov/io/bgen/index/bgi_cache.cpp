#include "bgi_cache.h"
#include <Python.h>
#include <fstream>
#include <iostream>
#include <sys/stat.h>

namespace ldcov {
namespace io {
namespace bgen {
namespace index {

std::string BGICache::ensureLocalBGI(const std::string& bgi_path) {
    // If it's already a local path, return as-is
    if (bgi_path.substr(0, 5) != "gs://") {
        return bgi_path;
    }
    
    // Extract filename from GCS path
    std::string filename = getFilename(bgi_path);
    std::string local_path = "./" + filename;
    
    // Check if file already exists locally
    if (fileExists(local_path)) {
        std::cout << "Using existing BGI file: " << local_path << std::endl;
        return local_path;
    }
    
    // Download from GCS
    std::cout << "Downloading BGI index from " << bgi_path << " to " << local_path << "..." << std::endl;
    downloadFromGCS(bgi_path, local_path);
    
    return local_path;
}

std::string BGICache::getFilename(const std::string& path) {
    size_t pos = path.find_last_of('/');
    if (pos != std::string::npos) {
        return path.substr(pos + 1);
    }
    return path;
}

bool BGICache::fileExists(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

void BGICache::downloadFromGCS(const std::string& gcs_path, const std::string& local_path) {
    PyGILState_STATE gil_state = PyGILState_Ensure();
    
    try {
        // Import gcsfs
        PyObject* gcsfs_module = PyImport_ImportModule("gcsfs");
        if (!gcsfs_module) {
            PyGILState_Release(gil_state);
            throw std::runtime_error("Failed to import gcsfs module. Please install: pip install gcsfs");
        }
        
        // Create GCSFileSystem
        PyObject* gcs_class = PyObject_GetAttrString(gcsfs_module, "GCSFileSystem");
        PyObject* fs = PyObject_CallObject(gcs_class, nullptr);
        Py_DECREF(gcs_class);
        
        if (!fs) {
            Py_DECREF(gcsfs_module);
            PyGILState_Release(gil_state);
            throw std::runtime_error("Failed to create GCSFileSystem");
        }
        
        // Call fs.get(gcs_path, local_path)
        PyObject* get_method = PyObject_GetAttrString(fs, "get");
        PyObject* args = PyTuple_Pack(2, 
                                      PyUnicode_FromString(gcs_path.c_str()),
                                      PyUnicode_FromString(local_path.c_str()));
        
        PyObject* result = PyObject_Call(get_method, args, nullptr);
        
        Py_DECREF(args);
        Py_DECREF(get_method);
        Py_DECREF(fs);
        Py_DECREF(gcsfs_module);
        
        if (!result) {
            // Handle Python error
            PyObject *type, *value, *traceback;
            PyErr_Fetch(&type, &value, &traceback);
            
            std::string error_msg = "Failed to download BGI file: ";
            if (value) {
                PyObject* str_value = PyObject_Str(value);
                if (str_value) {
                    const char* error_str = PyUnicode_AsUTF8(str_value);
                    if (error_str) {
                        error_msg += error_str;
                    }
                    Py_DECREF(str_value);
                }
            }
            
            Py_XDECREF(type);
            Py_XDECREF(value);
            Py_XDECREF(traceback);
            
            PyGILState_Release(gil_state);
            throw std::runtime_error(error_msg);
        }
        
        Py_DECREF(result);
        PyGILState_Release(gil_state);
        
        std::cout << "BGI index downloaded successfully" << std::endl;
        
    } catch (...) {
        PyGILState_Release(gil_state);
        throw;
    }
}

} // namespace index
} // namespace bgen
} // namespace io
} // namespace ldcov