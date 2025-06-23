#include "gcs_file_reader.h"
#include "gcs_retry_wrapper.h"
#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <cstring>

namespace ldcov {
namespace io {
namespace bgen {

GCSFileReader::GCSFileReader(const std::string& filename, size_t buffer_size)
    : gcsfs_module_(nullptr)
    , fs_(nullptr)
    , file_obj_(nullptr)
    , filename_(filename)
    , file_size_(0)
    , current_pos_(0)
    , is_open_(false)
    , buffer_size_(buffer_size)
    , buffer_start_(0)
    , buffer_valid_(0) {
    
    if (filename.substr(0, 5) != "gs://") {
        throw std::runtime_error("GCSFileReader: filename must start with gs://");
    }
    
    buffer_.resize(buffer_size_);
    
    initialize_python();
    open_file();
}

GCSFileReader::~GCSFileReader() {
    close();
    
    // Release Python objects
    gil_state_ = PyGILState_Ensure();
    
    Py_XDECREF(file_obj_);
    Py_XDECREF(fs_);
    Py_XDECREF(gcsfs_module_);
    
    PyGILState_Release(gil_state_);
}

void GCSFileReader::initialize_python() {
    gil_state_ = PyGILState_Ensure();
    
    // Import gcsfs module
    gcsfs_module_ = PyImport_ImportModule("gcsfs");
    if (!gcsfs_module_) {
        PyGILState_Release(gil_state_);
        throw std::runtime_error("Failed to import gcsfs module. Please install: pip install gcsfs");
    }
    
    // Create GCSFileSystem instance
    PyObject* gcs_class = PyObject_GetAttrString(gcsfs_module_, "GCSFileSystem");
    if (!gcs_class) {
        check_python_error("get GCSFileSystem class");
    }
    
    // Create filesystem with default credentials
    PyObject* args = PyTuple_New(0);
    PyObject* kwargs = PyDict_New();
    
    fs_ = PyObject_Call(gcs_class, args, kwargs);
    Py_DECREF(args);
    Py_DECREF(kwargs);
    Py_DECREF(gcs_class);
    
    if (!fs_) {
        check_python_error("create GCSFileSystem");
    }
    
    PyGILState_Release(gil_state_);
}

void GCSFileReader::open_file() {
    gil_state_ = PyGILState_Ensure();
    
    // Get file info to determine size
    PyObject* info_method = PyObject_GetAttrString(fs_, "info");
    if (!info_method) {
        check_python_error("get info method");
    }
    
    PyObject* path_arg = PyUnicode_FromString(filename_.c_str());
    PyObject* info_result = PyObject_CallFunctionObjArgs(info_method, path_arg, NULL);
    Py_DECREF(info_method);
    Py_DECREF(path_arg);
    
    if (!info_result) {
        check_python_error("get file info");
    }
    
    // Extract file size from info dict
    PyObject* size_obj = PyDict_GetItemString(info_result, "size");
    if (!size_obj) {
        Py_DECREF(info_result);
        throw std::runtime_error("Failed to get file size from GCS");
    }
    
    file_size_ = PyLong_AsUnsignedLongLong(size_obj);
    Py_DECREF(info_result);
    
    // Open file for reading
    PyObject* open_method = PyObject_GetAttrString(fs_, "open");
    if (!open_method) {
        check_python_error("get open method");
    }
    
    path_arg = PyUnicode_FromString(filename_.c_str());
    PyObject* mode_arg = PyUnicode_FromString("rb");
    
    // Set block_size for efficient reading
    PyObject* kwargs = PyDict_New();
    PyObject* block_size = PyLong_FromSize_t(buffer_size_);
    PyDict_SetItemString(kwargs, "block_size", block_size);
    Py_DECREF(block_size);
    
    PyObject* args = PyTuple_Pack(2, path_arg, mode_arg);
    file_obj_ = PyObject_Call(open_method, args, kwargs);
    
    Py_DECREF(args);
    Py_DECREF(kwargs);
    Py_DECREF(mode_arg);
    Py_DECREF(path_arg);
    Py_DECREF(open_method);
    
    if (!file_obj_) {
        check_python_error("open file");
    }
    
    is_open_ = true;
    PyGILState_Release(gil_state_);
}

size_t GCSFileReader::read(uint8_t* buffer, size_t size) {
    if (!is_open_) {
        throw std::runtime_error("GCSFileReader: file is not open");
    }
    
    size_t total_read = 0;
    
    while (total_read < size && current_pos_ < file_size_) {
        // Try to read from buffer first
        size_t from_buffer = read_from_buffer(buffer + total_read, size - total_read);
        total_read += from_buffer;
        current_pos_ += from_buffer;
        
        // If we need more data and haven't reached EOF, refill buffer
        if (total_read < size && current_pos_ < file_size_) {
            fill_buffer(current_pos_);
        }
    }
    
    return total_read;
}

size_t GCSFileReader::read_at(uint64_t offset, uint8_t* buffer, size_t size) {
    if (!is_open_) {
        throw std::runtime_error("GCSFileReader: file is not open");
    }
    
    // For read_at, we bypass the buffer and read directly
    return read_internal(offset, buffer, size);
}

void GCSFileReader::seek(uint64_t offset) {
    if (offset > file_size_) {
        throw std::runtime_error("GCSFileReader: seek beyond end of file");
    }
    current_pos_ = offset;
}

uint64_t GCSFileReader::tell() const {
    return current_pos_;
}

uint64_t GCSFileReader::size() const {
    return file_size_;
}

bool GCSFileReader::is_open() const {
    return is_open_;
}

void GCSFileReader::close() {
    if (!is_open_) {
        return;
    }
    
    gil_state_ = PyGILState_Ensure();
    
    if (file_obj_) {
        PyObject* close_method = PyObject_GetAttrString(file_obj_, "close");
        if (close_method) {
            PyObject* result = PyObject_CallObject(close_method, NULL);
            Py_XDECREF(result);
            Py_DECREF(close_method);
        }
    }
    
    PyGILState_Release(gil_state_);
    
    is_open_ = false;
}

size_t GCSFileReader::read_internal(uint64_t offset, uint8_t* buffer, size_t size) {
    // Wrap the entire read operation with retry logic
    return RetryWrapper<size_t>::execute_with_retry(
        [this, offset, buffer, size]() -> size_t {
            gil_state_ = PyGILState_Ensure();
            
            // Seek to position
            PyObject* seek_method = PyObject_GetAttrString(file_obj_, "seek");
            if (!seek_method) {
                check_python_error("get seek method");
            }
            
            PyObject* offset_arg = PyLong_FromUnsignedLongLong(offset);
            PyObject* seek_result = PyObject_CallFunctionObjArgs(seek_method, offset_arg, NULL);
            Py_DECREF(offset_arg);
            Py_DECREF(seek_method);
            
            if (!seek_result) {
                check_python_error("seek");
            }
            Py_DECREF(seek_result);
            
            // Read data
            PyObject* read_method = PyObject_GetAttrString(file_obj_, "read");
            if (!read_method) {
                check_python_error("get read method");
            }
            
            PyObject* size_arg = PyLong_FromSize_t(size);
            PyObject* data = PyObject_CallFunctionObjArgs(read_method, size_arg, NULL);
            Py_DECREF(size_arg);
            Py_DECREF(read_method);
            
            if (!data) {
                check_python_error("read");
            }
            
            // Extract bytes from Python object
            char* py_buffer;
            Py_ssize_t py_size;
            
            if (PyBytes_AsStringAndSize(data, &py_buffer, &py_size) < 0) {
                Py_DECREF(data);
                check_python_error("extract bytes");
            }
            
            size_t bytes_read = static_cast<size_t>(py_size);
            std::memcpy(buffer, py_buffer, bytes_read);
            
            Py_DECREF(data);
            PyGILState_Release(gil_state_);
            
            return bytes_read;
        },
        "GCS read",
        3  // max retries
    );
}

void GCSFileReader::fill_buffer(uint64_t offset) {
    buffer_start_ = offset;
    buffer_valid_ = read_internal(offset, buffer_.data(), buffer_size_);
}

size_t GCSFileReader::read_from_buffer(uint8_t* buffer, size_t size) {
    // Check if current position is within buffer
    if (current_pos_ < buffer_start_ || current_pos_ >= buffer_start_ + buffer_valid_) {
        return 0;  // Not in buffer
    }
    
    uint64_t buffer_offset = current_pos_ - buffer_start_;
    size_t available = buffer_valid_ - buffer_offset;
    size_t to_copy = std::min(size, available);
    
    std::memcpy(buffer, buffer_.data() + buffer_offset, to_copy);
    
    return to_copy;
}

void GCSFileReader::check_python_error(const std::string& operation) {
    if (PyErr_Occurred()) {
        PyObject *type, *value, *traceback;
        PyErr_Fetch(&type, &value, &traceback);
        
        std::string error_msg = "Python error in " + operation + ": ";
        
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
        
        PyGILState_Release(gil_state_);
        throw std::runtime_error(error_msg);
    }
}


} // namespace bgen
} // namespace io
} // namespace ldcov