#ifndef LDCOV_BGEN_FILE_READER_WRAPPER_H
#define LDCOV_BGEN_FILE_READER_WRAPPER_H

#include "reader_interface.h"
#include <Python.h>
#include <string>
#include <cstdint>
#include <stdexcept>

namespace ldcov {
namespace io {
namespace bgen {

/**
 * FileReaderWrapper - Wraps a Python file object to implement the C++ FileReader interface
 * 
 * This allows Python file objects (including mmap objects) to be used with the C++ 
 * decompressor infrastructure.
 */
class FileReaderWrapper : public FileReader {
public:
    explicit FileReaderWrapper(PyObject* py_file) 
        : py_file_(py_file), current_pos_(0), file_size_(0) {
        if (!py_file_) {
            throw std::invalid_argument("Python file object is null");
        }
        
        // Increase reference count
        Py_INCREF(py_file_);
        
        // Get file size
        PyObject* seek_method = PyObject_GetAttrString(py_file_, "seek");
        PyObject* tell_method = PyObject_GetAttrString(py_file_, "tell");
        
        if (seek_method && tell_method) {
            // Save current position
            PyObject* orig_pos = PyObject_CallObject(tell_method, nullptr);
            
            // Seek to end
            PyObject* args = Py_BuildValue("(ii)", 0, 2);  // SEEK_END
            PyObject_CallObject(seek_method, args);
            Py_DECREF(args);
            
            // Get size
            PyObject* size_obj = PyObject_CallObject(tell_method, nullptr);
            file_size_ = PyLong_AsLongLong(size_obj);
            Py_XDECREF(size_obj);
            
            // Restore position
            args = Py_BuildValue("(O)", orig_pos);
            PyObject_CallObject(seek_method, args);
            Py_DECREF(args);
            Py_XDECREF(orig_pos);
        }
        
        Py_XDECREF(seek_method);
        Py_XDECREF(tell_method);
        
        // Get filename if available
        PyObject* name_attr = PyObject_GetAttrString(py_file_, "name");
        if (name_attr) {
            if (PyUnicode_Check(name_attr)) {
                const char* name = PyUnicode_AsUTF8(name_attr);
                if (name) {
                    filename_ = name;
                }
            }
            Py_DECREF(name_attr);
        }
    }
    
    ~FileReaderWrapper() override {
        // Release Python object
        PyGILState_STATE gstate = PyGILState_Ensure();
        Py_XDECREF(py_file_);
        PyGILState_Release(gstate);
    }
    
    size_t read(uint8_t* buffer, size_t size) override {
        PyGILState_STATE gstate = PyGILState_Ensure();
        
        PyObject* read_method = PyObject_GetAttrString(py_file_, "read");
        if (!read_method) {
            PyGILState_Release(gstate);
            throw std::runtime_error("Python file object has no read method");
        }
        
        PyObject* args = Py_BuildValue("(n)", size);
        PyObject* result = PyObject_CallObject(read_method, args);
        
        Py_DECREF(args);
        Py_DECREF(read_method);
        
        if (!result) {
            PyGILState_Release(gstate);
            throw std::runtime_error("Failed to read from Python file object");
        }
        
        // Get bytes from result
        Py_ssize_t bytes_read = 0;
        char* data = nullptr;
        
        if (PyBytes_Check(result)) {
            PyBytes_AsStringAndSize(result, &data, &bytes_read);
            if (data && bytes_read > 0) {
                memcpy(buffer, data, bytes_read);
            }
        }
        
        Py_DECREF(result);
        PyGILState_Release(gstate);
        
        current_pos_ += bytes_read;
        return static_cast<size_t>(bytes_read);
    }
    
    size_t read_at(uint64_t offset, uint8_t* buffer, size_t size) override {
        PyGILState_STATE gstate = PyGILState_Ensure();
        
        // Try to use read_at method if available (for mmap)
        PyObject* read_at_method = PyObject_GetAttrString(py_file_, "read_at");
        if (read_at_method) {
            PyObject* args = Py_BuildValue("(Kn)", offset, size);
            PyObject* result = PyObject_CallObject(read_at_method, args);
            
            Py_DECREF(args);
            Py_DECREF(read_at_method);
            
            if (result) {
                Py_ssize_t bytes_read = 0;
                char* data = nullptr;
                
                if (PyBytes_Check(result)) {
                    PyBytes_AsStringAndSize(result, &data, &bytes_read);
                    if (data && bytes_read > 0) {
                        memcpy(buffer, data, bytes_read);
                    }
                }
                
                Py_DECREF(result);
                PyGILState_Release(gstate);
                return static_cast<size_t>(bytes_read);
            }
        }
        
        // Fall back to seek + read
        seek(offset);
        size_t bytes_read = read(buffer, size);
        PyGILState_Release(gstate);
        return bytes_read;
    }
    
    void seek(uint64_t offset) override {
        PyGILState_STATE gstate = PyGILState_Ensure();
        
        PyObject* seek_method = PyObject_GetAttrString(py_file_, "seek");
        if (!seek_method) {
            PyGILState_Release(gstate);
            throw std::runtime_error("Python file object has no seek method");
        }
        
        PyObject* args = Py_BuildValue("(K)", offset);
        PyObject* result = PyObject_CallObject(seek_method, args);
        
        Py_DECREF(args);
        Py_DECREF(seek_method);
        Py_XDECREF(result);
        
        current_pos_ = offset;
        PyGILState_Release(gstate);
    }
    
    uint64_t tell() const override {
        return current_pos_;
    }
    
    uint64_t size() const override {
        return file_size_;
    }
    
    bool is_open() const override {
        PyGILState_STATE gstate = PyGILState_Ensure();
        
        PyObject* closed_attr = PyObject_GetAttrString(py_file_, "closed");
        bool is_closed = false;
        
        if (closed_attr) {
            is_closed = PyObject_IsTrue(closed_attr);
            Py_DECREF(closed_attr);
        }
        
        PyGILState_Release(gstate);
        return !is_closed;
    }
    
    void close() override {
        PyGILState_STATE gstate = PyGILState_Ensure();
        
        PyObject* close_method = PyObject_GetAttrString(py_file_, "close");
        if (close_method) {
            PyObject* result = PyObject_CallObject(close_method, nullptr);
            Py_XDECREF(result);
            Py_DECREF(close_method);
        }
        
        PyGILState_Release(gstate);
    }
    
    const std::string& filename() const override {
        return filename_;
    }
    
private:
    PyObject* py_file_;
    uint64_t current_pos_;
    uint64_t file_size_;
    std::string filename_;
};

} // namespace bgen
} // namespace io
} // namespace ldcov

#endif // LDCOV_BGEN_FILE_READER_WRAPPER_H