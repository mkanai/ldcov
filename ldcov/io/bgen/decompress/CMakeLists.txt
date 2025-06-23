# CMakeLists.txt for BGEN decompression components

cmake_minimum_required(VERSION 3.12)

# C++14 standard required
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Buffer Manager library
add_library(buffer_manager STATIC
    buffer_manager.cpp
    buffer_manager.h
)

# Compression utilities library
add_library(compression_utils STATIC
    compression_utils.cpp
    compression_utils.h
    decompressor.h
    simple_decompressor.cpp
    sequential_decompressor.cpp
    sequential_decompressor.h
    parallel_decompressor.cpp
    parallel_decompressor.h
    decompressor_factory.cpp
)

# Set include directories
target_include_directories(buffer_manager PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_include_directories(compression_utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../zlib-ng
    ${CMAKE_CURRENT_SOURCE_DIR}/../zstd/lib
)

# Enable position independent code for static library
set_property(TARGET buffer_manager PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET compression_utils PROPERTY POSITION_INDEPENDENT_CODE ON)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(buffer_manager PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wno-unused-parameter
        -O3
        -march=native
    )
    target_compile_options(compression_utils PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wno-unused-parameter
        -O3
        -march=native
    )
endif()

# Thread support
find_package(Threads REQUIRED)
target_link_libraries(buffer_manager PUBLIC Threads::Threads)

# Link compression utilities with vendored libraries
# Assuming zlib-ng and zstd are built as part of the parent project
target_link_libraries(compression_utils PUBLIC 
    buffer_manager
    ${CMAKE_CURRENT_SOURCE_DIR}/../build/zlib-ng/libz.a
    ${CMAKE_CURRENT_SOURCE_DIR}/../build/zstd/lib/libzstd.a
)

# Optional: Build example executable
option(BUILD_BUFFER_MANAGER_EXAMPLE "Build buffer manager example" OFF)
if(BUILD_BUFFER_MANAGER_EXAMPLE)
    add_executable(buffer_manager_example
        buffer_manager_example.cpp
    )
    target_link_libraries(buffer_manager_example PRIVATE buffer_manager)
endif()

# Optional: Build migration example
option(BUILD_MIGRATION_EXAMPLE "Build migration example" OFF)
if(BUILD_MIGRATION_EXAMPLE)
    add_executable(sequential_decompressor_migration
        sequential_decompressor_migration.cpp
    )
    target_link_libraries(sequential_decompressor_migration PRIVATE buffer_manager)
endif()

# Optional: Build decompressor test
option(BUILD_DECOMPRESSOR_TEST "Build decompressor test" OFF)
if(BUILD_DECOMPRESSOR_TEST)
    add_executable(test_decompressor
        test_decompressor.cpp
    )
    target_link_libraries(test_decompressor PRIVATE compression_utils)
endif()

# Optional: Build sequential decompressor test
option(BUILD_SEQUENTIAL_DECOMPRESSOR_TEST "Build sequential decompressor test" OFF)
if(BUILD_SEQUENTIAL_DECOMPRESSOR_TEST)
    add_executable(test_sequential_decompressor
        test_sequential_decompressor.cpp
    )
    target_link_libraries(test_sequential_decompressor PRIVATE compression_utils)
endif()

# Optional: Build parallel decompressor test
option(BUILD_PARALLEL_DECOMPRESSOR_TEST "Build parallel decompressor test" OFF)
if(BUILD_PARALLEL_DECOMPRESSOR_TEST)
    add_executable(test_parallel_decompressor
        test_parallel_decompressor.cpp
    )
    target_link_libraries(test_parallel_decompressor PRIVATE compression_utils)
endif()

# Export targets for use by parent project
export(TARGETS buffer_manager compression_utils FILE BufferManagerTargets.cmake)

# Install rules (optional)
install(TARGETS buffer_manager compression_utils
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(FILES 
    buffer_manager.h
    compression_utils.h
    decompressor.h
    sequential_decompressor.h
    parallel_decompressor.h
    DESTINATION include/ldcov/io/bgen/decompress
)