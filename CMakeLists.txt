# =============================================================================
# Simple Chess — optional CMake build
#
# The Makefile is the primary build path; this file is provided for users who
# prefer CMake or IDE integration. It is not required.
#
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j
# =============================================================================

cmake_minimum_required(VERSION 3.16)
project(simplechess CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

file(GLOB SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/*.cpp")

add_executable(simplechess ${SOURCES})

# Bake the version (from the single-source-of-truth VERSION file) and build date
# into the binary, matching the Makefile so CMake builds also self-report via
# UCI `id name`.
file(STRINGS "${CMAKE_SOURCE_DIR}/VERSION" SC_VERSION LIMIT_COUNT 1)
string(TIMESTAMP SC_BUILD_DATE "%Y-%m-%d")
target_compile_definitions(simplechess PRIVATE
    "SC_VERSION=\"${SC_VERSION}\""
    "SC_BUILD_DATE=\"${SC_BUILD_DATE}\""
)

target_include_directories(simplechess PRIVATE
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/external/chess-library/include"
)

# Release tuning. -mcpu=native targets the host Apple Silicon core.
target_compile_options(simplechess PRIVATE
    $<$<CONFIG:Release>:-O3 -DNDEBUG -funroll-loops>
    $<$<CONFIG:Release>:-mcpu=native>
    -Wall -Wextra
)

set_target_properties(simplechess PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)

find_package(Threads REQUIRED)
target_link_libraries(simplechess PRIVATE Threads::Threads)
