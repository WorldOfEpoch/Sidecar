cmake_minimum_required(VERSION 3.25)

project(Sidecar LANGUAGES C CXX)

include(CTest)
include(FetchContent)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SIDECAR_BUILD_TESTS "Build Sidecar unit tests" ON)
option(SIDECAR_ENABLE_CUDA "Enable CUDA components when they are introduced" OFF)
option(SIDECAR_ENABLE_LLAMA "Enable pinned external llama.cpp integration" OFF)
set(SIDECAR_LLAMA_CPP_ROOT "C:/SidecarDeps/llama.cpp" CACHE PATH
    "Pinned official llama.cpp checkout")
set(SIDECAR_LLAMA_CPP_BUILD_DIR "C:/SidecarDeps/llama.cpp/build-sidecar" CACHE PATH
    "Prebuilt pinned llama.cpp CMake build directory")
set(SIDECAR_LLAMA_CUDA_KERNEL_POLICY "DEFAULT" CACHE STRING
    "CUDA kernel policy used by the pinned llama.cpp build (DEFAULT, CUBLAS, or MMQ)")
set_property(CACHE SIDECAR_LLAMA_CUDA_KERNEL_POLICY PROPERTY STRINGS DEFAULT CUBLAS MMQ)
if(NOT SIDECAR_LLAMA_CUDA_KERNEL_POLICY STREQUAL "DEFAULT" AND
   NOT SIDECAR_LLAMA_CUDA_KERNEL_POLICY STREQUAL "CUBLAS" AND
   NOT SIDECAR_LLAMA_CUDA_KERNEL_POLICY STREQUAL "MMQ")
    message(FATAL_ERROR
        "SIDECAR_LLAMA_CUDA_KERNEL_POLICY must be DEFAULT, CUBLAS, or MMQ")
endif()

set(SIDECAR_SPEC_VERSION "M0-FROZEN-1")
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" SIDECAR_VERSION)
string(STRIP "${SIDECAR_VERSION}" SIDECAR_VERSION)

execute_process(
    COMMAND git rev-parse --verify HEAD
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RESULT_VARIABLE SIDECAR_GIT_RESULT
    OUTPUT_VARIABLE SIDECAR_GIT_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
if(NOT SIDECAR_GIT_RESULT EQUAL 0)
    set(SIDECAR_GIT_COMMIT "UNCOMMITTED")
endif()

# SQLite is pinned so benchmark databases never silently change implementation.
# The SHA3-256 value is published on https://www.sqlite.org/download.html.
FetchContent_Declare(
    sqlite_amalgamation
    URL https://www.sqlite.org/2026/sqlite-amalgamation-3530400.zip
    URL_HASH SHA3_256=628a44cfe82c66aed1ccbbe85a562d2e33ebe64b3288981ed76285612227934e
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(sqlite_amalgamation)

add_library(sidecar_sqlite STATIC
    "${sqlite_amalgamation_SOURCE_DIR}/sqlite3.c")
target_include_directories(sidecar_sqlite PUBLIC
    "${sqlite_amalgamation_SOURCE_DIR}")
target_compile_definitions(sidecar_sqlite PRIVATE
    SQLITE_DEFAULT_FOREIGN_KEYS=1
    SQLITE_DQS=0
    SQLITE_OMIT_LOAD_EXTENSION
    SQLITE_THREADSAFE=1)
if(MSVC)
    target_compile_options(sidecar_sqlite PRIVATE /w)
else()
    target_compile_options(sidecar_sqlite PRIVATE -w)
endif()

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/schemas/sidecar.sql" SIDECAR_SCHEMA_SQL)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/schemas/migrations/002_hardware_identity.sql"
     SIDECAR_MIGRATION_2_SQL)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/schemas/migrations/003_flight_recorder.sql"
     SIDECAR_MIGRATION_3_SQL)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/schemas/migrations/004_host_memory_lifecycle.sql"
     SIDECAR_MIGRATION_4_SQL)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/schemas/migrations/005_cuda_transfer_physics.sql"
     SIDECAR_MIGRATION_5_SQL)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/schemas/migrations/006_cuda_compute_transfer_overlap.sql"
     SIDECAR_MIGRATION_6_SQL)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/schemas/migrations/007_nvme_storage_physics.sql"
     SIDECAR_MIGRATION_7_SQL)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/schemas/migrations/008_memory_highway_pipeline.sql"
     SIDECAR_MIGRATION_8_SQL)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/schemas/migrations/009_llama_gguf_observation.sql"
     SIDECAR_MIGRATION_9_SQL)
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/schema.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/generated/sidecar/database/schema.hpp"
    @ONLY)
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/migrations.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/generated/sidecar/database/migrations.hpp"
    @ONLY)
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/version_config.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/generated/sidecar/version_config.hpp"
    @ONLY)

add_library(sidecar_core STATIC
    src/core/sha256.cpp
    src/core/version.cpp
    src/cuda/overlap.cpp
    src/cuda/overlap_cli.cpp
    src/cuda/transfer.cpp
    src/cuda/transfer_cli.cpp
    src/database/database.cpp
    src/database/llama_database.cpp
    src/database/storage_database.cpp
    src/hardware/format.cpp
    src/hardware/identity.cpp
    src/hardware/persistence.cpp
    src/memory/benchmark.cpp
    src/memory/cli.cpp
    src/memory/safety.cpp
    src/llama/cli.cpp
    src/llama/gguf_inspection.cpp
    src/llama/observation.cpp
    src/llama/runtime.cpp
    src/pipeline/physics.cpp
    src/pipeline/pipeline_cli.cpp
    src/database/pipeline_database.cpp
    src/storage/physics.cpp
    src/storage/storage_cli.cpp
    src/trace/benchmark.cpp
    src/trace/cli.cpp
    src/trace/format.cpp
    src/trace/recorder.cpp)
add_library(sidecar::core ALIAS sidecar_core)
target_include_directories(sidecar_core
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_link_libraries(sidecar_core PUBLIC sidecar_sqlite)

if(SIDECAR_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

set(SIDECAR_LLAMA_CPP_UPSTREAM "https://github.com/ggml-org/llama.cpp.git")
if(SIDECAR_ENABLE_LLAMA)
    if(NOT EXISTS "${SIDECAR_LLAMA_CPP_ROOT}/include/llama.h" OR
       NOT EXISTS "${SIDECAR_LLAMA_CPP_ROOT}/ggml/include/gguf.h")
        message(FATAL_ERROR "SIDECAR_ENABLE_LLAMA requires the pinned official checkout at SIDECAR_LLAMA_CPP_ROOT")
    endif()
    execute_process(
        COMMAND git rev-parse --verify HEAD
        WORKING_DIRECTORY "${SIDECAR_LLAMA_CPP_ROOT}"
        RESULT_VARIABLE SIDECAR_LLAMA_GIT_RESULT
        OUTPUT_VARIABLE SIDECAR_LLAMA_CPP_COMMIT
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT SIDECAR_LLAMA_GIT_RESULT EQUAL 0)
        message(FATAL_ERROR "Unable to resolve the pinned llama.cpp revision")
    endif()
    execute_process(
        COMMAND git status --porcelain
        WORKING_DIRECTORY "${SIDECAR_LLAMA_CPP_ROOT}"
        OUTPUT_VARIABLE SIDECAR_LLAMA_GIT_STATUS
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(SIDECAR_LLAMA_GIT_STATUS STREQUAL "")
        set(SIDECAR_LLAMA_CPP_DIRTY 0)
    else()
        set(SIDECAR_LLAMA_CPP_DIRTY 1)
    endif()
    execute_process(
        COMMAND git branch --show-current
        WORKING_DIRECTORY "${SIDECAR_LLAMA_CPP_ROOT}"
        OUTPUT_VARIABLE SIDECAR_LLAMA_CPP_BRANCH
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(SIDECAR_LLAMA_CPP_BRANCH STREQUAL "")
        set(SIDECAR_LLAMA_CPP_BRANCH "DETACHED")
    endif()
    set(SIDECAR_LLAMA_BIN_DIR "${SIDECAR_LLAMA_CPP_BUILD_DIR}/bin/Release")
    set(SIDECAR_LLAMA_LIBRARY "${SIDECAR_LLAMA_CPP_BUILD_DIR}/src/Release/llama.lib")
    set(SIDECAR_GGML_LIBRARY "${SIDECAR_LLAMA_CPP_BUILD_DIR}/ggml/src/Release/ggml.lib")
    set(SIDECAR_GGML_BASE_LIBRARY "${SIDECAR_LLAMA_CPP_BUILD_DIR}/ggml/src/Release/ggml-base.lib")
    foreach(required_file IN ITEMS
            "${SIDECAR_LLAMA_LIBRARY}" "${SIDECAR_GGML_LIBRARY}"
            "${SIDECAR_GGML_BASE_LIBRARY}" "${SIDECAR_LLAMA_BIN_DIR}/llama.dll"
            "${SIDECAR_LLAMA_BIN_DIR}/ggml.dll" "${SIDECAR_LLAMA_BIN_DIR}/ggml-base.dll"
            "${SIDECAR_LLAMA_BIN_DIR}/ggml-cpu.dll")
        if(NOT EXISTS "${required_file}")
            message(FATAL_ERROR "Pinned llama.cpp build artifact is missing: ${required_file}")
        endif()
    endforeach()
    file(SHA256 "${SIDECAR_LLAMA_BIN_DIR}/llama.dll" SIDECAR_LLAMA_DLL_SHA256)
    file(SHA256 "${SIDECAR_LLAMA_BIN_DIR}/ggml.dll" SIDECAR_GGML_DLL_SHA256)
    file(SHA256 "${SIDECAR_LLAMA_BIN_DIR}/ggml-base.dll" SIDECAR_GGML_BASE_DLL_SHA256)
    file(SHA256 "${SIDECAR_LLAMA_BIN_DIR}/ggml-cpu.dll" SIDECAR_GGML_CPU_DLL_SHA256)
    if(EXISTS "${SIDECAR_LLAMA_BIN_DIR}/ggml-cuda.dll")
        file(SHA256 "${SIDECAR_LLAMA_BIN_DIR}/ggml-cuda.dll" SIDECAR_GGML_CUDA_DLL_SHA256)
    else()
        set(SIDECAR_GGML_CUDA_DLL_SHA256 "NOT_PRESENT")
    endif()
    if(EXISTS "${SIDECAR_LLAMA_BIN_DIR}/llama-bench.exe")
        file(SHA256 "${SIDECAR_LLAMA_BIN_DIR}/llama-bench.exe" SIDECAR_LLAMA_BENCH_SHA256)
    else()
        set(SIDECAR_LLAMA_BENCH_SHA256 "NOT_PRESENT")
    endif()
    add_library(sidecar_llama_runtime SHARED IMPORTED GLOBAL)
    set_target_properties(sidecar_llama_runtime PROPERTIES
        IMPORTED_IMPLIB "${SIDECAR_LLAMA_LIBRARY}"
        IMPORTED_LOCATION "${SIDECAR_LLAMA_BIN_DIR}/llama.dll")
    add_library(sidecar_ggml_runtime SHARED IMPORTED GLOBAL)
    set_target_properties(sidecar_ggml_runtime PROPERTIES
        IMPORTED_IMPLIB "${SIDECAR_GGML_LIBRARY}"
        IMPORTED_LOCATION "${SIDECAR_LLAMA_BIN_DIR}/ggml.dll")
    add_library(sidecar_ggml_base_runtime SHARED IMPORTED GLOBAL)
    set_target_properties(sidecar_ggml_base_runtime PROPERTIES
        IMPORTED_IMPLIB "${SIDECAR_GGML_BASE_LIBRARY}"
        IMPORTED_LOCATION "${SIDECAR_LLAMA_BIN_DIR}/ggml-base.dll")
    target_include_directories(sidecar_core PRIVATE
        "${SIDECAR_LLAMA_CPP_ROOT}/include"
        "${SIDECAR_LLAMA_CPP_ROOT}/ggml/include")
    target_link_libraries(sidecar_core PRIVATE
        sidecar_llama_runtime sidecar_ggml_runtime sidecar_ggml_base_runtime)
    target_compile_definitions(sidecar_core PUBLIC SIDECAR_LLAMA_ENABLED=1)
    if(SIDECAR_ENABLE_CUDA)
        set(SIDECAR_LLAMA_BUILD_CONFIGURATION
            "Release-GGML_CUDA-sm86-${SIDECAR_LLAMA_CUDA_KERNEL_POLICY}")
        set(SIDECAR_LLAMA_CUDA_TOOLKIT_VERSION "${CUDAToolkit_VERSION}")
    else()
        set(SIDECAR_LLAMA_BUILD_CONFIGURATION "Release-GGUF-inspection-CUDA-disabled-Sidecar")
        set(SIDECAR_LLAMA_CUDA_TOOLKIT_VERSION "DISABLED")
    endif()
    message(STATUS "Sidecar llama.cpp enabled at pinned commit ${SIDECAR_LLAMA_CPP_COMMIT}")
else()
    set(SIDECAR_LLAMA_CPP_COMMIT "DISABLED")
    set(SIDECAR_LLAMA_CPP_BRANCH "DISABLED")
    set(SIDECAR_LLAMA_BUILD_CONFIGURATION "DISABLED")
    set(SIDECAR_LLAMA_CPP_DIRTY 0)
    set(SIDECAR_LLAMA_DLL_SHA256 "DISABLED")
    set(SIDECAR_GGML_DLL_SHA256 "DISABLED")
    set(SIDECAR_GGML_BASE_DLL_SHA256 "DISABLED")
    set(SIDECAR_GGML_CPU_DLL_SHA256 "DISABLED")
    set(SIDECAR_GGML_CUDA_DLL_SHA256 "DISABLED")
    set(SIDECAR_LLAMA_BENCH_SHA256 "DISABLED")
    set(SIDECAR_LLAMA_CUDA_TOOLKIT_VERSION "DISABLED")
    target_compile_definitions(sidecar_core PUBLIC SIDECAR_LLAMA_ENABLED=0)
    message(STATUS "Sidecar llama.cpp integration disabled (SIDECAR_ENABLE_LLAMA=OFF)")
endif()
target_compile_definitions(sidecar_core PUBLIC
    SIDECAR_LLAMA_CPP_UPSTREAM="${SIDECAR_LLAMA_CPP_UPSTREAM}"
    SIDECAR_LLAMA_CPP_COMMIT="${SIDECAR_LLAMA_CPP_COMMIT}"
    SIDECAR_LLAMA_CPP_BRANCH="${SIDECAR_LLAMA_CPP_BRANCH}"
    SIDECAR_LLAMA_BUILD_CONFIGURATION="${SIDECAR_LLAMA_BUILD_CONFIGURATION}"
    SIDECAR_LLAMA_CUDA_TOOLKIT_VERSION="${SIDECAR_LLAMA_CUDA_TOOLKIT_VERSION}"
    SIDECAR_LLAMA_CPP_DIRTY=${SIDECAR_LLAMA_CPP_DIRTY}
    SIDECAR_LLAMA_DLL_SHA256="${SIDECAR_LLAMA_DLL_SHA256}"
    SIDECAR_GGML_DLL_SHA256="${SIDECAR_GGML_DLL_SHA256}"
    SIDECAR_GGML_BASE_DLL_SHA256="${SIDECAR_GGML_BASE_DLL_SHA256}"
    SIDECAR_GGML_CPU_DLL_SHA256="${SIDECAR_GGML_CPU_DLL_SHA256}"
    SIDECAR_GGML_CUDA_DLL_SHA256="${SIDECAR_GGML_CUDA_DLL_SHA256}"
    SIDECAR_LLAMA_BENCH_SHA256="${SIDECAR_LLAMA_BENCH_SHA256}")

if(WIN32)
    target_sources(sidecar_core PRIVATE
        src/hardware/windows_discovery.cpp
        src/llama/storage_provenance_windows.cpp
        src/memory/host_memory_windows.cpp
        src/storage/windows_storage.cpp)
    target_link_libraries(sidecar_core PRIVATE cfgmgr32 setupapi psapi)
else()
    target_sources(sidecar_core PRIVATE
        src/hardware/unsupported_discovery.cpp
        src/llama/storage_provenance_unsupported.cpp
        src/memory/host_memory_unsupported.cpp
        src/storage/unsupported_storage.cpp)
endif()

if(SIDECAR_ENABLE_CUDA)
    target_sources(sidecar_core PRIVATE
        src/cuda/cuda_discovery.cpp
        src/cuda/nvml_discovery.cpp
        src/cuda/overlap_cuda.cu
        src/cuda/transfer_cuda.cpp)
    target_sources(sidecar_core PRIVATE src/pipeline/pipeline_cuda.cu)
    target_link_libraries(sidecar_core PRIVATE CUDA::cudart CUDA::cublas)
    set_target_properties(sidecar_core PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES 86)
    target_compile_definitions(sidecar_core PUBLIC SIDECAR_CUDA_ENABLED=1)
    message(STATUS "Sidecar CUDA discovery enabled with toolkit ${CUDAToolkit_VERSION}")
else()
    target_sources(sidecar_core PRIVATE
        src/cuda/cuda_discovery_stub.cpp
        src/cuda/overlap_stub.cpp
        src/cuda/transfer_stub.cpp)
    target_sources(sidecar_core PRIVATE src/pipeline/pipeline_stub.cpp)
    target_compile_definitions(sidecar_core PUBLIC SIDECAR_CUDA_ENABLED=0)
    message(STATUS "Sidecar CUDA discovery disabled (SIDECAR_ENABLE_CUDA=OFF)")
endif()

if(MSVC)
    target_compile_options(sidecar_core PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4;/permissive-;/EHsc>)
else()
    target_compile_options(sidecar_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(sidecar-lab tools/sidecar_lab/main.cpp)
target_link_libraries(sidecar-lab PRIVATE sidecar::core)
if(SIDECAR_ENABLE_LLAMA)
    add_custom_command(TARGET sidecar-lab POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${SIDECAR_LLAMA_BIN_DIR}/llama.dll"
            "${SIDECAR_LLAMA_BIN_DIR}/ggml.dll"
            "${SIDECAR_LLAMA_BIN_DIR}/ggml-base.dll"
            "${SIDECAR_LLAMA_BIN_DIR}/ggml-cpu.dll"
            "$<TARGET_FILE_DIR:sidecar-lab>")
    if(SIDECAR_ENABLE_CUDA)
        add_custom_command(TARGET sidecar-lab POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${SIDECAR_LLAMA_BIN_DIR}/ggml-cuda.dll"
                "$<TARGET_FILE_DIR:sidecar-lab>")
    endif()
endif()
if(MSVC)
    target_compile_options(sidecar-lab PRIVATE /W4 /permissive- /EHsc)
else()
    target_compile_options(sidecar-lab PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(BUILD_TESTING AND SIDECAR_BUILD_TESTS)
    add_subdirectory(tests)
endif()
