cmake_minimum_required(VERSION 3.25.1)

project(InferenceRuntime LANGUAGES CXX)

include(CheckCXXCompilerFlag)
include(FetchContent)
include(CTest)

option(ENABLE_OPENMP "Enable OpenMP" ON)
option(ENABLE_AVX2 "Enable AVX2" ON)

set(HAVE_OPENMP_SUPPORT 0)
set(HAVE_AVX2_SUPPORT 0)

file(GLOB_RECURSE RUNTIME_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/simdjson/simdjson.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/simdjson/simdjson.h"
)

add_library(runtime_core STATIC ${RUNTIME_SOURCES})

target_include_directories(runtime_core PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty"
)

target_compile_features(runtime_core PUBLIC cxx_std_20)

if (ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(runtime_core PRIVATE OpenMP::OpenMP_CXX)
    set(HAVE_OPENMP_SUPPORT 1)
endif ()

if (ENABLE_AVX2)
    file(GLOB AVX2_SOURCES
            "${CMAKE_CURRENT_SOURCE_DIR}/src/Math/Math.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/Utils/Converters.cpp"
    )

    list(REMOVE_ITEM RUNTIME_SOURCES ${AVX2_SOURCES})

    if (MSVC)
        check_cxx_compiler_flag("/arch:AVX2" COMPILER_SUPPORTS_AVX2)

        if (COMPILER_SUPPORTS_AVX2)
            set_source_files_properties(${AVX2_SOURCES} PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
            set(HAVE_AVX2_SUPPORT 1)
        else ()
            message(WARNING "AVX2 requested, but compiler does not support /arch:AVX2")
        endif ()
    else ()
        check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
        check_cxx_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)
        check_cxx_compiler_flag("-mf16c" COMPILER_SUPPORTS_F16C)

        if (COMPILER_SUPPORTS_AVX2 AND COMPILER_SUPPORTS_FMA AND COMPILER_SUPPORTS_F16C)
            set_source_files_properties(${AVX2_SOURCES} PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
            set(HAVE_AVX2_SUPPORT 1)
        else ()
            message(WARNING "AVX2/FMA/F16C requested, but compiler does not support all required flags")
        endif ()
    endif ()
endif ()

target_compile_definitions(runtime_core PUBLIC
        HAVE_OPENMP_SUPPORT=${HAVE_OPENMP_SUPPORT}
        HAVE_AVX2_SUPPORT=${HAVE_AVX2_SUPPORT}
)

add_executable(InferenceRuntime src/Main.cpp)

target_link_libraries(InferenceRuntime PRIVATE runtime_core)

message(STATUS "OpenMP: ${HAVE_OPENMP_SUPPORT}")
message(STATUS "AVX2: ${HAVE_AVX2_SUPPORT}")

