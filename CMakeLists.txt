cmake_minimum_required(VERSION 3.25.1)

project(InferenceRuntime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CheckCXXCompilerFlag)
include(CTest)

option(ENABLE_CUDA "Enable CUDA backend" ON)
option(ENABLE_AVX2 "Enable AVX2" OFF)
option(BUILD_BENCHMARKS "Build runtime benchmarks" ON)

if (ENABLE_CUDA)
    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA compute capability")
    endif ()
    enable_language(CUDA)
endif ()

set(LOWP_ENABLE_AVX2 ${ENABLE_AVX2} CACHE BOOL "Enable AVX2/F16C acceleration in lowp" FORCE)
add_subdirectory(thirdparty/lowp)

file(GLOB_RECURSE RUNTIME_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp"
)

list(REMOVE_ITEM RUNTIME_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/Main.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/Backend/CUDA/Backend.cpp"
)

if (ENABLE_CUDA)
    file(GLOB_RECURSE CUDA_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")
    list(APPEND RUNTIME_SOURCES
            "${CMAKE_CURRENT_SOURCE_DIR}/src/Backend/CUDA/Backend.cpp"
            ${CUDA_SOURCES}
    )
endif ()

add_library(runtime_core STATIC ${RUNTIME_SOURCES})

target_include_directories(runtime_core PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

target_compile_features(runtime_core PUBLIC cxx_std_20)

find_package(simdjson CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(svector REQUIRED)

if (ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
endif ()

target_link_libraries(runtime_core
        PRIVATE simdjson::simdjson
        PUBLIC spdlog::spdlog
        PUBLIC svector::svector
        PUBLIC lowp::lowp
)

if (ENABLE_CUDA)
    target_link_libraries(runtime_core PUBLIC CUDA::cudart)
    set_target_properties(runtime_core PROPERTIES
            CUDA_STANDARD 20
            CUDA_STANDARD_REQUIRED ON
    )
endif ()

target_compile_definitions(runtime_core PRIVATE
        RUNTIME_ENABLE_AVX2=$<BOOL:${ENABLE_AVX2}>
        RUNTIME_ENABLE_CUDA=$<BOOL:${ENABLE_CUDA}>
)

if (MSVC)
    if (NOT ENABLE_AVX2)
        target_compile_options(runtime_core PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/arch:SSE2>")
    endif ()
else ()
    if (ENABLE_AVX2)
        check_cxx_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)

        if (NOT COMPILER_SUPPORTS_FMA)
            message(FATAL_ERROR "ENABLE_AVX2=ON, but compiler does not support -mfma")
        endif ()

        target_compile_options(runtime_core PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-mfma>")
    else ()
        target_compile_options(runtime_core PRIVATE
                "$<$<COMPILE_LANGUAGE:CXX>:-mno-avx>"
                "$<$<COMPILE_LANGUAGE:CXX>:-mno-avx2>"
                "$<$<COMPILE_LANGUAGE:CXX>:-mno-fma>"
                "$<$<COMPILE_LANGUAGE:CXX>:-mno-f16c>"
        )
    endif ()
endif ()

add_executable(InferenceRuntime src/Main.cpp)
target_link_libraries(InferenceRuntime PRIVATE runtime_core)

if (BUILD_TESTING)
    find_package(GTest CONFIG REQUIRED)

    file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp"
    )

    add_executable(RuntimeTests ${TEST_SOURCES})
    target_link_libraries(RuntimeTests PRIVATE runtime_core GTest::gtest)

    include(GoogleTest)
    gtest_discover_tests(RuntimeTests)
endif ()

if (BUILD_BENCHMARKS)
    find_package(benchmark CONFIG REQUIRED)

    add_executable(RuntimeBenchmark benchmarks/RuntimeBenchmark.cpp)
    target_link_libraries(RuntimeBenchmark PRIVATE runtime_core benchmark::benchmark)
endif ()

message(STATUS "ENABLE_AVX2: ${ENABLE_AVX2}")
message(STATUS "ENABLE_CUDA: ${ENABLE_CUDA}")
message(STATUS "LOWP_ENABLE_AVX2: ${LOWP_ENABLE_AVX2}")
message(STATUS "BUILD_BENCHMARKS: ${BUILD_BENCHMARKS}")
