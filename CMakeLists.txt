cmake_minimum_required(VERSION 3.25.1)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(InferenceRuntime LANGUAGES CXX)

include(CheckCXXCompilerFlag)
include(CTest)

option(ENABLE_AVX2 "Enable AVX2" ON)
option(BUILD_BENCHMARKS "Build runtime benchmarks" ON)


add_subdirectory(./thirdparty/lowp)

file(GLOB_RECURSE RUNTIME_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp"
)

list(REMOVE_ITEM RUNTIME_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/Main.cpp")

add_library(runtime_core STATIC ${RUNTIME_SOURCES})

target_include_directories(runtime_core PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

target_compile_features(runtime_core PUBLIC cxx_std_20)

find_package(simdjson CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(svector)

target_link_libraries(runtime_core PRIVATE simdjson::simdjson PUBLIC spdlog::spdlog svector::svector lowp::lowp)

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
        else ()
            message(WARNING "AVX2 requested, but compiler does not support /arch:AVX2")
        endif ()
    else ()
        check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)

        if (COMPILER_SUPPORTS_MARCH_NATIVE)
            set_source_files_properties(${AVX2_SOURCES} PROPERTIES COMPILE_OPTIONS "-march=native")
        else ()
            message(WARNING "Native CPU optimization requested, but compiler does not support -march=native")
        endif ()
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

message(STATUS "CPU optimization: ${ENABLE_AVX2}")
message(STATUS "Benchmark: ${BUILD_BENCHMARKS}")
