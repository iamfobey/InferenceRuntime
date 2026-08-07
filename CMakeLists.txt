cmake_minimum_required(VERSION 4.3)

project(InferenceRuntime LANGUAGES CXX)

include(CheckCXXCompilerFlag)
include(FetchContent)
include(CTest)

find_package(OpenMP REQUIRED)

file(GLOB_RECURSE RUNTIME_SOURCES CONFIGURE_DEPENDS
        # src
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp"

        # thirdparty
        "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/simdjson/simdjson.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/simdjson/simdjson.h"
)

list(FILTER RUNTIME_SOURCES EXCLUDE REGEX ".*/Main\\.cpp$")
list(FILTER RUNTIME_SOURCES EXCLUDE REGEX ".*/Math/(Default)\\.cpp$")

set(MATH_IMPLEMENTATION
        "${CMAKE_CURRENT_SOURCE_DIR}/src/Math/Default.cpp"
)

add_library(runtime_core STATIC
        ${RUNTIME_SOURCES}
        ${MATH_IMPLEMENTATION}
)

target_include_directories(runtime_core PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty"
)

target_compile_features(runtime_core PUBLIC cxx_std_20)

target_link_libraries(runtime_core PRIVATE OpenMP::OpenMP_CXX)

if (MATH_COMPILE_OPTIONS)
    set_property(
            SOURCE "${MATH_IMPLEMENTATION}"
            APPEND PROPERTY COMPILE_OPTIONS
            ${MATH_COMPILE_OPTIONS}
    )
endif ()

add_executable(InferenceRuntime
        src/Main.cpp
)

target_link_libraries(InferenceRuntime PRIVATE
        runtime_core
)