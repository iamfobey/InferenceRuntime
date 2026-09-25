cmake_minimum_required(VERSION 3.20)

project(
    lowp
    VERSION 0.1.0
    DESCRIPTION "Low-precision numeric types for C++"
    LANGUAGES CXX
)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

if(PROJECT_IS_TOP_LEVEL)
    set(LOWP_BUILD_TESTS_DEFAULT ON)
    set(LOWP_INSTALL_DEFAULT ON)
else()
    set(LOWP_BUILD_TESTS_DEFAULT OFF)
    set(LOWP_INSTALL_DEFAULT OFF)
endif()

option(LOWP_BUILD_TESTS "Build lowp tests" ${LOWP_BUILD_TESTS_DEFAULT})
option(LOWP_INSTALL "Enable lowp install rules" ${LOWP_INSTALL_DEFAULT})
option(LOWP_FETCH_GTEST "Fetch GoogleTest when a system package is unavailable" ON)

add_library(lowp INTERFACE)
add_library(lowp::lowp ALIAS lowp)

target_compile_features(lowp INTERFACE cxx_std_17)
target_include_directories(
    lowp
    INTERFACE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    file(GLOB_RECURSE LOWP_PUBLIC_HEADERS CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/lowp/*.hpp")

    add_custom_target(
        lowp-single-header
        COMMAND
            "${Python3_EXECUTABLE}"
            "${PROJECT_SOURCE_DIR}/scripts/amalgamate.py"
            --source-root "${PROJECT_SOURCE_DIR}/include/lowp"
            --entry lowp.hpp
            --output "${PROJECT_SOURCE_DIR}/single_include/lowp.hpp"
        DEPENDS ${LOWP_PUBLIC_HEADERS} "${PROJECT_SOURCE_DIR}/scripts/amalgamate.py"
        COMMENT "Generating single_include/lowp.hpp"
        VERBATIM
    )

    add_custom_target(
        lowp-check-single-header
        COMMAND
            "${Python3_EXECUTABLE}"
            "${PROJECT_SOURCE_DIR}/scripts/amalgamate.py"
            --source-root "${PROJECT_SOURCE_DIR}/include/lowp"
            --entry lowp.hpp
            --output "${PROJECT_SOURCE_DIR}/single_include/lowp.hpp"
            --check
        DEPENDS ${LOWP_PUBLIC_HEADERS} "${PROJECT_SOURCE_DIR}/scripts/amalgamate.py"
        COMMENT "Checking single_include/lowp.hpp"
        VERBATIM
    )
endif()

if(LOWP_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(LOWP_INSTALL)
    install(TARGETS lowp EXPORT lowpTargets)
    install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
    install(FILES "${PROJECT_SOURCE_DIR}/single_include/lowp.hpp" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

    configure_package_config_file(
        "${PROJECT_SOURCE_DIR}/cmake/lowpConfig.cmake.in"
        "${PROJECT_BINARY_DIR}/lowpConfig.cmake"
        INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/lowp"
    )

    write_basic_package_version_file(
        "${PROJECT_BINARY_DIR}/lowpConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )

    install(
        EXPORT lowpTargets
        FILE lowpTargets.cmake
        NAMESPACE lowp::
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/lowp"
    )

    install(
        FILES
            "${PROJECT_BINARY_DIR}/lowpConfig.cmake"
            "${PROJECT_BINARY_DIR}/lowpConfigVersion.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/lowp"
    )
endif()
