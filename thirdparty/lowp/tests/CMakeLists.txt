find_package(GTest CONFIG QUIET)

if(NOT GTest_FOUND)
    if(NOT LOWP_FETCH_GTEST)
        message(FATAL_ERROR "GoogleTest was not found and LOWP_FETCH_GTEST=OFF")
    endif()

    include(FetchContent)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(googletest)
endif()

include(GoogleTest)

add_executable(lowp_modular_tests modular_headers_test.cpp)
target_link_libraries(lowp_modular_tests PRIVATE lowp::lowp GTest::gtest_main)
target_compile_features(lowp_modular_tests PRIVATE cxx_std_17)
gtest_discover_tests(lowp_modular_tests)

add_executable(lowp_single_header_tests single_header_test.cpp)
target_include_directories(lowp_single_header_tests PRIVATE "${PROJECT_SOURCE_DIR}/single_include")
target_link_libraries(lowp_single_header_tests PRIVATE GTest::gtest_main)
target_compile_features(lowp_single_header_tests PRIVATE cxx_std_17)
gtest_discover_tests(lowp_single_header_tests)

if(Python3_Interpreter_FOUND)
    add_test(
        NAME lowp.single_header_up_to_date
        COMMAND
            "${Python3_EXECUTABLE}"
            "${PROJECT_SOURCE_DIR}/scripts/amalgamate.py"
            --source-root "${PROJECT_SOURCE_DIR}/include/lowp"
            --entry lowp.hpp
            --output "${PROJECT_SOURCE_DIR}/single_include/lowp.hpp"
            --check
    )
endif()
