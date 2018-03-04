# xemd: https:#github.com/rjsberry/xemd
#
# Copyright (C) 2018, Richard Berry <rjsberry@protonmail.com>
#
# Distributed under the terms of BSD 2-Clause "simplified" license. (See
# accompanying file LICENSE, or copy at
# https:#github.com/rjsberry/xemd/blob/master/LICENSE)
#

find_package(xtensor REQUIRED CONFIG)
set(XTENSOR_INCLUDE_DIR ${xtensor_INCLUDE_DIRS})
find_package(xtensor-blas REQUIRED CONFIG)
set(XTENSOR_BLAS_INCLUDE_DIR ${xblas_INCLUDE_DIRS})

configure_file(GTest.cmake.in googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
                RESULT_VARIABLE result
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download)
execute_process(COMMAND ${CMAKE_COMMAND} --build .
                RESULT_VARIABLE result
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download)
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
                 ${CMAKE_CURRENT_BINARY_DIR}/googletest-build)
set(GTEST_INCLUDE_DIRS "${gtest_SOURCE_DIR}/include")
set(GTEST_BOTH_LIBRARIES gtest_main gtest)

find_package(Threads)
find_package(xtensor REQUIRED)
find_package(xtensor-blas REQUIRED)
find_package(OpenBLAS REQUIRED)

include_directories(${XEMD_INCLUDE_DIR})
include_directories(${XTENSOR_INCLUDE_DIR})
include_directories(${XTENSOR_BLAS_INCLUDE_DIR})
include_directories(${GTEST_INCLUDE_DIRS})

set(XEMD_TESTS
    main.cpp
    test_xemd_interpolation.cpp
    test_xemd_xfindextrema.cpp
    test_xemd_xutils.cpp)

set(XEMD_TARGET test_xemd)

add_executable(${XEMD_TARGET} ${XEMD_TESTS} ${XEMD_HEADER})
add_dependencies(${XEMD_TARGET} gtest_main)

target_link_libraries(${XEMD_TARGET}
    ${OpenBLAS_LIBRARIES}
    ${GTEST_BOTH_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
)

add_custom_target(xtest COMMAND test_xemd)
