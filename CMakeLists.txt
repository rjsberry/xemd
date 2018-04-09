# xemd: https:#github.com/rjsberry/xemd
#
# Copyright (C) 2018, Richard Berry <rjsberry@protonmail.com>
#
# Distributed under the terms of BSD 2-Clause "simplified" license. (See
# accompanying file LICENSE, or copy at
# https:#github.com/rjsberry/xemd/blob/master/LICENSE)
#

cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_STANDARD 14)

set(XEMD_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(XEMD_HEADER ${XEMD_INCLUDE_DIR}/xemd/xemd.hpp)

project(xemd)

find_package(xtensor REQUIRED)
find_package(xtensor-interpolate REQUIRED)

enable_testing(true)
add_subdirectory(test)
