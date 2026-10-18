#
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

cmake_minimum_required(VERSION 2.8.9)
project(bytefast)

set(CMAKE_CXX_STANDARD 17)

# The version number.
set (bytefast_VERSION_MAJOR 0)
set (bytefast_VERSION_MINOR 1)

include_directories(bytefast)

set(CMAKE_CXX_FLAGS " -pthread -std=c++17 -funroll-loops -O3 -march=native")

set(HEADER_FILES
    src/args.h
    src/bytefast_c.h
    src/safetensors.h
    src/densematrix.h
    src/dictionary.h
    src/bytefast.h
    src/loss.h
    src/matrix.h
    src/meter.h
    src/model.h
    src/norm_table.h
    src/productquantizer.h
    src/quantmatrix.h
    src/real.h
    src/textnorm.h
    src/unicode_fold.h
    src/utils.h
    src/vector.h)

set(SOURCE_FILES
    src/args.cc
    src/safetensors.cc
    src/densematrix.cc
    src/dictionary.cc
    src/bytefast.cc
    src/bytefast_c.cc
    src/loss.cc
    src/matrix.cc
    src/meter.cc
    src/model.cc
    src/productquantizer.cc
    src/quantmatrix.cc
    src/utils.cc
    src/vector.cc)


if (NOT MSVC)
  include(GNUInstallDirs)
  configure_file("bytefast.pc.in" "bytefast.pc" @ONLY)
  install(FILES "${CMAKE_BINARY_DIR}/bytefast.pc" DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
endif()

add_library(bytefast-shared SHARED ${SOURCE_FILES} ${HEADER_FILES})
add_library(bytefast-static STATIC ${SOURCE_FILES} ${HEADER_FILES})
add_library(bytefast-static_pic STATIC ${SOURCE_FILES} ${HEADER_FILES})
set_target_properties(bytefast-shared PROPERTIES OUTPUT_NAME bytefast
  SOVERSION "${bytefast_VERSION_MAJOR}")
set_target_properties(bytefast-static PROPERTIES OUTPUT_NAME bytefast)
set_target_properties(bytefast-static_pic PROPERTIES OUTPUT_NAME bytefast_pic
  POSITION_INDEPENDENT_CODE True)
add_executable(bytefast-bin src/main.cc)
target_link_libraries(bytefast-bin pthread bytefast-static)
set_target_properties(bytefast-bin PROPERTIES PUBLIC_HEADER "${HEADER_FILES}" OUTPUT_NAME bytefast)
install (TARGETS bytefast-shared
    LIBRARY DESTINATION lib)
install (TARGETS bytefast-static
    ARCHIVE DESTINATION lib)
install (TARGETS bytefast-static_pic
    ARCHIVE DESTINATION lib)
install (TARGETS bytefast-bin
    RUNTIME DESTINATION bin
 PUBLIC_HEADER DESTINATION include/bytefast)
