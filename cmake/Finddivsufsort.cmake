#.rst:
# Finddivsufsort
# --------------
#
# Find divsufsort library
#
# This will define the following variables::
#
#   divsufsort_FOUND        - True if library found
#   divsufsort_INCLUDE_DIRS - Locations of include files
#   divsufsort_LIBRARIES    - Libraries for divsufsort
#   divsufsort_DEFINITIONS  - Defines to use, if any
#   divsufsort_VERSION      - Version of library found, if available
#
# and the :prop_tgt:`IMPORTED` target::
#
#   divsufsort::divsufsort
#
# These variables may be set before use::
#
#   divsufsort_ROOT_DIR     - Look for library here
#   divsufsort_STATIC       - If true, prefer static library

#=============================================================================
# Copyright (c) 2016 Joergen Ibsen
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#=============================================================================

if(divsufsort_STATIC)
  set(_divsufsort_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
  if(WIN32)
    set(CMAKE_FIND_LIBRARY_SUFFIXES .lib .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
  else()
    set(CMAKE_FIND_LIBRARY_SUFFIXES .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
  endif()
endif()

find_package(PkgConfig QUIET)
pkg_check_modules(PC_divsufsort QUIET libdivsufsort)

find_path(divsufsort_INCLUDE_DIR
  NAMES divsufsort.h
  HINTS
    "${divsufsort_ROOT_DIR}/include"
    ${PC_divsufsort_INCLUDEDIR}
    ${PC_divsufsort_INCLUDE_DIRS}
  DOC "The divsufsort include directory"
)
mark_as_advanced(divsufsort_INCLUDE_DIR)

find_library(divsufsort_LIBRARY
  NAMES divsufsort
  HINTS
    "${divsufsort_ROOT_DIR}/lib"
    ${PC_divsufsort_LIBDIR}
    ${PC_divsufsort_LIBRARY_DIRS}
  PATH_SUFFIXES
    Release
    Debug
  DOC "The divsufsort library"
)
mark_as_advanced(divsufsort_LIBRARY)

set(divsufsort_VERSION ${PC_divsufsort_VERSION})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(divsufsort
  REQUIRED_VARS
    divsufsort_LIBRARY
    divsufsort_INCLUDE_DIR
  VERSION_VAR divsufsort_VERSION
)

# Workaround for older CMake without FOUND_VAR
set(divsufsort_FOUND ${DIVSUFSORT_FOUND})

if(divsufsort_FOUND)
  set(divsufsort_INCLUDE_DIRS "${divsufsort_INCLUDE_DIR}")
  set(divsufsort_LIBRARIES "${divsufsort_LIBRARY}")
  set(divsufsort_DEFINITIONS ${PC_divsufsort_CFLAGS_OTHER})

  if(NOT TARGET divsufsort::divsufsort)
    add_library(divsufsort::divsufsort UNKNOWN IMPORTED)
    set_target_properties(divsufsort::divsufsort PROPERTIES
      IMPORTED_LOCATION "${divsufsort_LIBRARY}"
      INTERFACE_COMPILE_OPTIONS "${PC_divsufsort_CFLAGS_OTHER}"
      INTERFACE_INCLUDE_DIRECTORIES "${divsufsort_INCLUDE_DIR}"
  )
  endif()
endif()

if(divsufsort_STATIC)
  set(CMAKE_FIND_LIBRARY_SUFFIXES ${_divsufsort_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES})
  unset(_divsufsort_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES)
endif()
