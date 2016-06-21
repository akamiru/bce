cmake_minimum_required(VERSION 2.8.6)

project(bce4)

option(divsufsort_STATIC "Prefer static divsufsort" ON)
option(USE_OPENMP "Use OpenMP if available" ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include(CheckCXXCompilerFlag)

function(add_cxx_compiler_flags var flags)
  foreach(flag ${flags})
    string(REGEX REPLACE "[^a-zA-Z0-9]" "_" flag_var "CXXFLAG_${flag}")
    check_cxx_compiler_flag("${flag}" ${flag_var})
    if(${flag_var})
      set(${var} "${${var}} ${flag}")
    endif()
    unset(flag_var)
  endforeach()
  set(${var} "${${var}}" PARENT_SCOPE)
endfunction()

if(NOT MSVC)
  add_cxx_compiler_flags(CMAKE_CXX_FLAGS "-std=c++14;-Wall;-pedantic;-march=native;-fno-exceptions")
  if(WIN32)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static")
  endif()
endif()

find_package(divsufsort REQUIRED)
include_directories(${divsufsort_INCLUDE_DIRS})
add_definitions(${divsufsort_DEFINITIONS})

if(USE_OPENMP)
  find_package(OpenMP)
  if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  endif()
endif()

add_executable(bce main.cpp)

target_link_libraries(bce ${divsufsort_LIBRARIES})
if(NOT MSVC)
  target_link_libraries(bce pthread)
endif()

install(TARGETS bce RUNTIME DESTINATION bin)
