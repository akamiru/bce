cmake_minimum_required(VERSION 2.6)
project(bce4)

SET(CMAKE_CXX_FLAGS "-s -Wall -march=native -fopenmp -fno-exceptions -std=c++14")
SET(CMAKE_LINKER_FLAGS "-s -Wall -march=native -fopenmp -fno-exceptions -std=c++14")

add_executable(bce main.cpp)

target_link_libraries(bce divsufsort pthread)

install(TARGETS bce RUNTIME DESTINATION bin)
