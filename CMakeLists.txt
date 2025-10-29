cmake_minimum_required(VERSION 3.14)
project(pybind11_example)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(extern/pybind11)
add_subdirectory(extern/eigen)
pybind11_add_module(Point Point.cpp)
target_include_directories(Point PRIVATE extern/eigen)

