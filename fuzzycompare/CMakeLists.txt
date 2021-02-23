cmake_minimum_required(VERSION 3.15)

project(fuzzycompare)

set(CMAKE_CXX_STANDARD 11)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
execute_process(
  COMMAND
    "${Python_EXECUTABLE}" -c
    "import pybind11; print(pybind11.get_cmake_dir())"
  OUTPUT_VARIABLE _tmp_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(fuzzycompare fuzzycompare.cpp)

install(TARGETS fuzzycompare DESTINATION .)