cmake_minimum_required(VERSION 3.17)
set(CMAKE_BUILD_TYPE Debug)

project(cuprof)
enable_language(CUDA)
find_package(CUDAToolkit)

set(CMAKE_CXX_STANDARD 11)

set_source_files_properties(cuinj.cpp PROPERTIES LANGUAGE CUDA)
add_library(cuinj SHARED cuinj.cpp)
target_link_libraries(cuinj cupti cuda)

install(TARGETS cuinj DESTINATION .)
