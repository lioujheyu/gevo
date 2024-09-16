cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(sw_GPU LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 14 CACHE STRING "")
set(CMAKE_CXX_EXTENSIONS OFF CACHE BOOL "")
set(CMAKE_CUDA_STANDARD 14 CACHE STRING "")
set(CMAKE_CUDA_EXTENSIONS OFF CACHE BOOL "")

#find_package(timemory REQUIRED COMPONENTS headers cuda cupti compile-options arch)
#find_package(cuda REQUIRED)
find_package(OpenMP REQUIRED)


# file(GLOB sources ${PROJECT_SOURCE_DIR}/ utils.cpp alignments.cpp kernel.cpp driver.cpp main.cpp)
file(GLOB sources ${PROJECT_SOURCE_DIR}/ utils.cpp alignments.cpp kernel.cpp driver_ptx.cpp main.cpp)
file(GLOB headers ${PROJECT_SOURCE_DIR}/utils.hpp alignments.hpp kernel.hpp driver.hpp)

set_source_files_properties(${sources} PROPERTIES
    LANGUAGE CUDA
    LINKER_LANGUAGE CUDA)

add_executable(program_gpu ${sources} ${headers})
target_compile_options(program_gpu PUBLIC
    $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_50,code=sm_50>
    $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_52,code=sm_52>
    $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_60,code=sm_60>
    $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_61,code=sm_61>
    $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_70,code=sm_70>
    $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_80,code=sm_80>
)
#target_link_libraries(program_gpu PUBLIC timemory OpenMP::OpenMP_CXX)
target_link_libraries(program_gpu PUBLIC OpenMP::OpenMP_CXX cudart_static cuda)
target_include_directories(program_gpu PUBLIC ${PROJECT_SOURCE_DIR})

get_property(LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)
if(CMAKE_CUDA_COMPILER AND "CUDA" IN_LIST LANGUAGES)
    target_compile_options(program_gpu PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -fopenmp>)
endif()

#add_custom_target(
#    regression_tests ALL
#    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/regression_test
#    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/regression_test/* ${CMAKE_BINARY_DIR}/regression_test
#)
#
#add_custom_target(
#    test-data ALL
#    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/test-data
#    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/test-data/* ${CMAKE_BINARY_DIR}/test-data
#)
