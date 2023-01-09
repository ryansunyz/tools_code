cmake_minimum_required(VERSION 2.6)
project(pcl_test)

find_package(PCL REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
include_directories('~/workspace/baidu/adu-3rd/boost/output/include')
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(pcl_test pcl_test.cpp)

target_link_libraries (pcl_test ${PCL_LIBRARIES})

install(TARGETS pcl_test RUNTIME DESTINATION bin)