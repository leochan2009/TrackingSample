cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(openni_tracking)

find_package(PCL 1.7 REQUIRED)
#SET(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/../Slicer-unixBuild/VTKv6-build")
find_package(VTK 7.0 REQUIRED NO_MODULE)
FIND_LIBRARY(OpenH264Lib openh264 /Users/longquanchen/Desktop/Github/OpenIGTLink-xcodeBuild/Testing/OpenH264)
set(H264Directory /Users/longquanchen/Desktop/Github/OpenIGTLink-xcodeBuild/Testing/OpenH264)
include_directories(${H264Directory})
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (tracking_sample tracking_sample.cpp)
target_link_libraries (tracking_sample ${PCL_LIBRARIES})

add_executable (planar_segmentation planar_segmentation.cpp)
target_link_libraries (planar_segmentation ${PCL_LIBRARIES})

add_executable (cylinder_segmentation cylinder_segmentation.cpp)
target_link_libraries (cylinder_segmentation ${PCL_LIBRARIES})

SET(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/../Slicer-unixBuild/OpenIGTLink-build")
message(${CMAKE_PREFIX_PATH})
find_package(OpenIGTLink REQUIRED)
include(${OpenIGTLink_USE_FILE})
FIND_LIBRARY(OpenIGTLinkLib OpenIGTLink /Users/longquanchen/Desktop/Github/Slicer-unixBuild/OpenIGTLink-build/bin/Debug)
add_executable (KinectModeling KinectModeling.cpp CupModeling.cpp)
target_link_libraries (KinectModeling ${PCL_LIBRARIES} ${OpenIGTLinkLib} ${VTK_LIBRARIES} ${OpenH264Lib})

