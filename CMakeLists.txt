cmake_minimum_required(VERSION 3.2)
project( measure )

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( measure_v1.0 measure_v1.0.cpp )
target_link_libraries( measure_v1.0 ${OpenCV_LIBS} )

add_executable( measure measure.cpp )
target_link_libraries( measure ${OpenCV_LIBS} )

add_executable( src src.cpp )
target_link_libraries( src ${OpenCV_LIBS} )
