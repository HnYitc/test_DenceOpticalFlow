cmake_minimum_required(VERSION 2.8)
project( opt )
find_package( OpenCV 3.0 REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( opt src/opt.cpp)
target_link_libraries( opt ${OpenCV_LIBS} )

