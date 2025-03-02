set(target calib_app)

add_executable(${target} main.cpp)
target_link_libraries(${target} PRIVATE ${OpenCV_LIBS} argparse calib)
target_include_directories(${target} PRIVATE ../include ${OpenCV_INCLUDE_DIRS})
