set(target calib)

add_library(${target} STATIC intrinsic.cpp extrinsic.cpp jaco.cpp)
target_link_libraries(${target} PUBLIC ${OpenCV_LIBS} PRIVATE Eigen3::Eigen ceres)
target_include_directories(${target} PUBLIC ../include)