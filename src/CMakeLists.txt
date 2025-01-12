set(target calib)

add_library(${target} STATIC intrinsic.cpp)
target_link_libraries(${target} PRIVATE Eigen3::Eigen ceres)