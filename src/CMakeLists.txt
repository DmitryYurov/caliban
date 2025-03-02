set(target calib)

add_library(${target} STATIC intrinsic.cpp extrinsic.cpp)
target_link_libraries(${target} PRIVATE Eigen3::Eigen ceres)
target_include_directories(${target} PUBLIC ../include)