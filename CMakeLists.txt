cmake_minimum_required(VERSION 3.10)

# Set the project name
project(CalibanProject)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(cmake/get_cpm.cmake)

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

CPMAddPackage(
        GITLAB_REPOSITORY libeigen/eigen
        GIT_TAG 3.4.0
        OPTIONS "EIGEN_BUILD_DOC OFF"
        "EIGEN_BUILD_PKGCONFIG OFF"
        "EIGEN_BUILD_DOC OFF"
        "EIGEN_BUILD_TESTING OFF"
)

CPMAddPackage(
        GITHUB_REPOSITORY ceres-solver/ceres-solver
        GIT_TAG 2.2.0
        OPTIONS "MINIGLOG ON"
        "PROVIDE_UNINSTALL_TARGET OFF"
        "BUILD_TESTING OFF"
)

CPMAddPackage(
        GITHUB_REPOSITORY google/googletest
        GIT_TAG v1.16.0
)

CPMAddPackage(
        GITHUB_REPOSITORY p-ranav/argparse
        GIT_TAG v3.2
)

find_package(OpenCV REQUIRED)

# add subdirectories
add_subdirectory(src)
add_subdirectory(app)

add_subdirectory(tests)
