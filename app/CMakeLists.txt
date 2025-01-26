set(target calib_app)

find_package(OpenCV REQUIRED)

CPMAddPackage(
        GITHUB_REPOSITORY p-ranav/argparse
        GIT_TAG v3.2
)

add_executable(${target} main.cpp)
target_link_libraries(${target} PRIVATE ${OpenCV_LIBS} argparse calib)
target_include_directories(${target} PRIVATE ../include ${OpenCV_INCLUDE_DIRS})
