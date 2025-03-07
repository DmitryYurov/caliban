
add_definitions(-DTEST_DATA_PATH="${PROJECT_SOURCE_DIR}/datasets")

add_executable(intrinsic_tests intrinsic_tests.cpp)
target_link_libraries(intrinsic_tests PRIVATE ${OpenCV_LIBS} calib GTest::gtest_main)
target_include_directories(intrinsic_tests PRIVATE ../include ${OpenCV_INCLUDE_DIRS})

add_executable(extrinsic_tests extrinsic_tests.cpp)
target_link_libraries(extrinsic_tests PRIVATE ${OpenCV_LIBS} calib GTest::gtest_main)
target_include_directories(extrinsic_tests PRIVATE ../include ${OpenCV_INCLUDE_DIRS})

enable_testing()

include(GoogleTest)
gtest_discover_tests(intrinsic_tests)
