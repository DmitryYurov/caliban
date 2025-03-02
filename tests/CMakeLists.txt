add_executable(intrinsic_tests intrinsic_tests.cpp)
target_link_libraries(intrinsic_tests PRIVATE ${OpenCV_LIBS} calib GTest::gtest_main)
target_include_directories(intrinsic_tests PRIVATE ../include ${OpenCV_INCLUDE_DIRS})

enable_testing()

include(GoogleTest)
gtest_discover_tests(intrinsic_tests)
