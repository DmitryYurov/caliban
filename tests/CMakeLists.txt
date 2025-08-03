
add_definitions(-DTEST_DATA_PATH="${PROJECT_SOURCE_DIR}/datasets")

add_executable(intrinsic_tests intrinsic_tests.cpp)
target_link_libraries(intrinsic_tests PRIVATE ${OpenCV_LIBS} calib GTest::gtest_main)
target_include_directories(intrinsic_tests PRIVATE ../include ${OpenCV_INCLUDE_DIRS})

add_executable(extrinsic_tests extrinsic_tests.cpp)
target_link_libraries(extrinsic_tests PRIVATE ${OpenCV_LIBS} calib GTest::gtest_main)
target_include_directories(extrinsic_tests PRIVATE ../include ${OpenCV_INCLUDE_DIRS})

if(WIN32)
    get_target_property(__dll_dbg opencv_world IMPORTED_LOCATION_DEBUG)
    get_target_property(__dll_release opencv_world  IMPORTED_LOCATION_RELEASE)

    add_custom_command(TARGET intrinsic_tests POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<$<CONFIG:debug>:${__dll_dbg}>$<$<CONFIG:release>:${__dll_release}>"
            $<TARGET_FILE_DIR:intrinsic_tests>
    )
    add_custom_command(TARGET extrinsic_tests POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<$<CONFIG:debug>:${__dll_dbg}>$<$<CONFIG:release>:${__dll_release}>"
            $<TARGET_FILE_DIR:extrinsic_tests>
    )
endif()

include(GoogleTest)
gtest_discover_tests(intrinsic_tests)
gtest_discover_tests(extrinsic_tests)
