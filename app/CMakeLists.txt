set(target calib_app)

add_executable(${target} main.cpp)
target_link_libraries(${target} PRIVATE ${OpenCV_LIBS} argparse calib)
target_include_directories(${target} PRIVATE ../include ${OpenCV_INCLUDE_DIRS})

if(WIN32)
    get_target_property(__dll_dbg opencv_world IMPORTED_LOCATION_DEBUG)
    get_target_property(__dll_release opencv_world  IMPORTED_LOCATION_RELEASE)

    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<$<CONFIG:debug>:${__dll_dbg}>$<$<CONFIG:release>:${__dll_release}>"
            $<TARGET_FILE_DIR:${target}>
    )
endif()
