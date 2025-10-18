# SuperResolutionTest CMakeLists.txt

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(SuperResolutionTest SuperResolutionTest.cpp)

target_link_libraries(SuperResolutionTest 
    ${TORCH_LIBRARIES}
)

target_include_directories(SuperResolutionTest PRIVATE
    ${CMAKE_SOURCE_DIR}/source/Lib/CommonLib
    ${CMAKE_SOURCE_DIR}/source/Lib/CommonLib/..
)

target_compile_definitions(SuperResolutionTest PRIVATE
    _CRT_SECURE_NO_WARNINGS
)

set_property(TARGET SuperResolutionTest PROPERTY CXX_STANDARD 11)

# Copy LibTorch DLLs to output directory (Windows)
if(WIN32)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    foreach(DLL ${TORCH_DLLS})
        add_custom_command(TARGET SuperResolutionTest POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${DLL} $<TARGET_FILE_DIR:SuperResolutionTest>)
    endforeach()
endif()
