# NNModelTest CMakeLists.txt

# Find required packages
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Create the test executable
add_executable(NNModelTest NNModelTest.cpp)

# Link libraries
target_link_libraries(NNModelTest 
    ${TORCH_LIBRARIES}
)

# Set include directories
target_include_directories(NNModelTest PRIVATE
    ${CMAKE_SOURCE_DIR}/source/Lib/CommonLib
    ${CMAKE_SOURCE_DIR}/source/Lib/CommonLib/..
)

# Compiler definitions
target_compile_definitions(NNModelTest PRIVATE
    _CRT_SECURE_NO_WARNINGS
)

# Set C++ standard
set_property(TARGET NNModelTest PROPERTY CXX_STANDARD 11)
