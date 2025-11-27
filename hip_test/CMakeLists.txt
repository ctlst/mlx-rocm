cmake_minimum_required(VERSION 3.18)
project(SimpleHIPTest)

# Enable HIP
enable_language(HIP)

# Find HIP
find_package(hip REQUIRED)

# Set HIP architecture
set(CMAKE_HIP_ARCHITECTURES gfx1030)

# Create executable
add_executable(simple_hip_test simple_hip_test.cpp)

# Link HIP runtime
target_link_libraries(simple_hip_test hip::device)

# Set C++ standard
set_property(TARGET simple_hip_test PROPERTY CXX_STANDARD 17)
