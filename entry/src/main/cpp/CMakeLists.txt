cmake_minimum_required(VERSION 3.4.1)
project(lexifold_native)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -fvisibility=hidden -O2")

add_library(lexifold_native SHARED
    napi_init.cpp
    ppm_engine.cpp
    neural_engine.cpp
)

target_link_libraries(lexifold_native PUBLIC
    libace_napi.z.so
    libhilog_ndk.z.so
)

target_include_directories(lexifold_native PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
