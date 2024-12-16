cmake_minimum_required(VERSION 3.10)

project(cpp-random_analysis)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(statistical_analysis statistical_analysis.cpp xoshiro256ss.h)