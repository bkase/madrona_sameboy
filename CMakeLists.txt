cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
cmake_policy(VERSION 3.18)

include("${CMAKE_CURRENT_SOURCE_DIR}/vendored/madrona/cmake/madrona_init.cmake")

project(MadronaSameBoy LANGUAGES C CXX)

include(setup)
include(dependencies)

add_subdirectory("${MADRONA_DIR}" madrona EXCLUDE_FROM_ALL)
add_subdirectory(src)
