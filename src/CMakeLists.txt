set(SAMEBOY_CORE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../vendored/SameBoy/Core")

set(SAMEBOY_CORE_SRCS
    "${SAMEBOY_CORE_DIR}/apu.c"
    "${SAMEBOY_CORE_DIR}/camera.c"
    "${SAMEBOY_CORE_DIR}/display.c"
    "${SAMEBOY_CORE_DIR}/gb.c"
    "${SAMEBOY_CORE_DIR}/joypad.c"
    "${SAMEBOY_CORE_DIR}/mbc.c"
    "${SAMEBOY_CORE_DIR}/memory.c"
    "${SAMEBOY_CORE_DIR}/printer.c"
    "${SAMEBOY_CORE_DIR}/random.c"
    "${SAMEBOY_CORE_DIR}/rumble.c"
    "${SAMEBOY_CORE_DIR}/save_state.c"
    "${SAMEBOY_CORE_DIR}/sgb.c"
    "${SAMEBOY_CORE_DIR}/sm83_cpu.c"
    "${SAMEBOY_CORE_DIR}/timing.c"
    "${SAMEBOY_CORE_DIR}/workboy.c"
)

add_library(sameboy_core STATIC ${SAMEBOY_CORE_SRCS})
target_include_directories(sameboy_core PUBLIC ${SAMEBOY_CORE_DIR})
target_compile_definitions(sameboy_core PRIVATE
    GB_INTERNAL
    GB_VERSION=\"0.0\"
    GB_DISABLE_DEBUGGER
    GB_DISABLE_CHEATS
    GB_DISABLE_CHEAT_SEARCH
    GB_DISABLE_REWIND
)

set(SIMULATOR_SRCS
    consts.hpp
    types.hpp
    sim.hpp
    sim.cpp
)

add_library(madrona_sameboy_sim STATIC ${SIMULATOR_SRCS})
target_include_directories(madrona_sameboy_sim PUBLIC ${SAMEBOY_CORE_DIR})
target_compile_definitions(madrona_sameboy_sim PUBLIC
    GB_INTERNAL
    GB_DISABLE_DEBUGGER
    GB_DISABLE_CHEATS
    GB_DISABLE_CHEAT_SEARCH
    GB_DISABLE_REWIND
)
target_link_libraries(madrona_sameboy_sim
    PUBLIC
        madrona_mw_core
        sameboy_core
)

find_package(OpenGL REQUIRED)

add_executable(viewer viewer.cpp)

target_link_libraries(viewer PRIVATE
    madrona_mw_cpu
    madrona_mw_core
    madrona_common
    madrona_sameboy_sim
    glfw
    OpenGL::GL
)

target_compile_definitions(viewer PRIVATE
    DEFAULT_ROM_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../cpu_instrs.gb"
)

add_executable(headless headless.cpp)

target_link_libraries(headless PRIVATE
    madrona_mw_cpu
    madrona_mw_core
    madrona_common
    madrona_sameboy_sim
)

target_compile_definitions(headless PRIVATE
    DEFAULT_ROM_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../cpu_instrs.gb"
)
