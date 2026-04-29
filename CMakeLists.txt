cmake_minimum_required(VERSION 3.18)
project(reinforcement_mahjong_cpp LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

set(MAHJONG_SOURCES
    cpp/src/tile_utils.cpp
    cpp/src/hand_analysis.cpp
    cpp/src/yaku.cpp
    cpp/src/scoring.cpp
    cpp/src/engine.cpp
    cpp/src/bindings.cpp
)

pybind11_add_module(_mahjong_cpp ${MAHJONG_SOURCES})
target_include_directories(_mahjong_cpp PRIVATE cpp/include)
target_compile_features(_mahjong_cpp PRIVATE cxx_std_17)
target_compile_options(_mahjong_cpp PRIVATE
    $<$<CONFIG:Release>:-O2>
    $<$<CONFIG:Debug>:-g -O0>
)

install(TARGETS _mahjong_cpp DESTINATION .)
