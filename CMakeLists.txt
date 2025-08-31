cmake_minimum_required(VERSION 3.14)
project(ai_studio LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -----------------------------
# Find or fetch pybind11
# -----------------------------
find_package(pybind11 QUIET)

if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found, fetching from GitHub...")
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.13.6
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# -----------------------------
# Core library
# -----------------------------
file(GLOB_RECURSE CORE_SOURCES "${CMAKE_SOURCE_DIR}/src/core/*.cpp")
file(GLOB_RECURSE NODE_SOURCES "${CMAKE_SOURCE_DIR}/src/nodes/*.cpp")

add_library(aistudio_core STATIC ${CORE_SOURCES} ${NODE_SOURCES})
target_include_directories(aistudio_core PUBLIC "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(aistudio_core PUBLIC pybind11::module)
set_target_properties(aistudio_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# -----------------------------
# Python bindings
# -----------------------------
file(GLOB_RECURSE BINDINGS_SOURCES "${CMAKE_SOURCE_DIR}/aistudio/bindings/*.cpp")

if(BINDINGS_SOURCES)
    pybind11_add_module(graph ${BINDINGS_SOURCES})
    target_link_libraries(graph PRIVATE aistudio_core)

    # Output compiled module to build/bindings
    set_target_properties(graph PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bindings"
        OUTPUT_NAME graph
    )
endif()


# add_executable(ai_studio main.cpp)
# target_link_libraries(ai_studio PRIVATE aistudio_core)
