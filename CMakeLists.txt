cmake_minimum_required(VERSION 3.14)
project(ai_studio LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ========== THIS IS A CRITICAL SECTION ==========
if(DEFINED ENV{VIRTUAL_ENV})
    if(WIN32)
        set(Python_EXECUTABLE "$ENV{VIRTUAL_ENV}/Scripts/python.exe")
    else()
        set(Python_EXECUTABLE "$ENV{VIRTUAL_ENV}/bin/python")
    endif()
    message(STATUS "Using virtual environment Python: ${Python_EXECUTABLE}")
else()
    find_program(PYTHON_EXECUTABLE NAMES python python3)
    if(PYTHON_EXECUTABLE)
        set(Python_EXECUTABLE ${PYTHON_EXECUTABLE})
    endif()
endif()

find_package(Python COMPONENTS Interpreter Development REQUIRED)

set(PYTHON_EXECUTABLE ${Python_EXECUTABLE})
set(PYTHON_INCLUDE_DIR ${Python_INCLUDE_DIRS})
set(PYTHON_LIBRARY ${Python_LIBRARIES})

message(STATUS "Python executable: ${Python_EXECUTABLE}")
message(STATUS "Python include dir: ${Python_INCLUDE_DIRS}")
message(STATUS "Python libraries: ${Python_LIBRARIES}")
# =============== END OF THE SECTION ==============

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

file(GLOB_RECURSE CORE_SOURCES "${CMAKE_SOURCE_DIR}/src/core/*.cpp")
file(GLOB_RECURSE NODE_SOURCES "${CMAKE_SOURCE_DIR}/src/nodes/*.cpp")
file(GLOB_RECURSE UTILS_SOURCES "${CMAKE_SOURCE_DIR}/src/utils/*.cpp")

add_library(accelera_core SHARED
    ${CORE_SOURCES}
    ${NODE_SOURCES}
    ${UTILS_SOURCES}
)

target_include_directories(accelera_core PUBLIC "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(accelera_core PUBLIC pybind11::module)
set_target_properties(accelera_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_compile_definitions(accelera_core PUBLIC ACCELERA_BUILD_DLL)

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/bindings")

file(GLOB_RECURSE BINDINGS_SOURCES 
    "${CMAKE_SOURCE_DIR}/accelera/bindings/*.cpp"
    "${CMAKE_SOURCE_DIR}/accelera/bindings/**/*.cpp"
)

foreach(binding_file ${BINDINGS_SOURCES})
    get_filename_component(module_name ${binding_file} NAME_WE)
    
    if(TARGET ${module_name})
        message(WARNING "Duplicate module name '${module_name}' found! File: ${binding_file}")
        string(RANDOM LENGTH 4 RANDOM_SUFFIX)
        set(module_name "${module_name}_${RANDOM_SUFFIX}")
        message(STATUS "Renaming duplicate to: ${module_name}")
    endif()
    
    message(STATUS "Creating Python module: ${module_name} from ${binding_file}")
    
    pybind11_add_module(${module_name} ${binding_file})
    target_link_libraries(${module_name} PRIVATE accelera_core)
    
    set_target_properties(${module_name} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bindings"
        LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bindings"
        LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bindings"
        LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/bindings"
        LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_BINARY_DIR}/bindings"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bindings"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bindings"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bindings"
        RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/bindings"
        RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_BINARY_DIR}/bindings"
        OUTPUT_NAME ${module_name}
    )

    add_custom_command(TARGET ${module_name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:accelera_core>
        "${CMAKE_BINARY_DIR}/bindings"
    )
endforeach()

message(STATUS "Generated ${BINDINGS_SOURCES} Python modules in ${CMAKE_BINARY_DIR}/bindings/")

install(TARGETS accelera_core
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY "${CMAKE_BINARY_DIR}/bindings/"
    DESTINATION lib/python/site-packages
    FILES_MATCHING PATTERN "*.so" PATTERN "*.pyd"
)
