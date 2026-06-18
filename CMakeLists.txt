    cmake_minimum_required(VERSION 3.14)

    if(POLICY CMP0091)
        cmake_policy(SET CMP0091 NEW)
    endif()

    project(ai_studio LANGUAGES C CXX)

    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    if(MSVC)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
        foreach(_lang C CXX)
            foreach(_config DEBUG RELWITHDEBINFO MINSIZEREL RELEASE)
                string(REPLACE "/MDd" "/MD" CMAKE_${_lang}_FLAGS_${_config} "${CMAKE_${_lang}_FLAGS_${_config}}")
            endforeach()
        endforeach()
        add_compile_definitions($<$<CONFIG:Debug>:_ITERATOR_DEBUG_LEVEL=0>)
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(-Wno-deprecated-enum-enum-conversion)
    endif()

    set(ACCELERA_LLVM_VERSION "18" CACHE STRING "LLVM/Clang major version used by the code parallelizer")
    set(ACCELERA_LLVM_RELEASE "18.1.8" CACHE STRING "LLVM/Clang release used for automatic Windows installation")
    option(ACCELERA_AUTO_INSTALL_LLVM "Automatically install LLVM/Clang on supported systems" ON)
    option(ACCELERA_AUTO_INSTALL_GRAPHVIZ "Automatically install Graphviz/dot on supported systems" ON)
    set(ACCELERA_LLVM_WINDOWS_ROOT
        "${CMAKE_BINARY_DIR}/_deps/llvm-${ACCELERA_LLVM_RELEASE}-windows"
        CACHE PATH "Local LLVM/Clang install root used by the Windows code parallelizer build"
    )
    set(LLVM_AVAILABLE OFF)

    function(accelera_llvm_prefix_is_usable _llvm_prefix _result_var)
        set(_usable OFF)
        if(EXISTS "${_llvm_prefix}/bin/clang.exe"
            AND EXISTS "${_llvm_prefix}/include/clang/AST/ASTContext.h")
            if(EXISTS "${_llvm_prefix}/lib/cmake/llvm/LLVMConfig.cmake"
                OR EXISTS "${_llvm_prefix}/lib/clangTooling.lib"
                OR EXISTS "${_llvm_prefix}/lib/libclangTooling.a")
                set(_usable ON)
            endif()
        endif()

        set(${_result_var} ${_usable} PARENT_SCOPE)
    endfunction()

    function(accelera_register_llvm_prefix _llvm_prefix)
        if(EXISTS "${_llvm_prefix}")
            list(APPEND CMAKE_PREFIX_PATH
                "${_llvm_prefix}"
                "${_llvm_prefix}/lib/cmake/llvm"
                "${_llvm_prefix}/lib/cmake/clang"
            )

            set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}" PARENT_SCOPE)

            if(WIN32 AND EXISTS "${_llvm_prefix}/bin")
                set(ENV{PATH} "${_llvm_prefix}/bin;$ENV{PATH}")
            endif()

            if(EXISTS "${_llvm_prefix}/lib/cmake/llvm/LLVMConfig.cmake")
                set(LLVM_DIR "${_llvm_prefix}/lib/cmake/llvm" CACHE PATH "" FORCE)
            endif()

            if(EXISTS "${_llvm_prefix}/lib/cmake/clang/ClangConfig.cmake")
                set(Clang_DIR "${_llvm_prefix}/lib/cmake/clang" CACHE PATH "" FORCE)
            endif()
        endif()
    endfunction()

    function(accelera_collect_llvm_dirs _root _result_var)
        set(_dirs "")
        file(GLOB _candidates "${_root}/clang+llvm-*")
        foreach(_candidate ${_candidates})
            if(IS_DIRECTORY "${_candidate}")
                list(APPEND _dirs "${_candidate}")
            endif()
        endforeach()

        set(${_result_var} ${_dirs} PARENT_SCOPE)
    endfunction()

    function(accelera_find_diaguids _result_var)
        set(_diaguids_candidates "")

        find_program(VSWHERE_EXECUTABLE vswhere
            PATHS "$ENV{ProgramFiles\(x86\)}/Microsoft Visual Studio/Installer"
            NO_DEFAULT_PATH
        )

        if(VSWHERE_EXECUTABLE)
            execute_process(
                COMMAND "${VSWHERE_EXECUTABLE}" -latest -products * -property installationPath
                OUTPUT_VARIABLE _vs_install
                ERROR_QUIET
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
            if(_vs_install)
                list(APPEND _diaguids_candidates "${_vs_install}/DIA SDK/lib/amd64/diaguids.lib")
            endif()
        endif()

        if(DEFINED ENV{VSINSTALLDIR})
            list(APPEND _diaguids_candidates "$ENV{VSINSTALLDIR}/DIA SDK/lib/amd64/diaguids.lib")
        endif()

        file(GLOB _vs_diaguids
            "$ENV{ProgramFiles\(x86\)}/Microsoft Visual Studio/2022/*/DIA SDK/lib/amd64/diaguids.lib"
            "$ENV{ProgramFiles\(x86\)}/Microsoft Visual Studio/2019/*/DIA SDK/lib/amd64/diaguids.lib"
        )
        list(APPEND _diaguids_candidates ${_vs_diaguids})

        foreach(_candidate ${_diaguids_candidates})
            if(EXISTS "${_candidate}")
                set(${_result_var} "${_candidate}" PARENT_SCOPE)
                return()
            endif()
        endforeach()

        set(${_result_var} "" PARENT_SCOPE)
    endfunction()

    function(accelera_replace_diaguids_in_target _target _diaguids_lib)
        if(NOT TARGET "${_target}")
            return()
        endif()

        set(_props
            INTERFACE_LINK_LIBRARIES
            IMPORTED_LINK_INTERFACE_LIBRARIES
            IMPORTED_LINK_INTERFACE_LIBRARIES_DEBUG
            IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE
            IMPORTED_LINK_INTERFACE_LIBRARIES_RELWITHDEBINFO
            IMPORTED_LINK_INTERFACE_LIBRARIES_MINSIZEREL
        )

        foreach(_prop ${_props})
            get_target_property(_values "${_target}" ${_prop})
            if(_values AND NOT _values STREQUAL "_values-NOTFOUND")
                set(_new_values "")
                set(_changed OFF)
                foreach(_value ${_values})
                    if(_value MATCHES "DIA SDK[/\\\\]lib[/\\\\]amd64[/\\\\]diaguids\\.lib$")
                        list(APPEND _new_values "${_diaguids_lib}")
                        set(_changed ON)
                    else()
                        list(APPEND _new_values "${_value}")
                    endif()
                endforeach()

                if(_changed)
                    set_target_properties("${_target}" PROPERTIES ${_prop} "${_new_values}")
                endif()
            endif()
        endforeach()
    endfunction()

    function(accelera_fix_llvm_dia_sdk_links)
        if(NOT WIN32)
            return()
        endif()

        accelera_find_diaguids(_diaguids_lib)
        if(NOT _diaguids_lib)
            message(FATAL_ERROR
                "LLVM requires diaguids.lib from the Visual Studio DIA SDK, but CMake could not find it. "
                "Install the Visual Studio C++ profiling/tools component that includes the DIA SDK."
            )
        endif()

        get_property(_imported_targets DIRECTORY PROPERTY IMPORTED_TARGETS)
        foreach(_target ${_imported_targets})
            accelera_replace_diaguids_in_target("${_target}" "${_diaguids_lib}")
        endforeach()

        message(STATUS "Using DIA SDK library: ${_diaguids_lib}")
    endfunction()

    function(accelera_find_llvm_clang)
        set(_llvm_prefix "")

        if(WIN32)
            accelera_collect_llvm_dirs("${ACCELERA_LLVM_WINDOWS_ROOT}" _local_llvm_dirs)

            set(_llvm_candidates
                "${ACCELERA_LLVM_WINDOWS_ROOT}"
                ${_local_llvm_dirs}
                "$ENV{ProgramFiles}/LLVM"
                "$ENV{ProgramFiles\(x86\)}/LLVM"
                "$ENV{LOCALAPPDATA}/Programs/LLVM"
            )

            foreach(_candidate ${_llvm_candidates})
                accelera_register_llvm_prefix("${_candidate}")

                accelera_llvm_prefix_is_usable("${_candidate}" _candidate_usable)
                if(_candidate_usable)
                    set(_llvm_prefix "${_candidate}")
                endif()
            endforeach()
        endif()

        find_package(LLVM ${ACCELERA_LLVM_VERSION} QUIET CONFIG)
        if(NOT LLVM_FOUND)
            find_package(LLVM QUIET CONFIG)
        endif()

        find_package(Clang QUIET CONFIG)

        set(_clang_names clang-${ACCELERA_LLVM_VERSION} clang)
        if(LLVM_PACKAGE_VERSION)
            string(REGEX MATCH "^[0-9]+" _llvm_major "${LLVM_PACKAGE_VERSION}")
            if(_llvm_major)
                list(INSERT _clang_names 1 clang-${_llvm_major})
            endif()
        endif()

        find_program(ACCELERA_CLANG_EXECUTABLE NAMES ${_clang_names})

        # Windows fallback:
        # Some LLVM Windows installs have the C++ Clang development headers/libs
        # but CMake cannot load LLVMConfig/ClangConfig.
        if(WIN32 AND NOT LLVM_FOUND AND _llvm_prefix)
            message(WARNING "LLVM CMake config not found, but Clang development files exist. Using manual LLVM paths.")

            set(LLVM_FOUND ON)
            set(Clang_FOUND ON)
            set(LLVM_PACKAGE_VERSION "windows-manual")
            set(LLVM_DEFINITIONS "")
            set(LLVM_INCLUDE_DIRS "${_llvm_prefix}/include")
            set(CLANG_INCLUDE_DIRS "${_llvm_prefix}/include")
            set(LLVM_LIBRARY_DIRS "${_llvm_prefix}/lib")
            set(ACCELERA_CLANG_EXECUTABLE "${_llvm_prefix}/bin/clang.exe")
        endif()

        if(NOT Clang_FOUND AND WIN32 AND _llvm_prefix)
            message(WARNING "Clang CMake config not found, but Clang development files exist. Continuing with manual Clang paths.")
            set(Clang_FOUND ON)
            set(CLANG_INCLUDE_DIRS "${_llvm_prefix}/include")
            set(ACCELERA_CLANG_EXECUTABLE "${_llvm_prefix}/bin/clang.exe")
        endif()

        if(LLVM_FOUND AND Clang_FOUND)
            set(LLVM_AVAILABLE ON PARENT_SCOPE)
            set(LLVM_PACKAGE_VERSION "${LLVM_PACKAGE_VERSION}" PARENT_SCOPE)
            set(LLVM_DEFINITIONS "${LLVM_DEFINITIONS}" PARENT_SCOPE)
            set(LLVM_INCLUDE_DIRS "${LLVM_INCLUDE_DIRS}" PARENT_SCOPE)
            set(CLANG_INCLUDE_DIRS "${CLANG_INCLUDE_DIRS}" PARENT_SCOPE)
            set(LLVM_LIBRARY_DIRS "${LLVM_LIBRARY_DIRS}" PARENT_SCOPE)
            set(ACCELERA_CLANG_EXECUTABLE "${ACCELERA_CLANG_EXECUTABLE}" PARENT_SCOPE)
        endif()
    endfunction()

    function(accelera_try_install_llvm_windows)
        find_program(WINGET_EXECUTABLE winget)

        if(WINGET_EXECUTABLE)
            message(STATUS "LLVM/Clang not found; installing LLVM with winget")
            execute_process(
                COMMAND "${WINGET_EXECUTABLE}" install --id LLVM.LLVM --exact --silent
                        --accept-package-agreements --accept-source-agreements
                RESULT_VARIABLE _winget_result
            )

            message(STATUS "winget LLVM result: ${_winget_result}")
        endif()

        set(_llvm_candidates
            "${ACCELERA_LLVM_WINDOWS_ROOT}"
            "$ENV{ProgramFiles}/LLVM"
            "$ENV{ProgramFiles\(x86\)}/LLVM"
            "$ENV{LOCALAPPDATA}/Programs/LLVM"
        )

        # Search WinGet package install locations too
        file(GLOB _winget_llvm_dirs
            "$ENV{LOCALAPPDATA}/Microsoft/WinGet/Packages/LLVM.LLVM_*"
            "$ENV{ProgramFiles}/WinGet/Packages/LLVM.LLVM_*"
        )

        list(APPEND _llvm_candidates ${_winget_llvm_dirs})

        foreach(_llvm_prefix ${_llvm_candidates})
            message(STATUS "Checking LLVM path: ${_llvm_prefix}")

            accelera_llvm_prefix_is_usable("${_llvm_prefix}" _candidate_usable)
            if(_candidate_usable)
                message(STATUS "Found usable LLVM/Clang development install at: ${_llvm_prefix}")
                accelera_register_llvm_prefix("${_llvm_prefix}")
                return()
            endif()
        endforeach()

        file(MAKE_DIRECTORY "${ACCELERA_LLVM_WINDOWS_ROOT}")
        set(_llvm_archive "${ACCELERA_LLVM_WINDOWS_ROOT}/clang+llvm-${ACCELERA_LLVM_RELEASE}-x86_64-pc-windows-msvc.tar.xz")
        set(_llvm_archive_tmp "${_llvm_archive}.tmp")
        set(_llvm_archive_marker "${_llvm_archive}.complete")
        set(_llvm_url "https://github.com/llvm/llvm-project/releases/download/llvmorg-${ACCELERA_LLVM_RELEASE}/clang+llvm-${ACCELERA_LLVM_RELEASE}-x86_64-pc-windows-msvc.tar.xz")

        if(NOT EXISTS "${_llvm_archive_marker}")
            find_program(CURL_EXECUTABLE curl)
            if(CURL_EXECUTABLE)
                message(STATUS "Downloading/resuming LLVM/Clang development archive with curl: ${_llvm_url}")
                execute_process(
                    COMMAND "${CURL_EXECUTABLE}" -L --fail --retry 5 --retry-delay 2 --continue-at - --output "${_llvm_archive}" "${_llvm_url}"
                    RESULT_VARIABLE _download_result
                )
                if(NOT _download_result EQUAL 0)
                    message(FATAL_ERROR "Failed to download LLVM/Clang development archive with curl. Rerun CMake to resume from the partial file: ${_llvm_archive}")
                endif()
            else()
                message(STATUS "Downloading LLVM/Clang development archive: ${_llvm_url}")
                file(REMOVE "${_llvm_archive_tmp}")
                file(DOWNLOAD
                    "${_llvm_url}"
                    "${_llvm_archive_tmp}"
                    STATUS _download_status
                    SHOW_PROGRESS
                )
                list(GET _download_status 0 _download_result)
                list(GET _download_status 1 _download_message)
                if(NOT _download_result EQUAL 0)
                    file(REMOVE "${_llvm_archive_tmp}")
                    message(FATAL_ERROR "Failed to download LLVM/Clang development archive: ${_download_message}")
                endif()

                file(REMOVE "${_llvm_archive}")
                file(RENAME "${_llvm_archive_tmp}" "${_llvm_archive}")
            endif()

            file(REMOVE "${_llvm_archive_marker}")
            file(WRITE "${_llvm_archive_marker}" "complete\n")
        else()
            message(STATUS "Reusing existing LLVM/Clang development archive: ${_llvm_archive}")
        endif()

        accelera_collect_llvm_dirs("${ACCELERA_LLVM_WINDOWS_ROOT}" _existing_downloaded_llvm_dirs)
        if(NOT _existing_downloaded_llvm_dirs)
            message(STATUS "Extracting LLVM/Clang development archive")
            execute_process(
                COMMAND "${CMAKE_COMMAND}" -E tar xf "${_llvm_archive}"
                WORKING_DIRECTORY "${ACCELERA_LLVM_WINDOWS_ROOT}"
                RESULT_VARIABLE _extract_result
            )
            if(NOT _extract_result EQUAL 0)
                message(FATAL_ERROR "Failed to extract LLVM/Clang development archive")
            endif()
        endif()

        accelera_collect_llvm_dirs("${ACCELERA_LLVM_WINDOWS_ROOT}" _downloaded_llvm_dirs)
        foreach(_llvm_prefix ${_downloaded_llvm_dirs})
            accelera_llvm_prefix_is_usable("${_llvm_prefix}" _candidate_usable)
            if(_candidate_usable)
                message(STATUS "Found usable downloaded LLVM/Clang at: ${_llvm_prefix}")
                accelera_register_llvm_prefix("${_llvm_prefix}")
                return()
            endif()
        endforeach()

        message(FATAL_ERROR "LLVM/Clang was installed or downloaded, but CMake could not find the Clang C++ development headers/libraries")
    endfunction()

    function(accelera_try_install_llvm_linux)
        find_program(BASH_EXECUTABLE bash)
        find_program(SUDO_EXECUTABLE sudo)
        find_program(APT_GET_EXECUTABLE apt-get)

        if(NOT BASH_EXECUTABLE OR NOT APT_GET_EXECUTABLE)
            message(STATUS "Automatic LLVM install skipped: bash or apt-get was not found")
            return()
        endif()

        execute_process(COMMAND id -u OUTPUT_VARIABLE CURRENT_USER_ID ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(CURRENT_USER_ID STREQUAL "0")
            set(_install_command "${BASH_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/shell/install_llvm.sh" "${ACCELERA_LLVM_VERSION}")
        elseif(SUDO_EXECUTABLE)
            execute_process(COMMAND "${SUDO_EXECUTABLE}" -n true RESULT_VARIABLE _sudo_check OUTPUT_QUIET ERROR_QUIET)
            if(_sudo_check EQUAL 0)
                set(_install_command "${SUDO_EXECUTABLE}" "${BASH_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/shell/install_llvm.sh" "${ACCELERA_LLVM_VERSION}")
            else()
                message(STATUS "Automatic LLVM install skipped: sudo requires a password")
                return()
            endif()
        else()
            message(STATUS "Automatic LLVM install skipped: sudo was not found")
            return()
        endif()

        message(STATUS "LLVM/Clang not found; installing LLVM ${ACCELERA_LLVM_VERSION}")
        execute_process(COMMAND ${_install_command} RESULT_VARIABLE _install_result)
        if(NOT _install_result EQUAL 0)
            message(WARNING "Automatic LLVM/Clang installation failed with exit code ${_install_result}")
        endif()
    endfunction()

    function(accelera_register_graphviz_path)
        if(WIN32)
            set(_graphviz_bin_candidates
                "$ENV{ProgramFiles}/Graphviz/bin"
                "$ENV{ProgramFiles\(x86\)}/Graphviz/bin"
                "$ENV{LOCALAPPDATA}/Programs/Graphviz/bin"
            )

            foreach(_graphviz_bin ${_graphviz_bin_candidates})
                if(EXISTS "${_graphviz_bin}/dot.exe")
                    set(ENV{PATH} "${_graphviz_bin};$ENV{PATH}")
                    return()
                endif()
            endforeach()
        endif()
    endfunction()

    function(accelera_try_install_graphviz_windows)
        find_program(WINGET_EXECUTABLE winget)
        if(NOT WINGET_EXECUTABLE)
            message(WARNING "Automatic Graphviz install skipped: winget was not found")
            return()
        endif()

        message(STATUS "Graphviz/dot not found; installing Graphviz with winget")
        execute_process(
            COMMAND "${WINGET_EXECUTABLE}" install --id Graphviz.Graphviz --exact --silent
                    --accept-package-agreements --accept-source-agreements
            RESULT_VARIABLE _winget_result
        )
        message(STATUS "winget Graphviz result: ${_winget_result}")

        accelera_register_graphviz_path()
    endfunction()

    function(accelera_try_install_graphviz_linux)
        find_program(SUDO_EXECUTABLE sudo)
        find_program(APT_GET_EXECUTABLE apt-get)

        if(NOT APT_GET_EXECUTABLE)
            message(WARNING "Automatic Graphviz install skipped: apt-get was not found")
            return()
        endif()

        execute_process(COMMAND id -u OUTPUT_VARIABLE CURRENT_USER_ID ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(CURRENT_USER_ID STREQUAL "0")
            set(_install_command "${APT_GET_EXECUTABLE}" install -y graphviz)
        elseif(SUDO_EXECUTABLE)
            execute_process(COMMAND "${SUDO_EXECUTABLE}" -n true RESULT_VARIABLE _sudo_check OUTPUT_QUIET ERROR_QUIET)
            if(_sudo_check EQUAL 0)
                set(_install_command "${SUDO_EXECUTABLE}" "${APT_GET_EXECUTABLE}" install -y graphviz)
            else()
                message(WARNING "Automatic Graphviz install skipped: sudo requires a password")
                return()
            endif()
        else()
            message(WARNING "Automatic Graphviz install skipped: sudo was not found")
            return()
        endif()

        message(STATUS "Graphviz/dot not found; installing Graphviz")
        execute_process(COMMAND ${_install_command} RESULT_VARIABLE _install_result)
        if(NOT _install_result EQUAL 0)
            message(WARNING "Automatic Graphviz installation failed with exit code ${_install_result}")
        endif()
    endfunction()

    function(accelera_ensure_graphviz)
        accelera_register_graphviz_path()
        find_program(ACCELERA_DOT_EXECUTABLE dot)

        if(NOT ACCELERA_DOT_EXECUTABLE AND ACCELERA_AUTO_INSTALL_GRAPHVIZ)
            if(WIN32)
                accelera_try_install_graphviz_windows()
            elseif(UNIX AND NOT APPLE)
                accelera_try_install_graphviz_linux()
            endif()

            unset(ACCELERA_DOT_EXECUTABLE CACHE)
            find_program(ACCELERA_DOT_EXECUTABLE dot)
        endif()

        if(ACCELERA_DOT_EXECUTABLE)
            message(STATUS "Found Graphviz dot: ${ACCELERA_DOT_EXECUTABLE}")
        else()
            message(WARNING "Graphviz dot was not found; graph-rendering examples may fail")
        endif()
    endfunction()

    message(STATUS "Checking LLVM/Clang for code parallelizer")
    accelera_find_llvm_clang()

    if(NOT LLVM_AVAILABLE AND UNIX AND NOT APPLE AND ACCELERA_AUTO_INSTALL_LLVM)
        accelera_try_install_llvm_linux()
        unset(LLVM_DIR CACHE)
        unset(Clang_DIR CACHE)
        accelera_find_llvm_clang()
    elseif(NOT LLVM_AVAILABLE AND WIN32 AND ACCELERA_AUTO_INSTALL_LLVM)
        accelera_try_install_llvm_windows()
        accelera_find_llvm_clang()
    endif()

    if(LLVM_AVAILABLE)
        message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
        message(STATUS "Found Clang")
        if(ACCELERA_CLANG_EXECUTABLE)
            message(STATUS "Found clang executable: ${ACCELERA_CLANG_EXECUTABLE}")
        else()
            message(WARNING "clang executable was not found; AST bindings can build, but command-line clang tools may be unavailable")
        endif()

        accelera_fix_llvm_dia_sdk_links()
        
        add_definitions(${LLVM_DEFINITIONS})
        add_definitions(-DUSE_CLANG_AST)
        include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
        include_directories(SYSTEM ${CLANG_INCLUDE_DIRS})
        link_directories(${LLVM_LIBRARY_DIRS})
        
        set(CLANG_LIBS
            clangTooling
            clangFrontend
            clangDriver
            clangSerialization
            clangParse
            clangSema
            clangAnalysis
            clangAST
            clangBasic
            clangEdit
            clangLex
            clangRewrite
        )
        set(LLVM_AVAILABLE ON)
    else()
        if(WIN32)
            message(FATAL_ERROR "Cannot continue without LLVM/Clang development files. Automatic Windows installation failed.")
        elseif(UNIX AND NOT APPLE)
            message(FATAL_ERROR "Cannot continue without LLVM/Clang. Install LLVM/Clang or enable non-interactive sudo for automatic installation.")
        else()
            message(STATUS "LLVM/Clang not found - code parallelizer bindings will be disabled")
        endif()
        set(LLVM_AVAILABLE OFF)
    endif()

    message(STATUS "Checking Graphviz/dot for graph-rendering examples")
    accelera_ensure_graphviz()

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

    execute_process(
        COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
        RESULT_VARIABLE PYBIND11_CMAKEDIR_RESULT
        OUTPUT_VARIABLE PYBIND11_CMAKEDIR
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(PYBIND11_CMAKEDIR_RESULT EQUAL 0 AND EXISTS "${PYBIND11_CMAKEDIR}")
        list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKEDIR}")
    endif()

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

    set(UTILS_SOURCES "${CMAKE_SOURCE_DIR}/src/utils/graph_utils.cpp")

    if(LLVM_AVAILABLE)
        file(GLOB_RECURSE CODE_PARALLELIZER_SOURCES 
            "${CMAKE_SOURCE_DIR}/src/ast/*.cpp"
        )
        list(APPEND UTILS_SOURCES "${CMAKE_SOURCE_DIR}/src/utils/code_parallelizer_utils.cpp")
        message(STATUS "Including AST && code parallelizer sources")
    else()
        set(CODE_PARALLELIZER_SOURCES "")
        message(STATUS "⊗ AST disabled (not Linux or LLVM unavailable)")
    endif()

    add_library(accelera_core SHARED
        ${CORE_SOURCES}
        ${NODE_SOURCES}
        ${UTILS_SOURCES}
        ${CODE_PARALLELIZER_SOURCES}
    )

    target_include_directories(accelera_core PUBLIC "${CMAKE_SOURCE_DIR}/include")
    target_link_libraries(accelera_core PUBLIC pybind11::module)

    if(LLVM_AVAILABLE)
        target_link_libraries(accelera_core PUBLIC ${CLANG_LIBS})
        message(STATUS "Linking Clang libraries")
    endif()

    set_target_properties(accelera_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

    target_compile_definitions(accelera_core PUBLIC ACCELERA_BUILD_DLL)

    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/bindings")

    file(GLOB_RECURSE BINDINGS_SOURCES 
        "${CMAKE_SOURCE_DIR}/accelera/bindings/*.cpp"
        "${CMAKE_SOURCE_DIR}/accelera/bindings/**/*.cpp"
    )

    # If we're not on Linux (or LLVM/Clang isn't available), don't build anything related
    # to the code parallelizer bindings.
    if(NOT LLVM_AVAILABLE)
        foreach(_binding_to_remove ${BINDINGS_SOURCES})
            get_filename_component(_binding_name ${_binding_to_remove} NAME_WE)
            if(_binding_name STREQUAL "code_parallelizer_utils")
                list(REMOVE_ITEM BINDINGS_SOURCES ${_binding_to_remove})
                message(STATUS "Skipping code parallelizer binding on this platform: ${_binding_to_remove}")
            endif()
        endforeach()
    endif()

    if(NOT BINDINGS_SOURCES)
        message(STATUS "No Python bindings sources found to build")
    endif()

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

    if(BINDINGS_SOURCES)
        install(DIRECTORY "${CMAKE_BINARY_DIR}/bindings/"
            DESTINATION lib/python/site-packages
            FILES_MATCHING PATTERN "*.so" PATTERN "*.pyd"
        )
    endif()
