cmake_minimum_required(VERSION 3.14)
project("wav2vec2.cpp" C CXX)
project("wav2vec2.cpp" VERSION 0.1.0)
include(CheckIncludeFileCXX)

set(SOVERSION 1)

set(CMAKE_WARN_UNUSED_CLI YES)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Add path to modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(WAV2VEC2_STANDALONE ON)
    include(git-vars)
else()
    set(WAV2VEC2_STANDALONE OFF)
endif()

if (EMSCRIPTEN)
    set(BUILD_SHARED_LIBS_DEFAULT OFF)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -pthread")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s TOTAL_STACK=5242880")
else()
    if (MINGW)
        set(BUILD_SHARED_LIBS_DEFAULT OFF)
    else()
        set(BUILD_SHARED_LIBS_DEFAULT ON)
    endif()
endif()

option(BUILD_SHARED_LIBS "build shared libraries" ${BUILD_SHARED_LIBS_DEFAULT})

#
# option list
#

# debug
option(WAV2VEC2_ALL_WARNINGS           "wav2vec2: enable all compiler warnings"                   ON)
option(WAV2VEC2_ALL_WARNINGS_3RD_PARTY "wav2vec2: enable all compiler warnings in 3rd party libs" OFF)

# build
option(WAV2VEC2_FATAL_WARNINGS  "wav2vec2: enable -Werror flag"               OFF)

# sanitizers
option(WAV2VEC2_SANITIZE_THREAD    "wav2vec2: enable thread sanitizer"    OFF)
option(WAV2VEC2_SANITIZE_ADDRESS   "wav2vec2: enable address sanitizer"   OFF)
option(WAV2VEC2_SANITIZE_UNDEFINED "wav2vec2: enable undefined sanitizer" OFF)

# extra artifacts
option(WAV2VEC2_BUILD_TESTS    "wav2vec2: build tests"    ${WAV2VEC2_STANDALONE})
option(WAV2VEC2_BUILD_EXAMPLES "wav2vec2: build examples" ${WAV2VEC2_STANDALONE})

# Required for relocatable CMake package
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/build-info.cmake)

# override ggml options
set(GGML_SANITIZE_THREAD    ${WAV2VEC2_SANITIZE_THREAD})
set(GGML_SANITIZE_ADDRESS   ${WAV2VEC2_SANITIZE_ADDRESS})
set(GGML_SANITIZE_UNDEFINED ${WAV2VEC2_SANITIZE_UNDEFINED})
set(GGML_ALL_WARNINGS       ${WAV2VEC2_ALL_WARNINGS})
set(GGML_FATAL_WARNINGS     ${WAV2VEC2_FATAL_WARNINGS})

#
# build the library
#

if (NOT TARGET ggml)
    add_subdirectory(ggml)
    if(WIN32)
        target_compile_definitions(ggml-base PRIVATE _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
    endif()
endif()

add_subdirectory(src)

#
# install
#

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(WAV2VEC2_BUILD_NUMBER        ${BUILD_NUMBER})
set(WAV2VEC2_BUILD_COMMIT        ${BUILD_COMMIT})
set(WAV2VEC2_INSTALL_VERSION     ${CMAKE_PROJECT_VERSION})

set(WAV2VEC2_INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR} CACHE PATH "Location of header  files")
set(WAV2VEC2_LIB_INSTALL_DIR     ${CMAKE_INSTALL_LIBDIR}     CACHE PATH "Location of library files")
set(WAV2VEC2_BIN_INSTALL_DIR     ${CMAKE_INSTALL_BINDIR}     CACHE PATH "Location of binary  files")

set_target_properties(wav2vec2 PROPERTIES PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/wav2vec2.h)
install(TARGETS wav2vec2 LIBRARY PUBLIC_HEADER)

target_compile_definitions(wav2vec2 PRIVATE
    WAV2VEC2_VERSION="${PROJECT_VERSION}"
)

#
# programs, examples and tests
#

if (WAV2VEC2_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if (MSVC)
    set(MSVC_WARNING_FLAGS
        /wd4101
        /wd4005
        /wd4065
        /wd4267
        /wd4244
        /wd4805
        /wd4305
        /wd4996
    )
    function(disable_msvc_warnings target_name)
        if(TARGET ${target_name})
            target_compile_options(${target_name} PRIVATE ${MSVC_WARNING_FLAGS})
        endif()
    endfunction()

    if (WAV2VEC2_BUILD_EXAMPLES)
        disable_msvc_warnings(wav2vec2)
        disable_msvc_warnings(common)
        disable_msvc_warnings(wav2vec2-cli)
        disable_msvc_warnings(quantize-wav2vec2)
    endif()
endif()
