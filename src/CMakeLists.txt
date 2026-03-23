# TODO: should not use this
if (WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)

    if (BUILD_SHARED_LIBS)
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
endif()

if (WAV2VEC2_ALL_WARNINGS)
    if (NOT MSVC)
        list(APPEND WARNING_FLAGS -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function)
        list(APPEND C_FLAGS       -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes
                                  -Werror=implicit-int -Werror=implicit-function-declaration)
        list(APPEND CXX_FLAGS     -Wmissing-declarations -Wmissing-noreturn)

        list(APPEND C_FLAGS   ${WARNING_FLAGS})
        list(APPEND CXX_FLAGS ${WARNING_FLAGS})

        add_compile_options("$<$<COMPILE_LANGUAGE:C>:${C_FLAGS}>"
                            "$<$<COMPILE_LANGUAGE:CXX>:${CXX_FLAGS}>")
    else()
        set(C_FLAGS   "")
        set(CXX_FLAGS "")
    endif()
endif()

#
# wav2vec2 library
#

add_library(wav2vec2
            ../include/wav2vec2.h
            wav2vec2-arch.h
            wav2vec2.cpp
            )

set_target_properties(wav2vec2 PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${SOVERSION}
)

target_include_directories(wav2vec2 PUBLIC . ../include)
target_compile_features   (wav2vec2 PUBLIC cxx_std_11)

find_package(Threads REQUIRED)
target_link_libraries(wav2vec2 PUBLIC ggml Threads::Threads)

if (BUILD_SHARED_LIBS)
    set_target_properties(wav2vec2 PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(wav2vec2 PRIVATE WAV2VEC2_SHARED WAV2VEC2_BUILD)
endif()

#
# wav2vec2 library
#

add_library(wav2vec2-bert
            ../include/wav2vec2.h
            wav2vec2bert-arch.h
            wav2vec2bert.cpp
            )

set_target_properties(wav2vec2-bert PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${SOVERSION}
)

target_include_directories(wav2vec2-bert PUBLIC . ../include)
target_compile_features   (wav2vec2-bert PUBLIC cxx_std_11)

find_package(Threads REQUIRED)
target_link_libraries(wav2vec2-bert PUBLIC ggml Threads::Threads)

if (BUILD_SHARED_LIBS)
    set_target_properties(wav2vec2-bert PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(wav2vec2-bert PRIVATE WAV2VEC2_SHARED WAV2VEC2_BUILD)
endif()
