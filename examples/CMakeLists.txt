# dependencies

find_package(Threads REQUIRED)

# common library

set(TARGET common)

add_library(${TARGET} STATIC
    common.h
    common.cpp
    common-ggml.h
    common-ggml.cpp
    )

include(DefaultTargetOptions)

target_link_libraries(${TARGET} PRIVATE wav2vec2 ${CMAKE_DL_LIBS})

set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(${TARGET} PROPERTIES FOLDER "libs")

# examples

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if (NOT EMSCRIPTEN AND NOT CMAKE_JS_VERSION)
    add_subdirectory(wav2vec2)
endif()

# common-bert library

set(TARGET common-bert)

add_library(${TARGET} STATIC
    common.h
    common.cpp
    common-ggml.h
    common-ggml.cpp
    )

include(DefaultTargetOptions)

target_link_libraries(${TARGET} PRIVATE wav2vec2-bert ${CMAKE_DL_LIBS})

set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(${TARGET} PROPERTIES FOLDER "libs")

# examples

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
