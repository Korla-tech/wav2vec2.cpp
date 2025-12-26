# wav2vec2 examples

# wav2vec2-cli
set(TARGET wav2vec2-cli)
add_executable(${TARGET} wav2vec2-cli.cpp)

include(DefaultTargetOptions)

target_link_libraries(${TARGET} PRIVATE common wav2vec2 ${CMAKE_THREAD_LIBS_INIT})

# quantize-wav2vec2
set(TARGET quantize-wav2vec2)
add_executable(${TARGET} quantize-wav2vec2.cpp)

include(DefaultTargetOptions)

target_link_libraries(${TARGET} PRIVATE common wav2vec2 ${CMAKE_THREAD_LIBS_INIT})
