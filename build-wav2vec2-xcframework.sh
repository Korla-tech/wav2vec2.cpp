#!/bin/bash
#
# Build wav2vec2.xcframework for iOS
# Based on whisper.cpp's build-xcframework.sh
#

set -e

IOS_MIN_OS_VERSION=16.4

BUILD_SHARED_LIBS=OFF
WAV2VEC2_BUILD_EXAMPLES=OFF
WAV2VEC2_BUILD_TESTS=OFF
GGML_METAL=ON
GGML_METAL_EMBED_LIBRARY=ON
GGML_BLAS_DEFAULT=ON
GGML_METAL_USE_BF16=ON
GGML_OPENMP=OFF

COMMON_C_FLAGS="-Wno-macro-redefined -Wno-shorten-64-to-32 -Wno-unused-command-line-argument -g"
COMMON_CXX_FLAGS="-Wno-macro-redefined -Wno-shorten-64-to-32 -Wno-unused-command-line-argument -g"

COMMON_CMAKE_ARGS=(
    -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED=NO
    -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY=""
    -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED=NO
    -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}
    -DWAV2VEC2_BUILD_EXAMPLES=${WAV2VEC2_BUILD_EXAMPLES}
    -DWAV2VEC2_BUILD_TESTS=${WAV2VEC2_BUILD_TESTS}
    -DGGML_METAL_EMBED_LIBRARY=${GGML_METAL_EMBED_LIBRARY}
    -DGGML_BLAS_DEFAULT=${GGML_BLAS_DEFAULT}
    -DGGML_METAL=${GGML_METAL}
    -DGGML_METAL_USE_BF16=${GGML_METAL_USE_BF16}
    -DGGML_NATIVE=OFF
    -DGGML_OPENMP=${GGML_OPENMP}
)

echo "Building wav2vec2.xcframework for iOS..."
echo ""

# Clean up previous builds
rm -rf build-wav2vec2-ios-sim
rm -rf build-wav2vec2-ios-device
rm -rf build-wav2vec2-apple
rm -rf wav2vec2.xcframework

# Configure iOS simulator
echo "Configuring iOS simulator..."
cmake -B build-wav2vec2-ios-sim -G Xcode \
    "${COMMON_CMAKE_ARGS[@]}" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=${IOS_MIN_OS_VERSION} \
    -DIOS=ON \
    -DCMAKE_SYSTEM_NAME=iOS \
    -DCMAKE_OSX_SYSROOT=iphonesimulator \
    -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64" \
    -DCMAKE_XCODE_ATTRIBUTE_SUPPORTED_PLATFORMS=iphonesimulator \
    -DCMAKE_C_FLAGS="${COMMON_C_FLAGS}" \
    -DCMAKE_CXX_FLAGS="${COMMON_CXX_FLAGS}" \
    -S . > /dev/null 2>&1

# Configure iOS device
echo "Configuring iOS device..."
cmake -B build-wav2vec2-ios-device -G Xcode \
    "${COMMON_CMAKE_ARGS[@]}" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=${IOS_MIN_OS_VERSION} \
    -DCMAKE_OSX_SYSROOT=iphoneos \
    -DCMAKE_OSX_ARCHITECTURES="arm64" \
    -DCMAKE_XCODE_ATTRIBUTE_SUPPORTED_PLATFORMS=iphoneos \
    -DCMAKE_C_FLAGS="${COMMON_C_FLAGS}" \
    -DCMAKE_CXX_FLAGS="${COMMON_CXX_FLAGS}" \
    -S . > /dev/null 2>&1

# Build iOS simulator
echo "Building iOS simulator..."
cmake --build build-wav2vec2-ios-sim --config Release --target wav2vec2 ggml -- -quiet

# Build iOS device
echo "Building iOS device..."
cmake --build build-wav2vec2-ios-device --config Release --target wav2vec2 ggml -- -quiet

# Create framework structure
echo "Creating framework structure..."

create_framework() {
    local build_dir=$1
    local platform=$2
    local archs=$3
    local sdk=$4
    local min_version_flag=$5
    local release_suffix=$6

    local framework_dir="${build_dir}/wav2vec2.framework"
    mkdir -p "${framework_dir}"

    # Combine static libraries (Xcode uses Release-iphoneos/Release-iphonesimulator)
    local libs=(
        "${build_dir}/src/${release_suffix}/libwav2vec2.a"
        "${build_dir}/ggml/src/${release_suffix}/libggml.a"
        "${build_dir}/ggml/src/${release_suffix}/libggml-base.a"
        "${build_dir}/ggml/src/${release_suffix}/libggml-cpu.a"
        "${build_dir}/ggml/src/ggml-metal/${release_suffix}/libggml-metal.a"
        "${build_dir}/ggml/src/ggml-blas/${release_suffix}/libggml-blas.a"
    )

    # Check which libraries exist
    local existing_libs=()
    for lib in "${libs[@]}"; do
        if [[ -f "$lib" ]]; then
            existing_libs+=("$lib")
            echo "  Found: $lib"
        else
            echo "  Missing: $lib (skipping)"
        fi
    done

    # Combine with libtool
    libtool -static -o "${build_dir}/combined.a" "${existing_libs[@]}" 2>/dev/null

    # Create dynamic library
    local arch_flags=""
    for arch in $archs; do
        arch_flags+=" -arch $arch"
    done

    xcrun clang++ \
        -dynamiclib \
        -isysroot $(xcrun --sdk ${sdk} --show-sdk-path) \
        ${arch_flags} \
        ${min_version_flag} \
        -install_name @rpath/wav2vec2.framework/wav2vec2 \
        -Wl,-all_load "${build_dir}/combined.a" \
        -framework Foundation \
        -framework Metal \
        -framework Accelerate \
        -o "${framework_dir}/wav2vec2"

    # Create Info.plist
    cat > "${framework_dir}/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>wav2vec2</string>
    <key>CFBundleIdentifier</key>
    <string>org.wav2vec2cpp.wav2vec2</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>wav2vec2</string>
    <key>CFBundlePackageType</key>
    <string>FMWK</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>MinimumOSVersion</key>
    <string>${IOS_MIN_OS_VERSION}</string>
</dict>
</plist>
EOF

    # Copy headers
    mkdir -p "${framework_dir}/Headers"
    cp include/wav2vec2.h "${framework_dir}/Headers/"
    cp src/wav2vec2-arch.h "${framework_dir}/Headers/" 2>/dev/null || true

    # Copy ggml headers (required by wav2vec2.h)
    cp ggml/include/ggml.h "${framework_dir}/Headers/"
    cp ggml/include/ggml-alloc.h "${framework_dir}/Headers/" 2>/dev/null || true
    cp ggml/include/ggml-backend.h "${framework_dir}/Headers/" 2>/dev/null || true
    cp ggml/include/ggml-cpu.h "${framework_dir}/Headers/" 2>/dev/null || true
    cp ggml/include/ggml-metal.h "${framework_dir}/Headers/" 2>/dev/null || true

    # Create module map
    mkdir -p "${framework_dir}/Modules"
    cat > "${framework_dir}/Modules/module.modulemap" << EOF
framework module wav2vec2 {
    header "ggml.h"
    header "ggml-alloc.h"
    header "ggml-backend.h"
    header "wav2vec2.h"
    export *
}
EOF
}

echo ""
echo "Creating iOS simulator framework..."
create_framework "build-wav2vec2-ios-sim" "iphonesimulator" "arm64 x86_64" "iphonesimulator" "-mios-simulator-version-min=${IOS_MIN_OS_VERSION}" "Release-iphonesimulator"

echo ""
echo "Creating iOS device framework..."
create_framework "build-wav2vec2-ios-device" "iphoneos" "arm64" "iphoneos" "-mios-version-min=${IOS_MIN_OS_VERSION}" "Release-iphoneos"

# Create xcframework
echo ""
echo "Creating wav2vec2.xcframework..."
xcodebuild -create-xcframework \
    -framework build-wav2vec2-ios-sim/wav2vec2.framework \
    -framework build-wav2vec2-ios-device/wav2vec2.framework \
    -output wav2vec2.xcframework

echo ""
echo "Done! Created wav2vec2.xcframework"
echo ""
ls -la wav2vec2.xcframework/
echo ""
echo "To use in your iOS project:"
echo "  1. Drag wav2vec2.xcframework into your Xcode project"
echo "  2. Add to 'Frameworks, Libraries, and Embedded Content'"
echo "  3. Set 'Embed' to 'Embed & Sign'"
echo ""
echo "Don't forget to also include the Q4_K model:"
echo "  models/wav2vec2-phoneme/ggml-model-q4_k.bin (204 MB)"
