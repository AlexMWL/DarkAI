#!/bin/bash
set -e

REPO_URL="https://github.com/leejet/stable-diffusion.cpp.git"
REPO_DIR="stable-diffusion.cpp"
BUILD_DIR="sd_build"
OUTPUT_XCFRAMEWORK="StableDiffusion.xcframework"
# Must match DarkAI.xcodeproj's IPHONEOS_DEPLOYMENT_TARGET. Without this, CMake tags
# every object file with the full iOS SDK version instead of the app's actual minimum
# deployment target, producing a "was built for newer iOS version" linker warning for
# every single object file merged into libsd_merged.a.
#
# Keep this in sync when you change the app's deployment target — a static library built
# for a newer minimum than the app it links into is not just noisy, it can fail to link.
IOS_DEPLOYMENT_TARGET="17.0"

if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning stable-diffusion.cpp..."
    git clone --depth 1 --recursive "$REPO_URL" "$REPO_DIR"
fi

# 2. Build for iOS Device (arm64)
echo "Building for iOS Device (arm64)..."
rm -rf "$BUILD_DIR/ios"
cmake -S "$REPO_DIR" -B "$BUILD_DIR/ios" \
    -DCMAKE_SYSTEM_NAME=iOS \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="$IOS_DEPLOYMENT_TARGET" \
    -DSD_METAL=ON \
    -DSD_BUILD_EXAMPLES=OFF \
    -DSD_BUILD_SHARED_LIBS=OFF
cmake --build "$BUILD_DIR/ios" --config Release -j4

# 3. Build for iOS Simulator (arm64)
echo "Building for iOS Simulator (arm64)..."
rm -rf "$BUILD_DIR/sim"
cmake -S "$REPO_DIR" -B "$BUILD_DIR/sim" \
    -DCMAKE_SYSTEM_NAME=iOS \
    -DCMAKE_OSX_SYSROOT=iphonesimulator \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="$IOS_DEPLOYMENT_TARGET" \
    -DSD_METAL=ON \
    -DSD_BUILD_EXAMPLES=OFF \
    -DSD_BUILD_SHARED_LIBS=OFF
cmake --build "$BUILD_DIR/sim" --config Release -j4

# 4. Create XCFramework
echo "Creating XCFramework..."
rm -rf "$OUTPUT_XCFRAMEWORK"

mkdir -p "$BUILD_DIR/merged/ios"
mkdir -p "$BUILD_DIR/merged/sim"

# Merge all .a libraries in the build tree into one static library per platform
find "$BUILD_DIR/ios" -name "*.a" -exec libtool -static -o "$BUILD_DIR/merged/ios/libsd_merged.a" {} +
find "$BUILD_DIR/sim" -name "*.a" -exec libtool -static -o "$BUILD_DIR/merged/sim/libsd_merged.a" {} +

# Hide this library's ggml/gguf symbols so they can never bind to llama.cpp's copy.
#
# The app links two independent builds of ggml: this one, and the one inside the llama.swift
# package's llama.framework. Both used to export `_ggml_*` and `_gguf_*` as public symbols, so
# the linker resolved stable-diffusion.cpp's calls to whichever it found first — in practice
# llama's.
#
# That is not a cosmetic problem, because the two are compiled with different values of
# GGML_MAX_NAME: stable-diffusion.cpp's CMakeLists sets 160 (see the comment there referencing
# ggml PR #682, which SD needs because its tensor names run to ~83 characters), while llama.cpp
# uses the default 64. `ggml_tensor` embeds `char name[GGML_MAX_NAME]`, so binding across the two
# means SD's C++ reasons about a struct layout that the called code doesn't share. The visible
# symptom was every SD tensor name being truncated to 63 characters by llama's `ggml_set_name`,
# which made the model manager fail to resolve CLIP's weights and abort inside the text encoder.
#
# `ld -r` with `-unexported_symbols_list` demotes these to private extern: still resolvable
# within this library, invisible to the linker for anything else. SD's public entry points
# (`new_sd_ctx`, `generate_image`, `sd_*`) stay exported, which is all the Swift wrapper needs.
# The cost is one extra copy of ggml in the binary; the benefit is that each library actually
# calls the ggml it was compiled against.
hide_ggml_symbols() {
    local lib="$1"
    # "ios" for device, "ios-simulator" for the simulator slice. `ld -r` requires it.
    local platform="$2"
    local work
    work="$(mktemp -d)"

    # Collect every ggml/gguf symbol this archive defines.
    nm -gj "$lib" 2>/dev/null | grep -E '^_(ggml|gguf)' | sort -u > "$work/unexport.txt"
    local count
    count="$(wc -l < "$work/unexport.txt" | tr -d ' ')"
    if [ "$count" -eq 0 ]; then
        echo "  (no ggml symbols found in $(basename "$lib") — skipping)"
        rm -rf "$work"
        return
    fi

    # -all_load so every member is pulled into the relocatable output rather than only those
    # resolving an undefined symbol.
    ld -r -arch arm64 \
       -platform_version "$platform" "$IOS_DEPLOYMENT_TARGET" "$IOS_DEPLOYMENT_TARGET" \
       -all_load "$lib" \
       -unexported_symbols_list "$work/unexport.txt" \
       -o "$work/merged.o"
    libtool -static -o "$lib" "$work/merged.o"

    echo "  hid $count ggml/gguf symbols in $(basename "$lib")"
    rm -rf "$work"
}

echo "Privatising ggml symbols to avoid collision with llama.cpp..."
hide_ggml_symbols "$BUILD_DIR/merged/ios/libsd_merged.a" ios
hide_ggml_symbols "$BUILD_DIR/merged/sim/libsd_merged.a" ios-simulator

# Extract the public headers we need
mkdir -p "$BUILD_DIR/headers"
cp "$REPO_DIR/include/stable-diffusion.h" "$BUILD_DIR/headers/"

xcodebuild -create-xcframework \
    -library "$BUILD_DIR/merged/ios/libsd_merged.a" -headers "$BUILD_DIR/headers" \
    -library "$BUILD_DIR/merged/sim/libsd_merged.a" -headers "$BUILD_DIR/headers" \
    -output "$OUTPUT_XCFRAMEWORK"

echo "Success! Created $OUTPUT_XCFRAMEWORK"
