#!/bin/bash
set -e

REPO_URL="https://github.com/leejet/stable-diffusion.cpp.git"
REPO_DIR="stable-diffusion.cpp"
BUILD_DIR="sd_build"
OUTPUT_XCFRAMEWORK="StableDiffusion.xcframework"

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

# Extract the public headers we need
mkdir -p "$BUILD_DIR/headers"
cp "$REPO_DIR/include/stable-diffusion.h" "$BUILD_DIR/headers/"

xcodebuild -create-xcframework \
    -library "$BUILD_DIR/merged/ios/libsd_merged.a" -headers "$BUILD_DIR/headers" \
    -library "$BUILD_DIR/merged/sim/libsd_merged.a" -headers "$BUILD_DIR/headers" \
    -output "$OUTPUT_XCFRAMEWORK"

echo "Success! Created $OUTPUT_XCFRAMEWORK"
