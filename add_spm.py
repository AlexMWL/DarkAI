import re

project_path = "/Users/vmware/Desktop/DarkAI/DarkAI.xcodeproj/project.pbxproj"

with open(project_path, 'r') as file:
    content = file.read()

# 1. Add packageReferences to PBXProject targets section
targets_match = re.search(r'targets\s*=\s*\([^\)]*D5B592AC2FF1D16900788D64\s*/\*\s*DarkAI\s*\*/[^\)]*\);', content)
if targets_match:
    targets_block = targets_match.group(0)
    package_ref_block = targets_block + "\n\t\t\tpackageReferences = (\n\t\t\t\tD5B592C52FF1D16B00788D64 /* XCRemoteSwiftPackageReference \"llama.swift\" */,\n\t\t\t);"
    content = content.replace(targets_block, package_ref_block)

# 2. Add packageProductDependencies to PBXNativeTarget section
native_target_match = re.search(r'packageProductDependencies\s*=\s*\(\s*\);', content)
if native_target_match:
    dependencies_block = native_target_match.group(0)
    new_dependencies = "packageProductDependencies = (\n\t\t\t\tD5B592C62FF1D16B00788D64 /* llama */,\n\t\t\t);"
    content = content.replace(dependencies_block, new_dependencies)

# 3. Add the XCRemoteSwiftPackageReference and XCSwiftPackageProductDependency sections
# We will insert them right before /* End XCBuildConfiguration section */
xcbuild_cfg_match = re.search(r'/\* End XCBuildConfiguration section \*/', content)
if xcbuild_cfg_match:
    insertion = """/* Begin XCRemoteSwiftPackageReference section */
		D5B592C52FF1D16B00788D64 /* XCRemoteSwiftPackageReference "llama.swift" */ = {
			isa = XCRemoteSwiftPackageReference;
			repositoryURL = "https://github.com/mattt/llama.swift.git";
			requirement = {
				kind = upToNextMajorVersion;
				minimumVersion = "2.9692.0";
			};
		};
/* End XCRemoteSwiftPackageReference section */

/* Begin XCSwiftPackageProductDependency section */
		D5B592C62FF1D16B00788D64 /* llama */ = {
			isa = XCSwiftPackageProductDependency;
			package = D5B592C52FF1D16B00788D64 /* XCRemoteSwiftPackageReference "llama.swift" */;
			productName = llama;
		};
/* End XCSwiftPackageProductDependency section */

"""
    content = content.replace(xcbuild_cfg_match.group(0), insertion + xcbuild_cfg_match.group(0))

# 4. Enable SWIFT_CXX_INTEROPERABILITY_MODE = default under buildSettings in XCBuildConfiguration section
# Let's search for buildSettings blocks and add it.
build_settings_blocks = re.findall(r'buildSettings\s*=\s*\{([^\}]*)\};', content)
for block in build_settings_blocks:
    if "PRODUCT_BUNDLE_IDENTIFIER" in block or "SDKROOT" in block:
        # Check if already has SWIFT_CXX_INTEROPERABILITY_MODE
        if "SWIFT_CXX_INTEROPERABILITY_MODE" not in block:
            new_block = block + "\t\t\t\tSWIFT_CXX_INTEROPERABILITY_MODE = default;\n"
            content = content.replace(block, new_block)

with open(project_path, 'w') as file:
    file.write(content)

print("Successfully injected Swift Package Manager dependencies and C++ Interop flags!")
