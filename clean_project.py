import re

project_path = "/Users/vmware/Desktop/DarkAI/DarkAI.xcodeproj/project.pbxproj"

# We will read the clean original project file from git if possible? No, we don't have git repository.
# Let's read the file as text, and filter out all added sections first to revert to original clean layout, then apply the correct edits.

with open(project_path, 'r') as file:
    content = file.read()

# Revert previous script changes:
# Remove all custom blocks
content = re.sub(r'/\* Begin XCRemoteSwiftPackageReference section \*/.*?/\* End XCRemoteSwiftPackageReference section \*/\n*', '', content, flags=re.DOTALL)
content = re.sub(r'/\* Begin XCSwiftPackageProductDependency section \*/.*?/\* End XCSwiftPackageProductDependency section \*/\n*', '', content, flags=re.DOTALL)

# Revert native target dependencies
content = re.sub(r'packageProductDependencies\s*=\s*\(\s*.*?\s*\);', 'packageProductDependencies = (\n\t\t\t);', content, flags=re.DOTALL)

# Revert packageReferences in PBXProject
content = re.sub(r'packageReferences\s*=\s*\(\s*.*?\s*\);', '', content, flags=re.DOTALL)

# Revert duplicate SWIFT_CXX_INTEROPERABILITY_MODE
content = re.sub(r'\t+SWIFT_CXX_INTEROPERABILITY_MODE\s*=\s*\w+;\n*', '', content)

# Now apply clean modifications:

# 1. Add packageReferences in PBXProject
targets_match = re.search(r'targets\s*=\s*\(\s*D5B592AC2FF1D16900788D64\s*/\*\s*DarkAI\s*\*/,\s*\);', content)
if targets_match:
    content = content.replace(targets_match.group(0), targets_match.group(0) + "\n\t\t\tpackageReferences = (\n\t\t\t\tD5B592C52FF1D16B00788D64 /* XCRemoteSwiftPackageReference \"llama.swift\" */,\n\t\t\t);")

# 2. Add packageProductDependencies in PBXNativeTarget
content = content.replace("packageProductDependencies = (\n\t\t\t);", "packageProductDependencies = (\n\t\t\t\tD5B592C62FF1D16B00788D64 /* llama */,\n\t\t\t);")

# 3. Add sections before the closing line `rootObject = D5B592A52FF1D16900788D64 /* Project object */;`
root_obj_match = re.search(r'rootObject = D5B592A52FF1D16900788D64 /\* Project object \*/;', content)
if root_obj_match:
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
    content = content.replace(root_obj_match.group(0), insertion + root_obj_match.group(0))

# 4. Enable SWIFT_CXX_INTEROPERABILITY_MODE = default under both configurations
build_settings_blocks = re.findall(r'buildSettings\s*=\s*\{([^\}]*)\};', content)
for block in build_settings_blocks:
    if "PRODUCT_BUNDLE_IDENTIFIER" in block or "SDKROOT" in block:
        # Check if already has SWIFT_CXX_INTEROPERABILITY_MODE
        if "SWIFT_CXX_INTEROPERABILITY_MODE" not in block:
            new_block = block + "\t\t\t\tSWIFT_CXX_INTEROPERABILITY_MODE = default;\n"
            content = content.replace(block, new_block)

with open(project_path, 'w') as file:
    file.write(content)

print("Successfully cleaned and re-applied dependencies with lowercase 'llama' product!")
