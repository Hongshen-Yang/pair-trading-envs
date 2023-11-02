'''
Because I have modified some packages from PYPI to fix some outdate bugs
I want to move all my packages under \libs folder, so that other users can use it as well
The code is generated from ChatGPT
'''

import shutil
import os

# Create a directory called 'libs' if it doesn't exist
libs_directory = "libs"
if not os.path.exists(libs_directory):
    os.makedirs(libs_directory)

# Read the packages from the requirements.txt file
with open("requirements.txt", "r") as req_file:
    packages = req_file.readlines()

for package in packages:
    # Parse the package name and file path
    if "@ file://" in package:
        parts = package.strip().split("@ file://")
        package_name = parts[0]
        package_path = parts[1]
        source_path = os.path.expanduser(package_path)
        destination_path = os.path.join(libs_directory, package_name)

        if os.path.exists(source_path):
            shutil.move(source_path, destination_path)
            print(f"Moved {package_name} to {libs_directory}")
        else:
            print(f"{package_name} not found in {source_path}")
    else:
        print(f"Skipping non-file requirement: {package}")

# Optionally, you can generate or update your requirements.txt with the paths to the custom packages in the "libs" directory.
