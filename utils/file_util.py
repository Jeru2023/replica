import os
import pkg_resources


def get_root_path():
    package_path = pkg_resources.resource_filename(__name__, "")
    parent_path = os.path.dirname(package_path)
    return parent_path


def get_file_paths(directory):
    file_paths = []

    # iterate all files in the directory
    for root, directories, files in os.walk(directory):
        for file in files:
            # construct full path
            file_path = os.path.join(root, file)
            file_paths.append(file_path)

    return file_paths

