import os
import pathlib
import sys

if __name__ == "__main__":
    if len(sys.argv < 2):
        print("please provide the directory in which the files to rename")
        print("usage: python rename.py <absolute_path_to_directory_in_which_files_to_rename>")
        sys.exit(1)

    base_dir = sys.argv[1]

    for filename in os.listdir(base_dir):
        filepath = pathlib.Path(os.path.join(base_dir, filename)).absolute()
        if filepath.suffix == ".tfrecords":
            modified_filepath = filepath.with_suffix(".tfrecord").absolute()
            filepath.rename(modified_filepath)
