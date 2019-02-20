import numpy as np

FILE_NAMES = ["training_set", "training_labels", "test_set", "test_labels"]

# Loads pickle objects
def obj_loader(name):
    with open(f"{name}.P", mode="rb") as pickle_file:
        print(f"Reading {name}.P.")
        return np.load(pickle_file)

# Pickle dump the image arrays
file_object = {name : obj_loader(name) for name in FILE_NAMES}

if __name__ == "__main__":
    print(file_object)
    print(file_object["training_set"].shape)