import os
import numpy as np
from preprocess import get_preprocessed_data, interpolate, calculate_velocity
from multiprocessing import Pool
from matplotlib import pyplot as plt

def load_data_points(directory, num_points=0, name=None):
    if name is not None:
        file_path = os.path.join(directory, name)
        if os.path.isfile(file_path):
            return [np.load(file_path, allow_pickle=True)]
        else:
            raise FileNotFoundError(f"File '{name}' not found in directory '{directory}'.")

    files = sorted([f for f in os.listdir(directory)])
    data_points = []
    for file in files[:num_points]:
        file_path = os.path.join(directory, file)
        data = np.load(file_path, allow_pickle=True)
        data_points.append(data)
    return data_points


def process_file(name):
    print(name)
    # Unpack file name parts
    parts = name.split("_")
    person = int(parts[0])
    character = int(parts[1])
    finger = parts[2]
    glyph = int(parts[3])

    # Load original data
    X1, Y1, T, _, _, _ = get_preprocessed_data(person, character, n_points=None, nfs=None,
                                               smoothing_window=3, smooth_poly=2,
                                               finger=finger, glyph=glyph)

    # Load the saved data sample
    data_sample = load_data_points(current_directory, name=name)[0]

    # Interpolate to the original number of points
    X = [interpolate(x, n_points=len(X1[0]), interp="cubic") for x in data_sample["X1"]]
    Y = [interpolate(y, n_points=len(X1[0]), interp="cubic") for y in data_sample["Y1"]]
    T = [interpolate(t, n_points=len(X1[0]), interp="linear") for t in T]
    V = calculate_velocity(X, Y, T)

    # Interpolate again to the target number of points
    X = [interpolate(x, n_points=n_points, interp="cubic") for x in data_sample["X1"]]
    Y = [interpolate(y, n_points=n_points, interp="cubic") for y in data_sample["Y1"]]
    T = [interpolate(t, n_points=n_points, interp="linear") for t in T]
    V = [interpolate(v, n_points=n_points, interp="cubic") for v in V]

    # Save corrected data
    output_dir = f"{current_directory}_corrected"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{name}")
    np.savez(file_path, X=np.array(X, dtype=object), Y=np.array(Y, dtype=object),
             T=np.array(T, dtype=object), V=np.array(V, dtype=object))


directory_original = "synthetic/original"
directory_manipulated = "synthetic/manipulated"
directory_represented = "synthetic/represented"
directory_manipulated_2_modes = "synthetic/manipulated_2_modes"
directory_edge_data = "synthetic/second mode edge data"
directories = [directory_original, directory_manipulated, directory_represented, directory_manipulated_2_modes, directory_edge_data]
current_directory = directory_original
n_points = 175

# Get list of files in the current directory

if __name__ == "__main__":
    files = [f"{directory_original}/{f}" for f in os.listdir(directory_original) if os.path.isfile(os.path.join(directory_original, f))]
    for f in files:
        with np.load(f, allow_pickle=True) as sample:
            for i in range(len(sample["X"])):
                plt.plot(sample["X"][i], sample["Y"][i])
                plt.axis("equal")
                plt.figure(2)
                plt.plot(sample["T"][i], sample["V"][i])
                plt.show()
