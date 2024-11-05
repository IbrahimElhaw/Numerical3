import itertools
import math
import os
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import pickle
from scipy.stats import wasserstein_distance


def calculate_curve_length(X_, Y_):
    total_length = 0.0
    for stroke_x, stroke_y in zip(X_, Y_):
        stroke_length = 0.0
        for i in range(1, len(stroke_x)):
            dx = stroke_x[i] - stroke_x[i - 1]
            dy = stroke_y[i] - stroke_y[i - 1]
            distance = np.sqrt(dx ** 2 + dy ** 2)
            stroke_length += distance
        total_length += stroke_length
    return total_length


def filter_data(X_, Y_, T_, V_, infos_, number=None, gender=None):
    if number in range(10):
        X_ = [x for x, info in zip(X_, infos_) if info[1] == str(number)]
        Y_ = [y for y, info in zip(Y_, infos_) if info[1] == str(number)]
        T_ = [t for t, info in zip(T_, infos_) if info[1] == str(number)]
        V_ = [v for v, info in zip(V_, infos_) if info[1] == str(number)]
        infos_ = infos_[infos_[:, 1] == str(number)]
    if gender in [0, 1]:
        X_ = [x for x, info in zip(X_, infos_) if info[4] == str(gender)]
        Y_ = [y for y, info in zip(Y_, infos_) if info[4] == str(gender)]
        T_ = [t for t, info in zip(T_, infos_) if info[4] == str(gender)]
        V_ = [v for v, info in zip(V_, infos_) if info[4] == str(gender)]
        infos_ = infos_[infos_[:, 1] == str(gender)]
    return X_, Y_, T_, V_, infos_


def load_and_save_files(current_dir):
    X, Y, T, V, infos = [], [], [], [], []

    files = [os.path.join(current_dir, f) for f in os.listdir(current_dir) if
             os.path.isfile(os.path.join(current_dir, f))]
    for f in files:
        with np.load(f, allow_pickle=True) as sample:
            X.append(sample["X"])
            Y.append(sample["Y"])
            T.append(sample["T"])
            V.append(sample["V"])
            infos.append(sample["infos"])
    place=current_dir.split("/")[1]
    with open(rf"arranged\{place}\X.pkl", "wb") as f:
        pickle.dump(X, f)
    with open(rf"arranged\{place}\Y.pkl", "wb") as f:
        pickle.dump(Y, f)
    with open(rf"arranged\{place}\T.pkl", "wb") as f:
        pickle.dump(T, f)
    with open(fr"arranged\{place}\V.pkl", "wb") as f:
        pickle.dump(V, f)
    with open(rf"arranged\{place}\infos.pkl", "wb") as f:
        pickle.dump(infos, f)
    return X, Y, T, V, np.array(infos)


def load_data(path):
    with open(f"{path}/X.pkl", "rb") as f:
        X = pickle.load(f)
    with open(f"{path}/Y.pkl", "rb") as f:
        Y = pickle.load(f)
    with open(f"{path}/T.pkl", "rb") as f:
        T = pickle.load(f)
    with open(f"{path}/V.pkl", "rb") as f:
        V = pickle.load(f)
    with open(f"{path}/infos.pkl", "rb") as f:
        infos = pickle.load(f)
    infos = np.array(infos)
    return X, Y, T, V, infos


def filter_outliers(data):
    Q1 = np.percentile(data, 20)
    Q3 = np.percentile(data, 85)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return [x for x in data if lower_bound <= x <= upper_bound]


def plt_length_distribution(Xr_, Yr_, Xo_, Yo_):
    lengths_r = [calculate_curve_length(x, y) for x, y in zip(Xr_, Yr_)]
    lengths_o = [calculate_curve_length(x, y) for x, y in zip(Xo_, Yo_)]

    distance = wasserstein_distance(lengths_r, lengths_o)
    print(f"Wasserstein Distance: {distance}")


    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 4)
    sns.histplot(lengths_o, bins=30, color='blue', kde=True, label='Original')
    sns.histplot(lengths_r, bins=30, color='orange', kde=True, label='Represented')
    plt.title("Histogram of Curve Lengths")
    plt.xlabel("Curve Length")
    plt.ylabel("Frequency")
    plt.legend()

    # plt.subplot(2, 2, 2)
    # sns.kdeplot(lengths_o, bw_adjust=0.5, color='blue', label='Original')
    # sns.kdeplot(lengths_r, bw_adjust=0.5, color='orange', label='Represented')
    # plt.title("KDE Plot of Curve Lengths")
    # plt.xlabel("Curve Length")
    # plt.legend()
    #
    # plt.subplot(2, 2, 3)
    # sns.boxplot(data=[lengths_o, lengths_r], orient="h")
    # plt.yticks([0, 1], ['Original', 'Represented'])
    # plt.title("Box Plot of Curve Lengths")
    # plt.xlabel("Curve Length")
    #
    # plt.subplot(2, 2, 4)
    # sns.violinplot(data=[lengths_o, lengths_r], orient="h")
    # plt.yticks([0, 1], ['Original', 'Represented'])
    # plt.title("Violin Plot of Curve Lengths")
    # plt.xlabel("Curve Length")

    # Show plots
    plt.tight_layout()
    # plt.show()
    return distance


def plt_V_distribution(Vo_, Vr_):
    # Flatten the velocity lists
    flattened_vo = list(itertools.chain.from_iterable(itertools.chain.from_iterable(Vo_)))
    flattened_vr = list(itertools.chain.from_iterable(itertools.chain.from_iterable(Vr_)))

    distance = wasserstein_distance(flattened_vo, flattened_vr)
    print(f"Wasserstein Distance: {distance}")

    flattened_vo = filter_outliers(flattened_vo)
    flattened_vr = filter_outliers(flattened_vr)

    # plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 3)
    sns.histplot(flattened_vo, bins=30, color='blue', kde=True, label='Original Velocities', alpha=0.6)
    sns.histplot(flattened_vr, bins=30, color='orange', kde=True, label='Represented Velocities', alpha=0.6)
    plt.title("Histogram of Velocities")
    plt.xlabel("Velocity")
    plt.ylabel("Frequency")
    plt.legend()

    # plt.subplot(2, 2, 2)
    # sns.kdeplot(flattened_vo, bw_adjust=0.5, color='blue', label='Original Velocities')
    # sns.kdeplot(flattened_vr, bw_adjust=0.5, color='orange', label='Represented Velocities')
    # plt.title("KDE Plot of Velocities")
    # plt.xlabel("Velocity")
    # plt.legend()
    #
    # plt.subplot(2, 2, 3)
    # sns.boxplot(data=[flattened_vo, flattened_vr], orient="h")
    # plt.yticks([0, 1], ['Original', 'Represented'])
    # plt.title("Box Plot of Velocities")
    # plt.xlabel("Velocity")
    #
    # plt.subplot(2, 2, 4)
    # sns.violinplot(data=[flattened_vo, flattened_vr], orient="h")
    # plt.yticks([0, 1], ['Original', 'Represented'])
    # plt.title("Violin Plot of Velocities")
    # plt.xlabel("Velocity")

    plt.tight_layout()
    # plt.show()
    return distance

def calculate_angle_change(X, Y):
    all_angle_changes = []
    for sample_x, sample_y in zip(X, Y):
        sample_angle_changes = []
        for stroke_x, stroke_y in zip(sample_x, sample_y):
            # Calculate differences between consecutive points
            deltaX = [stroke_x[i + 1] - stroke_x[i] for i in range(len(stroke_x) - 1)]
            deltaY = [stroke_y[i + 1] - stroke_y[i] for i in range(len(stroke_y) - 1)]
            angles = [math.atan2(dy, dx) for dx, dy in zip(deltaX, deltaY)]

            angle_changes = [(angles[i + 1] - angles[i] + math.pi) % (2 * math.pi) - math.pi for i in
                             range(len(angles) - 1)]
            sample_angle_changes.extend(angle_changes)

            all_angle_changes.append(sample_angle_changes)
    flattened_angle_changes = [item for sublist in all_angle_changes for item in sublist]
    return flattened_angle_changes


def plt_angle_distribution(Xo_, Yo_, Xr_, Yr_):
    angle_changes_original = calculate_angle_change(Xo_, Yo_)
    angle_changes_synthetic = calculate_angle_change(Xr_, Yr_)

    angle_changes_original = filter_outliers(angle_changes_original)
    angle_changes_synthetic = filter_outliers(angle_changes_synthetic)

    distance = wasserstein_distance(angle_changes_original, angle_changes_synthetic)
    print(f"Wasserstein Distance: {distance}")

    # plt.figure(figsize=(12, 8))

    # Histogram of angle changes
    plt.subplot(2, 2, 2)
    sns.histplot(angle_changes_original, bins=30, color='blue', kde=True, label='Original Angle Changes', alpha=0.6)
    sns.histplot(angle_changes_synthetic, bins=30, color='orange', kde=True, label='Represented Angle Changes',
                 alpha=0.6)
    plt.title("Histogram of Angle Changes")
    plt.xlabel("Angle Change (radians)")
    plt.ylabel("Frequency")
    plt.legend()
    #
    # # KDE plot of angle changes
    # plt.subplot(2, 2, 2)
    # sns.kdeplot(angle_changes_original, bw_adjust=0.5, color='blue', label='Original Angle Changes')
    # sns.kdeplot(angle_changes_synthetic, bw_adjust=0.5, color='orange', label='Represented Angle Changes')
    # plt.title("KDE Plot of Angle Changes")
    # plt.xlabel("Angle Change (radians)")
    # plt.legend()
    #
    # # Box plot of angle changes
    # plt.subplot(2, 2, 3)
    # sns.boxplot(data=[angle_changes_original, angle_changes_synthetic], orient="h")
    # plt.yticks([0, 1], ['Original', 'Represented'])
    # plt.title("Box Plot of Angle Changes")
    # plt.xlabel("Angle Change (radians)")
    #
    # # Violin plot of angle changes
    # plt.subplot(2, 2, 4)
    # sns.violinplot(data=[angle_changes_original, angle_changes_synthetic], orient="h")
    # plt.yticks([0, 1], ['Original', 'Represented'])
    # plt.title("Violin Plot of Angle Changes")
    # plt.xlabel("Angle Change (radians)")

    plt.tight_layout()
    # plt.show()
    return distance

def calculate_acceleration(velocities):
    acceleration = [velocities[i + 1] - velocities[i] for i in range(len(velocities) - 1)]
    return acceleration


def plt_acc_distribution(Vo_, Vr_):
    flattened_vo = list(itertools.chain.from_iterable(itertools.chain.from_iterable(Vo_)))
    flattened_vr = list(itertools.chain.from_iterable(itertools.chain.from_iterable(Vr_)))

    acceleration_original = calculate_acceleration(flattened_vo)
    acceleration_synthetic = calculate_acceleration(flattened_vr)
    acceleration_original = filter_outliers(acceleration_original)
    acceleration_synthetic = filter_outliers(acceleration_synthetic)

    distance = wasserstein_distance(acceleration_original, acceleration_synthetic)
    print(f"Wasserstein Distance: {distance}")


    # Set up the plot for acceleration
    # plt.figure(figsize=(12, 6))

    # Histogram of acceleration
    plt.subplot(2, 2, 1)
    sns.histplot(acceleration_original, bins=30, color='blue', kde=True, label='Original Acceleration', alpha=0.6)
    sns.histplot(acceleration_synthetic, bins=30, color='orange', kde=True, label='Represented Acceleration', alpha=0.6)
    plt.title("Histogram of Acceleration")
    plt.xlabel("Acceleration")
    plt.ylabel("Frequency")
    plt.legend()
    #
    # # KDE plot of acceleration
    # plt.subplot(2, 2, 2)
    # sns.kdeplot(acceleration_original, bw_adjust=0.5, color='blue', label='Original Acceleration')
    # sns.kdeplot(acceleration_synthetic, bw_adjust=0.5, color='orange', label='Represented Acceleration')
    # plt.title("KDE Plot of Acceleration")
    # plt.xlabel("Acceleration")
    # plt.legend()
    #
    # # Box plot of acceleration
    # plt.subplot(2, 2, 3)
    # sns.boxplot(data=[acceleration_original, acceleration_synthetic], orient="h")
    # plt.yticks([0, 1], ['Original', 'Represented'])
    # plt.title("Box Plot of Acceleration")
    # plt.xlabel("Acceleration")
    #
    # # Violin plot of acceleration
    # plt.subplot(2, 2, 4)
    # sns.violinplot(data=[acceleration_original, acceleration_synthetic], orient="h")
    # plt.yticks([0, 1], ['Original', 'Represented'])
    # plt.title("Violin Plot of Acceleration")
    # plt.xlabel("Acceleration")

    plt.tight_layout()
    # plt.show()
    return distance

directory_original = "synthetic/original"
directory_manipulated = "synthetic/manipulated"
directory_represented = "synthetic/represented"
directory_manipulated_2_modes = "synthetic/manipulated_2_modes"
directory_edge_data = "synthetic/second mode edge data"

arranged_original = "arranged/original"
arranged_manipulated = "arranged/manipulated"
arranged_represented = "arranged/represented"
arranged_manipulated_2_modes = "arranged/manipulated_2_modes"
arranged_edge_data = "arranged/second mode edge data"


i=7
Xo, Yo, To, Vo, infos_o = load_data(arranged_original)
data_o = Xo, Yo, To, Vo, infos = filter_data(Xo, Yo, To, Vo, infos_o, i)

Xr, Yr, Tr, Vr, infos_r = load_data(arranged_represented)
Xr, Yr, Tr, Vr, infos_r = filter_data(Xr, Yr, Tr, Vr, infos_r, i)

plt_length_distribution(Xo, Yo, Xr, Yr)
plt_acc_distribution(Vo, Vr)
plt_V_distribution(Vo, Vr)
plt_angle_distribution(Xo, Yo, Xr, Yr)
plt.show()