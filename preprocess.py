import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

import data
frequency = 150

def calculate_velocity(x, y, timestamps):
    velocity = []
    for stroke in (range(len(x))):
        velocity_one_stroke = [0]
        for i in range(1, len(x[stroke])):
            distance = ((x[stroke][i] - x[stroke][i-1]) ** 2 + (
                    y[stroke][i] - y[stroke][i-1]) ** 2) ** 0.5
            time = timestamps[stroke][i] - timestamps[stroke][i-1]
            if time == 0:
                velocity_one_stroke.append(velocity_one_stroke[-1])
                continue
            velocity_one_stroke.append(distance / time)
        velocity.append(velocity_one_stroke)
    return velocity

def preprocess(t, x, y):

    # Normalize X and Y coordinates
    x, y = normalize(x, y)

    t = (t - np.min(t))  # / 1000

    # Calculate Velocity
    velocity = calculate_velocity(x, y, t)

    # Smooth Velocity
    smoothed_velocity = extra_smooth(velocity, int(frequency/10))  # int(n_points/5)

    x = extra_smooth(x, int(frequency/10))  # int(n_points/5)
    y = extra_smooth(y, int(frequency/10))  # int(n_points/5)

    return np.array(x), np.array(y), np.array(t), np.array(smoothed_velocity), np.array(velocity)

def normalize(x, y):
    m_x = np.min(x)
    m_y = np.min(y)
    M_x = np.max(x, axis=0)
    M_y = np.max(y, axis=0)
    normalized_X = (x - m_x) / np.max(M_x - m_x)
    normalized_Y = (y - m_y) / np.max(M_y - m_y)
    return normalized_X, normalized_Y

def extra_smooth(velocity, window_size=5, poly=2): # int((global_number/5))
    smoothed_velocity1 = smooth_curve_2(velocity, window_size, poly)
    d_window = 10
    while window_size - d_window > poly:
        smoothed_velocity1 = smooth_curve_2(smoothed_velocity1, window_size - 10 , poly)
        d_window+=10
    return smoothed_velocity1

def smooth_curve_2(velocity_data, window_size=6, poly_order=3):
    window_size = window_size  # Adjust as needed
    poly_order = poly_order  # Adjust as needed
    try:
        smoothed_velocity = savgol_filter(velocity_data, window_size, poly_order)
    except ValueError:
        return velocity_data
    return smoothed_velocity

def smooth_V(v, window_size=5, poly=3):
    smoothed_v = []
    for stroke in range(len(v)):
        smoothed_v.append(extra_smooth(v[stroke], window_size, poly))
    return smoothed_v

def interpolate(y_values, nfs=2, n_points=None, interp="cubic"):
    time = np.linspace(0, len(y_values) - 1, len(y_values), endpoint=True)
    if n_points is None:
        time_inter = np.linspace(0, len(y_values) - 1, 1 + nfs * (len(y_values) - 1), endpoint=True)
    else:
        time_inter = np.linspace(0, len(y_values) - 1, n_points, endpoint=True)
    f = interp1d(time, y_values, kind=interp)
    return f(time_inter)

def shift_to_origin(x, y):
    min_x = np.min(np.concatenate(x))
    min_y = np.min(np.concatenate(y))

    x[0] -= min_x
    y[0] -= min_y

    if len(x) == 2:
        x[1] -= min_x
        y[1] -= min_y

    return x, y


def get_preprocessed_data(person,
                          charachter,
                          smoothing_window=6,
                          smooth_poly=5,
                          nfs=15,
                          finger="index",
                          glyph=None):
    X, Y, T, bio_infos_ = data.retrieve_data(person, charachter, finger, glyph)
    X, Y = shift_to_origin(X, Y)
    V = calculate_velocity(X, Y, T)
    print(smoothing_window)
    smoothed_V = smooth_V(V,  smoothing_window, int(smooth_poly))
    X = [interpolate(x, nfs=nfs, interp="cubic") for x in X]
    Y = [interpolate(y, nfs=nfs, interp="cubic") for y in Y]
    T = [interpolate(t, nfs=nfs, interp="linear") for t in T]
    V = [interpolate(t, nfs=nfs, interp="cubic") for t in V]
    smoothed_V = [interpolate(t, nfs=nfs, interp="cubic") for t in smoothed_V]

    return X, Y, T, V, smoothed_V, bio_infos_


if __name__ == '__main__':
    X, Y, T, V, smoothed_V, bio_infos = get_preprocessed_data(53, 8)
    print(bio_infos)
    for stroke in range(len(X)):
        plt.title(f"stroke {stroke}")
        plt.plot(T[stroke], V[stroke], label="originale Geschwendigkeit")
        plt.plot(T[stroke], smoothed_V[stroke], marker="o", label="smoothed Geschwindigkeit")
        plt.legend()
        plt.show()

    data.show_data(X, Y)

