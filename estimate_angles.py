import itertools
import math
import os
import warnings
import random
from multiprocessing import Pool, cpu_count

import pandas as pd
from scipy.signal import argrelextrema

import numpy as np
# from matplotlib import pyplot as plt

from extract_phy_parameter import get_phy_param_mat, generate_curve_from_parameters
from preprocess import get_preprocessed_data
# from redraw import draw_stroke_original
from fastdtw import fastdtw


def calculate_dtw_measure(X_original, Y_original, X_produced, Y_produced):
    # x and y will be 1D array
    flattened_X_original = np.concatenate(X_original)
    flattened_Y_original = np.concatenate(Y_original)

    flattened_X_produced = np.concatenate(X_produced)
    flattened_Y_produced = np.concatenate(Y_produced)

    # Stack the flattened original arrays to create the original graph
    original_graph = np.column_stack((flattened_X_original, flattened_Y_original))

    # Stack the flattened produced arrays to create the produced graph
    redrawn_graph = np.column_stack((flattened_X_produced, flattened_Y_produced))

    # Calculate the DTW distance between the original and produced graphs
    redrawn_distance, _ = fastdtw(original_graph, redrawn_graph)

    # Normalize the DTW distance
    normalized_redrawn_distance = (redrawn_distance / len(flattened_Y_original))
    #10.5 was good

    # Return the normalized DTW distance
    return normalized_redrawn_distance


def normalize(x, y):
    m_x = np.min(x)
    m_y = np.min(y)
    M_x = np.max(x, axis=0)
    M_y = np.max(y, axis=0)
    normalized_X = (x - m_x) / np.max(M_x - m_x)
    normalized_Y = (y - m_y) / np.max(M_y - m_y)
    return normalized_X, normalized_Y


def estimate_angles(X_values, Y_values, strokes_list, time):
    angles = []
    # plt.plot(X_values, Y_values, color="black", label="Bewegung")

    for i, stroke in enumerate(strokes_list):
        D, sigma, meu, t0 = stroke
        characteristic_points = find_char_points_lognormal(time, sigma, meu, t0)
        theta_s, theta_e = estimate_theta_SE(X_values, Y_values, D, sigma, characteristic_points, i)
        angles.append((theta_s, theta_e))
    # plt.axis("equal")
    # plt.legend()
    # plt.show()
    return angles


def find_char_points_lognormal(x_values, sigma, meu, x_0):
    v1 = x_0 + np.exp(meu - 3 * sigma)
    p1 = find_nearest_index(x_values, v1)
    v2 = x_0 + np.exp(meu - (1.5 * sigma ** 2 + sigma * np.sqrt(0.25 * sigma ** 2 + 1)))
    p2 = find_nearest_index(x_values, v2) + 1
    v3 = x_0 + np.exp(meu - sigma ** 2)
    p3 = find_nearest_index(x_values, v3)
    v4 = x_0 + np.exp(meu - (1.5 * sigma ** 2 - sigma * np.sqrt(0.25 * sigma ** 2 + 1)))
    p4 = find_nearest_index(x_values, v4) - 1
    v5 = x_0 + np.exp(meu + 3 * sigma)
    p5 = find_nearest_index(x_values, v5)
    return np.array([p1, p2, p3, p4, p5])


def find_nearest_index(arr, value):
    absolute_diff = np.abs(np.array(arr) - value)
    return np.argmin(absolute_diff)


def calculate_distance_cp(D, sigma, point):
    if point == 1:
        return 0
    if point == 5:
        return D
    if 0 < point < 5:
        point -= 2
        sig_sq = sigma ** 2
        a = [(3 / 2) * sig_sq + sigma * np.sqrt(((sig_sq) / 4) + 1),
             sig_sq,
             (3 / 2) * sig_sq - sigma * np.sqrt(((sig_sq) / 4) + 1)]

        return D / 2 * (1 + math.erf(-a[point] / (sigma * (2 ** 0.5))))
    return None


def estimate_theta_SE(x_values, y_values, D, sigma, characteristic_points, i=-1):
    _, xinf1, x3, xinf2, _ = characteristic_points
    d1, d2, d3, d4, d5 = [calculate_distance_cp(D, sigma, i) for i in range(1, 6)]

    # plt.scatter(x_values[characteristic_points], y_values[characteristic_points], label=f"char. points: {i}")

    dy = np.gradient(y_values)
    dx = np.gradient(x_values)
    anlges_list = np.arctan2(dy, dx)
    angle_t2 = anlges_list[xinf1]
    angle_t3 = anlges_list[x3]
    angle_t4 = anlges_list[xinf2]

    dAngle = angle_t4 - angle_t2

    # dieser Linie ist von dem Projekt SynSig2Vec inspiriert
    dAngle = math.copysign(2 * math.pi - abs(dAngle), -dAngle) if abs(dAngle) > 3. / 2 * math.pi else dAngle

    dDistanz = d4 - d2
    # print(dDistanz)
    dA_dD = dAngle / dDistanz if dDistanz > 20 else 0.01  # 20 is emperical chosen

    theta_s = angle_t3 - dA_dD * (d3 - d1)
    theta_e = angle_t3 + dA_dD * (d5 - d3)

    return theta_s, theta_e


def get_angels_matrix(T, X, Y, parameter_mat):
    angels_matrix = []
    for stroke in range(len(T)):
        X_stroke = X[stroke]
        Y_stroke = Y[stroke]
        parameter_plane = parameter_mat[stroke]
        t_stroke = T[stroke]
        angles1 = estimate_angles(X_stroke, Y_stroke, parameter_plane, t_stroke)
        angels_matrix.append(angles1)
    return angels_matrix


# retunrs all indexes of local max of a give curve
def get_local_max(curve):
    local_maxs = np.array([i for i in range(1, len(curve) - 1) if
                           curve[i] > curve[i - 1] and curve[i] >
                           curve[i + 1]])
    return local_maxs


# retunrs all indexes of local min of a give curve
def get_local_min(curve):
    local_maxs = np.array([i for i in range(1, len(curve) - 1) if
                           curve[i] < curve[i - 1] and curve[i] <
                           curve[i + 1]])
    return local_maxs


def get_extrems(curve):
    return get_local_max(curve), get_local_min(curve)


def draw_stroke_original(D, theta_s, theta_e, time, x_0, meu, sigma):
    S_x = []
    S_y = []
    denominator = theta_e - theta_s
    if denominator == 0:
        denominator += 0.01
    for t in time:
        point_x = (D / denominator) * (np.sin(phi(t, x_0, meu, sigma, theta_s, theta_e)) - np.sin(theta_s))
        S_x.append(point_x)
        point_y = (D / denominator) * (-np.cos(phi(t, x_0, meu, sigma, theta_s, theta_e)) + np.cos(theta_s))
        S_y.append(point_y)
    return np.array(S_x), np.array(S_y)


def phi(x, x_0, meu, sigma, theta_s, theta_e):
    modified_x = x - x_0
    modified_x = np.maximum(modified_x, 0.0000001)
    term2 = (theta_e - theta_s) / 2
    term3 = 1 + math.erf((np.log(modified_x) - meu) / (sigma * 2 ** 0.5))
    return_value = theta_s + term2 * term3
    return return_value


def redraw(smoothed_v, time, strokes_list, angles_list, regenerated_curve, one_stroke=False):
    acX = np.zeros_like(time)
    acY = np.zeros_like(time)
    local_min = []
    if not one_stroke:
        local_max, local_min = get_extrems(smoothed_v)
        local_min, local_max = correct_local_extrems(local_min, local_max, smoothed_v)
        local_min = np.insert(local_min, 0, 0)

    for stroke, angle, i in zip(strokes_list, angles_list, range(len(strokes_list))):
        D1, sigma, meu, t0 = stroke
        theta_s, theta_e = angle

        if not one_stroke:
            vx_selected = time[local_min[i]:local_min[i + 1]]
            vy_selected = regenerated_curve[local_min[i]:local_min[i + 1]]
            area_under_curve = np.trapezoid(vy_selected, vx_selected)
            D2 = area_under_curve
            # TODO: choose between D1 and D2 and the mean
            D = np.mean([D1, D2])
        else:
            D = D1
        # D = D2
        # D = D1
        # print((D, D2))
        drawn_X, drawn_Y = draw_stroke_original(D, theta_s, theta_e, time, t0, meu, sigma)
        condition = ~np.isnan(drawn_X) & ~np.isnan(drawn_Y)
        acX[condition] += drawn_X[condition]
        acY[condition] += drawn_Y[condition]
        # plt.plot(drawn_X, drawn_Y, color="blue")
        # plt.plot(acX, acY, color="black")
        # plt.axis("equal")
        # plt.show()
    return acX, acY


def correct_local_extrems(local_min, local_max, velocity, threshold_height=0.01, threshold_diff=0.03):
    summit = np.max(velocity)
    local_min_copy = local_min.copy()
    local_max_copy = local_max.copy()

    if len(local_min_copy) == 0 or len(local_min_copy) < len(local_max_copy):
        # print(len(local_min_copy) == 0, len(local_min_copy)< len(local_max_copy))
        local_min_copy = np.insert(local_min_copy, len(local_min_copy), len(velocity) - 1)

    local_min_copy = local_min_copy.astype(int)
    y_values_local_min = velocity[local_min_copy]
    y_values_local_max = velocity[local_max_copy]

    if len(local_min_copy) == 0 or len(local_min_copy) < len(local_max_copy):
        warnings.warn("something wrong in correct local extremes")

    for min, max, index_min, index_max in zip(y_values_local_min, y_values_local_max, local_min_copy, local_max_copy):
        min_to_remove = local_min_copy[np.where(velocity[local_min_copy] == min)[0]]
        max_to_remove = local_max_copy[np.where(velocity[local_max_copy] == max)[0]]
        condition_1 = max < (threshold_height * summit)
        condition_2 = velocity[index_max] - velocity[index_min] < threshold_diff * summit * 1.3
        # print((condition_1, condition_2, (velocity[index_max] - velocity[index_min]) / summit))
        if condition_1 or condition_2:
            local_min_copy = local_min_copy[local_min_copy != min_to_remove]
            local_max_copy = local_max_copy[local_max_copy != max_to_remove]

    return local_min_copy, local_max_copy


def shift_to_origin(x, y):
    min_x = np.min(np.concatenate(x))
    min_y = np.min(np.concatenate(y))

    x[0] -= min_x
    y[0] -= min_y

    if len(x) == 2:
        x[1] -= min_x
        y[1] -= min_y

    return x, y


def find_minimums_and_intercepts(T, diff_curve):
    local_min_indexes = argrelextrema(np.array(diff_curve), np.less)[0]
    zero_crossings = np.where(np.diff(np.sign(diff_curve)))[0]
    combined_indexes = np.sort(np.concatenate((local_min_indexes, zero_crossings)))
    return list(combined_indexes)


def find_largest_area_indexes(x_co, y_co, combined_indexes):
    max_area = 0
    max_indexes = [None, None]
    for i in range(len(combined_indexes) - 1):
        idx1 = combined_indexes[i]
        idx2 = combined_indexes[i + 1]
        area = np.trapezoid(y_co[idx1:idx2 + 1], x_co[idx1:idx2 + 1])
        if abs(area) > abs(max_area):
            max_area = area
            max_indexes = [idx1, idx2]
    return max_indexes


def full_redraw(smoothed_V, parameter_matrix, angels_matrix, regenerated_curve, X_, Y_, T_):
    redrawn_X, redraw_Y = [], []
    for plane in range(len(smoothed_V)):
        stroke_X, stroke_Y = redraw(smoothed_V[plane], T_[plane], parameter_matrix[plane],
                                    angels_matrix[plane], regenerated_curve[plane])
        stroke_X += X_[plane][0] - stroke_X[0]
        stroke_Y += Y_[plane][0] - stroke_Y[0]
        redrawn_X.append(stroke_X)
        redraw_Y.append(stroke_Y)
    redrawn_X, redraw_Y = shift_to_origin(redrawn_X, redraw_Y)
    return redrawn_X, redraw_Y


def full_redraw_second_mode(T_, diff_curve_, X_, Y_, full_X_, full_Y_):
    copy_full_X = []
    copy_full_Y = []
    for i in range(len(full_X_)):
        copy_full_Y.append(full_Y_[i].copy())
        copy_full_X.append(full_X_[i].copy())

    regenerated_curve2_ = []
    for plane in range(len(T_)):
        indexes = find_minimums_and_intercepts(T_[plane], diff_curve_[plane])
        biggest_area_indexes = find_largest_area_indexes(T_[plane], diff_curve_[plane], indexes)

        trimmed_curve = diff_curve_[plane].copy()
        trimmed_curve[T_[plane] < T_[plane][biggest_area_indexes[0]]] = 0
        trimmed_curve[T_[plane] > T_[plane][biggest_area_indexes[1]]] = 0

        factor = -1 if trimmed_curve[(biggest_area_indexes[0] + biggest_area_indexes[1]) // 2] < 0 else 1
        trimmed_curve *= factor

        parameter_matrix2 = get_phy_param_mat([T_[plane]], [trimmed_curve], True)

        b = 1 if parameter_matrix2[0][0][0] == 0 else 0
        parameter_matrix2[0][0] = (
            parameter_matrix2[0][0][0] * factor + b,  # Modify the first element
            parameter_matrix2[0][0][1],
            parameter_matrix2[0][0][2],
            parameter_matrix2[0][0][3]
        )

        regenerated_curve2_.append(generate_curve_from_parameters(parameter_matrix2, [T_[plane]])[0])
        angels_matrix2 = get_angels_matrix([T_[plane]], [X_[plane]], [Y_[plane]], parameter_matrix2)
        acX2, acY2 = redraw(trimmed_curve, T_[plane], parameter_matrix2[0],
                            angels_matrix2[0], regenerated_curve2_[0], one_stroke=True)
        copy_full_X[plane] += acX2
        copy_full_Y[plane] += acY2

    return copy_full_X, copy_full_Y, regenerated_curve2_


def modify_all_parameters(strokes, angles):
    modified_strokes = []
    modified_angles = []
    for i in range(len(strokes)):
        stroke_plane = strokes[i]
        angles_plane = angles[i]
        manipulated_stroke_plane =[]
        manipulated_angles_plane =[]
        for stroke, angle_pair in zip(stroke_plane, angles_plane):
            D, sigma, mu, t0 = stroke
            theta_s, theta_e = angle_pair
            mod_D, mod_sigma, mod_mu, mod_t0, mod_theta_s, mod_theta_e = modify_stroke_parameters(D, sigma, mu, t0, theta_s,
                                                                                                theta_e)
            manipulated_stroke_plane.append((mod_D, mod_sigma, mod_mu, mod_t0))
            manipulated_angles_plane.append((mod_theta_s, mod_theta_e))
        modified_strokes.append(manipulated_stroke_plane)
        modified_angles.append(manipulated_angles_plane)
    return modified_strokes, modified_angles


def modify_stroke_parameters(D, sigma, mu, t_0, theta_s, theta_e, d_mu=0.05, d_sigma=0.10, d_t0=0.01, d_D=0.2,
                             d_theta_s=0.1,
                             d_theta_e=0.1):
    changed_D = np.random.normal(D, (D * d_D) ** 2)
    changed_sigma = np.random.normal(sigma, (sigma * d_sigma) ** 2)
    changed_mu = np.random.normal(mu, (mu * d_mu) ** 2)
    changed_t_0 = t_0 + np.random.normal(0, (d_t0) ** 2)
    changed_theta_s = theta_s + np.random.normal(0, (d_theta_s) ** 2)
    changed_theta_e = theta_e + np.random.normal(0, (d_theta_e) ** 2)

    changed_D = np.clip(changed_D, D - 2 * d_D, D + 2 * d_D)
    changed_sigma = np.clip(changed_sigma, sigma - 2 * d_sigma, sigma + 2 * d_sigma)
    changed_mu = np.clip(changed_mu, mu - 2 * d_mu, mu + 2 * d_mu)
    changed_t_0 = np.clip(changed_t_0, t_0 - 2 * d_t0, t_0 + 2 * d_t0)
    changed_theta_s = np.clip(changed_theta_s, theta_s - 2 * d_theta_s, theta_s + 2 * d_theta_s)
    changed_theta_e = np.clip(changed_theta_e, theta_e - 2 * d_theta_e, theta_e + 2 * d_theta_e)

    if changed_sigma == 0:
        changed_sigma += 0.01
    if changed_D == 0:
        changed_D += 0.01
    return changed_D, changed_sigma, changed_mu, changed_t_0, changed_theta_s, changed_theta_e

def represent_curve_lognormal(X_, Y_, T_, V, smoothed_V, name, V1):
    parameter_matrix = get_phy_param_mat(T_, smoothed_V)
    regenerated_curve = generate_curve_from_parameters(parameter_matrix, T_)
    angles_matrix = get_angels_matrix(T_, X_, Y_, parameter_matrix)
    full_X, full_Y = full_redraw(smoothed_V, parameter_matrix, angles_matrix, regenerated_curve, X_, Y_, T_)

    best_dtw = calculate_dtw_measure(X_, Y_, full_X, full_Y)
    print("DTW after frst mode: ", best_dtw)
    diff_curve = [smoothed_V[stroke] - regenerated_curve[stroke] for stroke in range(len(smoothed_V))]
    # plt.axis("equal")
    # for plane in range(len(T_)):
    #     plt.plot(X_[plane], Y_[plane], label="original", color="black")
    #     plt.plot(full_X[plane], full_Y[plane], label="made1", color="blue")
    #     plt.legend()
    while True:
        full_Xnew, full_Ynew, regenerated_curve2 = full_redraw_second_mode(T_, diff_curve, X_, Y_, full_X, full_Y)
        current_dtw = calculate_dtw_measure(X_, Y_, full_Xnew, full_Ynew)
        print(current_dtw)
        if current_dtw < best_dtw:
            best_dtw = current_dtw
            full_X, full_Y = full_Xnew, full_Ynew
            diff_curve = [diff_curve[stroke] - regenerated_curve2[stroke] for stroke in range(len(smoothed_V))]
        else:
            break
    DTW = calculate_dtw_measure(X_, Y_, full_X, full_Y)
    print("DTW after second mode: ", DTW)

    # plt.figure(2)
    # plt.axis("equal")
    # for plane in range(len(T_)):
    #     plt.plot(X_[plane], Y_[plane], label="original", color="black")
    #     plt.plot(full_X[plane], full_Y[plane], label="made2", color="blue")
    #     plt.legend()

    save_data(full_X, full_Y, V1, name+f"_{DTW}", "synthetic/represented")
    return parameter_matrix, angles_matrix, regenerated_curve2, DTW


def represent_manipulated_curve_lognormal(parameter_matrix_, angles_matrix_, regenerated_curve2_, smoothed_V_, X_,
                                          Y_, T_, end_DTW, name, V1):
    manipulated_parameters, manipulated_angles = modify_all_parameters(parameter_matrix_, angles_matrix_)
    manipulated_regenerated_curve = generate_curve_from_parameters(manipulated_parameters, T_)
    manipulated_full_X, manipulated_full_Y = full_redraw(smoothed_V_,
                                                         manipulated_parameters,
                                                         manipulated_angles,
                                                         manipulated_regenerated_curve, X_, Y_, T_)
    print("DTW manipulation frst mode: ", calculate_dtw_measure(X_, Y_, manipulated_full_X, manipulated_full_Y))

    # plt.figure(3)
    # plt.axis("equal")
    # for plane in range(len(T_)):
    #     plt.plot(X_[plane], Y_[plane], label="original", color="black")
    #     plt.plot(manipulated_full_X[plane], manipulated_full_Y[plane], label="made3", color="blue")
    #     plt.legend()

    best_dtw_mani = calculate_dtw_measure(X_, Y_, manipulated_full_X, manipulated_full_Y)
    diff_curve_mani = [smoothed_V_[stroke] - manipulated_regenerated_curve[stroke] for stroke in range(len(smoothed_V_))]
    save_data(manipulated_full_X, manipulated_full_Y, V1, name, "synthetic/manipulated")
    while True:
        full_Xnew_mani, full_Ynew_mani, regenerated_curve2_mani = full_redraw_second_mode(T_, diff_curve_mani, X_, Y_,
                                                                                          manipulated_full_X,
                                                                                          manipulated_full_Y)
        current_dtw_mani = calculate_dtw_measure(X_, Y_, full_Xnew_mani, full_Ynew_mani)
        print(current_dtw_mani)
        if current_dtw_mani < best_dtw_mani:
            best_dtw_mani = current_dtw_mani
            manipulated_full_X, manipulated_full_Y = full_Xnew_mani, full_Ynew_mani
            diff_curve_mani = [diff_curve_mani[stroke] - regenerated_curve2_[stroke] for stroke in
                               range(len(smoothed_V_))]
        else:
            break
    end2_DTW = calculate_dtw_measure(X_, Y_, manipulated_full_X, manipulated_full_Y)
    print("DTW manipulation second mode: ", end2_DTW)
    if end2_DTW < 2.5*end_DTW:
        save_data(manipulated_full_X, manipulated_full_Y, V1, name, "synthetic/manipulated_2_modes")
    else:
        save_data(manipulated_full_X, manipulated_full_Y, V1, name, "synthetic/second mode edge data")

    # plt.figure(4)
    # plt.axis("equal")
    # for plane in range(len(T_)):
    #     plt.plot(X_[plane], Y_[plane], label="original", color="black")
    #     plt.plot(manipulated_full_X[plane], manipulated_full_Y[plane], label="made4", color="blue")
    #     plt.legend()
    # plt.show()



def save_data(X1, Y1, V1, name, folder="synthetic/original"):
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, f"{name}.npz")
    np.savez(file_path, X1=np.array(X1, dtype=object),
                       Y1=np.array(Y1, dtype=object),
                       V1=np.array(V1, dtype=object))

def check_file_exists(name, directory="synthetic/original"):
    file_path = os.path.join(directory, f"{name}.npz")
    return os.path.exists(file_path)

# if __name__ == '__main__':
#     # 21.803250541503335
#     persons_list = pd.read_csv('subject.csv', sep=',')["Z_PK"].to_numpy()
#     character_list = range(10)
#     finger_list = ("index", "thumb")
#     glyph_list = range(4)
#
#     combinations = list(itertools.product(persons_list, character_list, finger_list,
#                                           glyph_list))
#     print(len(combinations))
#     random.shuffle(combinations)
#     sample_size = 10
#     sampled_combinations = combinations[:sample_size]
#
#     # person = 53
#     # character = 6
#     # X1, Y1, T1, V1, smoothed_V1, bio_infos = get_preprocessed_data(person, character)
#     # name = f"{person}_{character}_{"index"}_{0}_{bio_infos[0]}_{bio_infos[1]}_{bio_infos[2]}"
#     # save_data(X1, Y1, V1, name)
#     #
#     # parameter_matrix, angles_matrix, regenerated_curve2 = represent_curve_lognormal(X1, Y1, T1, V1, smoothed_V1)
#     # represent_manipulated_curve_lognormal(parameter_matrix, angles_matrix, regenerated_curve2, smoothed_V1, X1, Y1, T1)
#
#
#     for index, (person, character, finger, glyph) in enumerate(sampled_combinations):
#         print(f"{index+1}: {(person, character, finger, glyph)}")
#         try:
#             X1, Y1, T1, V, smoothed_V1, bio_infos = get_preprocessed_data(person, character, finger=finger, glyph=glyph)
#         except ValueError:
#             print("glyph misses")
#             continue
#
#         # bio_infos: tuple (sex, hand, age)
#         name_ = f"{person}_{character}_{finger}_{glyph}_{bio_infos[0]}_{bio_infos[1]}_{bio_infos[2]}"
#         if check_file_exists(name_):
#             print("exists")
#             continue
#
#         save_data(X1, Y1, V, name_)
#         parameter_matrix, angles_matrix, regenerated_curve2, end_DTW = represent_curve_lognormal(X1, Y1, T1, V, smoothed_V1, name_, V)
#         represent_manipulated_curve_lognormal(parameter_matrix, angles_matrix, regenerated_curve2, smoothed_V1, X1, Y1, T1, end_DTW, name_, V)

def process_combination(args):
    index, person, character, finger, glyph = args
    print(f"{index + 1}: {(person, character, finger, glyph)}")
    try:
        X1, Y1, T1, V, smoothed_V1, bio_infos = get_preprocessed_data(person, character, finger=finger, glyph=glyph)
    except ValueError:
        print("glyph misses")
        return

    # bio_infos: tuple (sex, hand, age)
    name_ = f"{person}_{character}_{finger}_{glyph}_{bio_infos[0]}_{bio_infos[1]}_{bio_infos[2]}"
    if check_file_exists(name_):
        print("exists")
        return

    save_data(X1, Y1, V, name_)
    parameter_matrix, angles_matrix, regenerated_curve2, end_DTW = represent_curve_lognormal(
        X1, Y1, T1, V, smoothed_V1, name_, V
    )
    represent_manipulated_curve_lognormal(parameter_matrix, angles_matrix, regenerated_curve2,
                                          smoothed_V1, X1, Y1, T1, end_DTW, name_, V)



def main():
    # Load the list of persons
    persons_list = pd.read_csv('subject.csv', sep=',')["Z_PK"].to_numpy()
    character_list = range(10)
    finger_list = ("index", "thumb")
    glyph_list = range(4)

    # Generate all combinations
    combinations = list(itertools.product(persons_list, character_list, finger_list, glyph_list))
    print(f"Total combinations: {len(combinations)}")

    # Shuffle and sample
    random.shuffle(combinations)
    sample_size = 10
    sampled_combinations = combinations[:sample_size]
    args_list = [(index, person, character, finger, glyph) for index, (person, character, finger, glyph) in
                 enumerate(sampled_combinations)]

    # Calculate number of workers based on desired CPU usage (70%)
    total_cores = cpu_count()  # Get the total number of CPU cores
    desired_cpu_usage = 0.7    # Desired CPU usage percentage
    num_workers = max(1, int(total_cores * desired_cpu_usage))  # Calculate the number of workers

    print(f"Using {num_workers} out of {total_cores} cores ({desired_cpu_usage * 100}% CPU usage)")

    # Use multiprocessing to execute each combination in parallel
    with Pool(processes=num_workers) as pool:
        pool.map(process_combination, args_list)


if __name__ == '__main__':
    main()