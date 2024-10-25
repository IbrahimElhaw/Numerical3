import math
import warnings

import numpy as np
from matplotlib import pyplot as plt
from data import show_data
from preprocess import get_preprocessed_data

first_range_start = 0.35
first_range_end = 0.68
second_range_start = 0.50
second_range_end = 0.80
sigma_accuracy = 40
alpha_beta = [(2, 3), (3, 4), (4, 2)]


def get_local_max(curve):
    local_maxes = []
    for stroke in range(len(curve)):
        local_maxes.append(np.array([i for i in range(1, len(curve[stroke]) - 1) if
                                     curve[stroke][i] > curve[stroke][i - 1] and curve[stroke][i] >
                                     curve[stroke][i + 1]]))
    return local_maxes


# retunrs all indexes of local min of a give curve
def get_local_min(curve):
    local_maxs = []
    for stroke in range(len(curve)):
        local_maxs.append(np.array([i for i in range(1, len(curve[stroke]) - 1) if
                                    curve[stroke][i] < curve[stroke][i - 1] and curve[stroke][i] <
                                    curve[stroke][i + 1]]))
    return local_maxs


def show_min_max(x, y, t, v, smoothed_v, loc_min, loc_max):
    for stroke in range(len(x)):
        plt.title(f"stroke {stroke}")
        plt.plot(t[stroke], v[stroke], label="originale Geschwendigkeit")
        plt.plot(t[stroke], smoothed_v[stroke], marker="o", label="smoothed Geschwindigkeit")
        plt.scatter(t[stroke][loc_min[stroke]], smoothed_v[stroke][loc_min[stroke]], color="purple", label="local min",
                    zorder=3)
        plt.scatter(t[stroke][loc_max[stroke]], smoothed_v[stroke][loc_max[stroke]], color="pink", label="local max",
                    zorder=3)
        plt.legend()
        plt.show()
    show_data(x, y)


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


def get_local_extrems(sv):
    local_min = get_local_min(sv)
    local_max = get_local_max(sv)
    # show_min_max(X, Y, T, V, smoothed_V, local_min, local_max)

    updated_local_min = []
    updated_local_max = []

    for stroke in range(len(sv)):
        cur_local_min = local_min[stroke]
        cur_local_max = local_max[stroke]
        cur_SV = sv[stroke]
        up_min, up_max = correct_local_extrems(cur_local_min, cur_local_max, cur_SV)
        updated_local_min.append(up_min)
        updated_local_max.append(up_max)

    return updated_local_min, updated_local_max


def corresponding_x_values(x_values, velocity_profile, v_3, x_3, sigma_accuracy, min_index):
    condition = (velocity_profile < (first_range_start * v_3)) & (x_values < x_3)
    corresponding_x_value1 = x_values[condition][-1]
    condition = (velocity_profile < (first_range_end * v_3)) & (x_values < x_3)
    corresponding_x_value2 = x_values[condition][-1]
    condition = (velocity_profile < (second_range_start * v_3)) & (x_values > x_3)
    if np.any(condition):
        corresponding_x_value3 = x_values[condition][0]
    else:
        points_after_x3 = x_values[x_values > x_3]
        corresponding_x_value3 = points_after_x3[int(len(points_after_x3) / 2)]
    condition = (velocity_profile < (second_range_end * v_3)) & (x_values > x_3)
    if np.any(condition):
        corresponding_x_value4 = x_values[condition][0]
    else:
        points_after_x4 = x_values[x_values > x_3]
        corresponding_x_value4 = points_after_x4[int(len(points_after_x4) * 4 / 5)]
    x_values_v2_inf1 = np.linspace(corresponding_x_value1, corresponding_x_value2, sigma_accuracy)
    if corresponding_x_value4 - x_values[min_index] > corresponding_x_value2:
        corresponding_x_value4 -= x_values[min_index]
        corresponding_x_value3 -= x_values[min_index]
    x_values_v4_inf2 = np.linspace(corresponding_x_value3, corresponding_x_value4, sigma_accuracy)
    return x_values_v2_inf1, x_values_v4_inf2


def calculate_sigmas(first_inflection, local_max, second_inflection):
    sigmas = []
    # 1
    beta_23 = first_inflection / local_max
    sigma_quadrad = -2 - 2 * np.log(beta_23) - (1 / (2 * np.log(beta_23)))
    sigma = np.sqrt(sigma_quadrad)
    sigmas.append(sigma)

    # 2
    beta_42 = second_inflection / first_inflection
    sigma_quadrad = -2 + 2 * np.sqrt(1 + (np.log(beta_42)) ** 2)
    sigma = np.sqrt(sigma_quadrad)
    sigmas.append(sigma)

    # 3
    beta_43 = second_inflection / local_max
    sigma_quadrad = -2 - 2 * np.log(beta_43) - (1 / (2 * np.log(beta_43)))
    sigma = np.sqrt(sigma_quadrad)
    sigmas.append(sigma)

    return np.array(sigmas)


def calculate_meus(x_values, y_values, x2_inf1, x4_inf2, sigma):
    meus = []
    a = [np.nan, np.nan]
    a.append((3 / 2) * sigma ** 2 + sigma * np.sqrt(((sigma ** 2) / 4) + 1))
    a.append(sigma ** 2)
    a.append((3 / 2) * sigma ** 2 - sigma * np.sqrt(((sigma ** 2) / 4) + 1))

    t_3 = x_values[np.argmax(y_values)]
    t2_inf1, t4_inf2 = x2_inf1, x4_inf2
    carachterstic_times = [np.nan, np.nan, t2_inf1, t_3, t4_inf2]
    for alpha, beta in alpha_beta:
        first_term = (carachterstic_times[alpha] - carachterstic_times[beta])
        second_term = (np.exp(-a[alpha]) - math.exp(-a[beta]))

        if second_term == 0:
            second_term += 0.001
        ratio = first_term / second_term
        if ratio <= 0:
            ratio = 0.001
        meus.append(np.log(ratio))
    return np.array(meus)


def calculate_t_0(x_values, y_values, x2_inf1, x4_inf2, sigmas, meus):
    modified_meus = meus.copy()
    t_0_liste = []
    a = [np.nan, np.nan]
    a.append((3 / 2) * sigmas ** 2 + sigmas * np.sqrt(((sigmas ** 2 / 4) + 1)))
    a.append(sigmas ** 2)
    a.append((3 / 2) * sigmas ** 2 - sigmas * np.sqrt(((sigmas ** 2 / 4) + 1)))

    while len((modified_meus)) != len(a):
        modified_meus = np.insert(modified_meus, 0, np.nan)

    t_3 = x_values[np.argmax(y_values)]
    t2_inf1, t4_inf2 = x2_inf1, x4_inf2
    carachterstic_times = [np.nan, np.nan, t2_inf1, t_3, t4_inf2]
    for alpha, beta in alpha_beta:
        t_0_liste.append(carachterstic_times[alpha] - np.exp(modified_meus[alpha] - a[alpha]))
    return np.array(t_0_liste)


def calculate_Ds(x_values, y_values, x2_inf1, x4_inf2, sigma, meus):
    modified_meus = meus.copy()
    D = []
    a = [np.nan, np.nan]
    a.append((3 / 2) * sigma ** 2 + sigma * np.sqrt(((sigma ** 2 / 4) + 1)))
    a.append(sigma ** 2)
    a.append((3 / 2) * sigma ** 2 - sigma * np.sqrt(((sigma ** 2 / 4) + 1)))

    # while len((modified_sigmas)) != len(a):
    #     modified_sigmas = np.insert(modified_sigmas, 0, np.nan)
    while len((modified_meus)) != len(a):
        modified_meus = np.insert(modified_meus, 0, np.nan)

    t_3 = x_values[np.argmax(y_values)]  # the biggest summit in the remaining curve
    t2_inf1, t4_inf2 = x2_inf1, x4_inf2
    carachterstic_times = [np.nan, np.nan, t2_inf1, t_3, t4_inf2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for alpha, beta in alpha_beta:
            v_0 = y_values[np.argmin(np.abs(x_values - carachterstic_times[alpha]))]
            D.append(v_0 * sigma * np.sqrt(2 * np.pi) * np.exp(
                modified_meus[alpha] + ((a[alpha] ** 2) / (2 * sigma ** 2)) - a[alpha]))
    return np.array(D)


def calculate_MSE(real_y_values, forged_yvalues):
    return np.sqrt(np.mean((real_y_values - forged_yvalues) ** 2))


def generate_lognormal_curve(D, std_dev, mean, x_0, start, end, number_of_points):
    time = np.linspace(start, end, number_of_points)
    curve = np.zeros_like(time)
    if std_dev == 0:
        return curve

    condition = time > x_0
    curve[condition] = (D / ((time[condition] - x_0) * std_dev * np.sqrt(2 * np.pi))) * np.exp(
        -((np.log(time[condition] - x_0) - mean) ** 2) / (2 * std_dev ** 2))
    return curve


def calculate_parameters(v_inf1_range, x_inf1_range, v_inf2_range, x_inf2_range, max_v, time, velo, stroke_end_index):
    parameters = []
    for v2_inf1, x2_inf1 in zip(v_inf1_range, x_inf1_range):
        for v4_inf2, x4_inf2 in zip(v_inf2_range, x_inf2_range):
            sigmas = calculate_sigmas(v2_inf1, max_v, v4_inf2)
            for sigma in sigmas:
                meus = calculate_meus(time, velo, x2_inf1, x4_inf2, sigma)
                t_0s = calculate_t_0(time, velo, x2_inf1, x4_inf2, sigma, meus)
                Ds = calculate_Ds(time, velo, x2_inf1, x4_inf2, sigma, meus)
                for meu, t_0, D in zip(meus, t_0s, Ds):
                    parameters.append([D, sigma, meu, t_0])

    bestMSE = float("inf")
    bestfit = None
    best_generate = np.zeros_like(velo)

    for param in parameters:
        D, sigma, meu, t_0, = param
        generated_profile = generate_lognormal_curve(D, sigma, meu, t_0, time[0], time[-1], len(time))
        MSE = calculate_MSE(generated_profile[time < time[stroke_end_index]],
                            velo[time < time[stroke_end_index]])
        if MSE < bestMSE:
            best_generate = generated_profile
            bestMSE = MSE
            bestfit = (D, sigma, meu, t_0)
            # plt.plot(time, best_generate)
            # plt.plot(time, velo)
            # plt.show()
    return bestfit, best_generate


def trim_velocity(trimmed_vel, time, trim_index):
    trimmed_vel[time > time[trim_index]] = 0
    return trimmed_vel


def get_phy_param_mat(t, s_v, one_Stroke=False):
    parameter_matrix = []
    local_minimums, local_maximums = get_local_extrems(s_v)
    # show_min_max(X, Y, T, V, smoothed_V, local_minimums, local_maximums)

    for stroke in range(len(s_v)):
        ebene = []
        curr_T = t[stroke]
        curr_V = s_v[stroke].copy()
        cur_local_min = local_minimums[stroke]
        cur_local_max = local_maximums[stroke]
        if one_Stroke:
            if cur_local_min[0]<cur_local_max[0]:
                cur_local_min = np.insert(cur_local_min, -1, len(curr_V)-1)
        # plt.plot(curr_T, curr_V)
        # plt.scatter(curr_T[cur_local_min], curr_V[cur_local_min])
        # plt.scatter(curr_T[cur_local_max], curr_V[cur_local_max])
        # plt.show()
        for i in range(len(cur_local_max)):
            trimmed_velocity = curr_V.copy()

            used_local_max_index = cur_local_max[i]
            used_local_min_index = cur_local_min[cur_local_min > used_local_max_index][0]

            trimmed_velocity = trim_velocity(trimmed_velocity, curr_T, used_local_min_index)

            t_3 = curr_T[used_local_max_index]
            v_3 = curr_V[used_local_max_index]

            v2_inf1_range = np.linspace(first_range_start * v_3, first_range_end * v_3, sigma_accuracy)
            v4_inf2_range = np.linspace(second_range_start * v_3, second_range_end * v_3, sigma_accuracy)

            x_values_v2_inf1, x_values_v4_inf2 = corresponding_x_values(curr_T, curr_V, v_3, t_3,
                                                                        sigma_accuracy, used_local_min_index)

            # plot every stroke of the curve alone
            # plt.plot(curr_T, trimmed_velocity, color="red", label="velocity")
            # plt.scatter(t_3, v_3, color="black")
            # plt.plot(x_values_v2_inf1, v2_inf1_range, color="black", label="range of inflection points")
            # plt.plot(x_values_v4_inf2, v4_inf2_range, color="black", label="range of inflection points")
            # plt.show()

            params, generated = calculate_parameters(v2_inf1_range, x_values_v2_inf1, v4_inf2_range, x_values_v4_inf2,
                                                     v_3, curr_T,
                                                     trimmed_velocity,
                                                     used_local_min_index)
            ebene.append(params)
            curr_V -= generated
            # y_values[x_values < x_values[np.argmax(best_generate)]] = 0
        parameter_matrix.append(ebene)
    return parameter_matrix


def generate_curve_from_parameters(para_matrix, timestamps):
    generated_velocity = []
    for stroke in range(len(timestamps)):
        generated_velocity.append(np.zeros_like(timestamps[stroke]))
        sheet = para_matrix[stroke]
        for row in sheet:  # (D, sigma, meu, t_0)
            generated_velocity[stroke] += generate_lognormal_curve(row[0],
                                                                   row[1],
                                                                   row[2],
                                                                   row[3],
                                                                   timestamps[stroke][0],
                                                                   timestamps[stroke][-1],
                                                                   len(timestamps[stroke]))
    return generated_velocity


if __name__ == '__main__':
    X, Y, T, V, smoothed_V, bio_infos = get_preprocessed_data(53, 8, smoothing_window=3, smooth_poly=5)
    parameter_matrix = get_phy_param_mat(T, smoothed_V)
    for plane in parameter_matrix:
        print("new plane")
        for line in plane:
            print(line)
    regenerated_curve = generate_curve_from_parameters(parameter_matrix, T)
    for stroke in range(len(T)):
        pass
        plt.plot(T[stroke], regenerated_curve[stroke], label="regenerated", color="black")
        plt.plot(T[stroke], smoothed_V[stroke], label="smoothed", color="red")
        plt.plot(T[stroke], V[stroke], label="original", color="grey")

        plt.legend()
        plt.show()
    print(np.shape(parameter_matrix))
