import cvxpy as cp
import numpy as np
from sklearn.metrics import mean_squared_error
from datetime import datetime, date
import datetime as dt
import copy
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from tslearn.clustering import TimeSeriesKMeans


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def moving_sum(x, w):
    return np.convolve(x, np.ones(w), 'valid') 


def np_vector_min_max_scale(x):
    return (x - np.amin(x)) / (np.amax(x) - np.amin(x))


def compute_weight_between_target_neighbor(target_b, b_list, print_info=False):
    b_array = np.array(b_list).T


    #### solve linear regression using optimization technique
    K = cp.Variable(len(b_list), nonneg=True)
    # K = cp.Variable(len(b_list))
    # objective = cp.Minimize(cp.sum_squares((b_array@K) - target_b))
    objective = cp.Minimize(cp.sum_squares((b_array@K) - target_b) + 0.5*cp.norm(K, 1))
    prob = cp.Problem(objective)
    result = prob.solve()

    if print_info:
        print("cv error for convex = ", cv_error(target_b, np.matmul(b_array, K.value)))

    return K.value


def compute_ratio_between_target_neighbor(target_b, b_list, print_info=False):
    ## find the solar generation ratio between target home and neighbor homes
    
    PV_scale_list = []
    k = cp.Variable(pos=True)
    y = cp.Parameter((target_b.shape[0],))
    y_hat = cp.Parameter((target_b.shape[0],))
    cost = cp.sum_squares(y - k*y_hat)
    prob = cp.Problem(cp.Minimize(cost))

    for i in range(len(b_list)):
        y.value = target_b
        y_hat.value = b_list[i]
        prob.solve()
        PV_scale_list.append(float(k.value))

    if print_info:
        b_list_copy = copy.deepcopy(b_list)
        for i in range(len(b_list)):
            print("mean = ", np.mean(PV_scale_list[i] * b_list[i]), " std = ", np.std(PV_scale_list[i] * b_list[i]))
            b_list_copy[i] = PV_scale_list[i] * b_list_copy[i]
            
        print("target mean = ", np.mean(target_b), " target std = ", np.std(target_b))
        print("group mean = ", np.mean(np.array(b_list_copy)), " group std = ", np.std(np.array(b_list_copy)))


    return PV_scale_list

def read_data_path(filename):
    # if filename != "data_path.txt" and filename != "../data_path.txt":
    #     raise Exception("file name must be data_path.txt")
    
    f = open(filename, "r")
    data_path = f.readline()
    f.close()
    return data_path


def time_index(start_hour, end_hour, num_points_per_hour, maximum_index):
    idx_list = []
    idx = start_hour * num_points_per_hour
    offset = ((end_hour - start_hour) * num_points_per_hour) + (num_points_per_hour - 1)
    min_range_idx = idx
    max_range_idx = min_range_idx + offset
    
    while idx < maximum_index:
        
        if idx <= max_range_idx:
            idx_list.append(idx)
            idx += 1
        else:
            idx = min_range_idx + (24*num_points_per_hour)
            min_range_idx = idx
            max_range_idx = min_range_idx + offset

    return np.array(idx_list)


def split_time(sr, ss, t, light_time_separation_num):
    # if light_time_separation_num <= 1:
    #     raise Exception("the separation number should at least be 2")

    if light_time_separation_num == 0:
        result = { "total_time": t.index }

    elif light_time_separation_num == 1:
        result = {
                    "light_time": t[(t.dt.time > sr.time()) & (t.dt.time < ss.time())].index,
                    "evening_time": t[~((t.dt.time > sr.time()) & (t.dt.time < ss.time()))].index,
                }

    else:
        result = {}

        duration = datetime.combine(date.min, ss.time()) - datetime.combine(date.min, sr.time())
        time_delta = duration / light_time_separation_num

        time_flag_list = [sr, ]
        for i in range(light_time_separation_num-1):
            time_flag_list.append(datetime.combine(date.min, sr.time()) + (time_delta * (i+1)))
        time_flag_list.append(ss)

        result['tp_0'] = t[(t.dt.time < sr.time()) | (t.dt.time >= ss.time())].index

        for i in range(len(time_flag_list)-1):
            key = "tp_" + str(i+1)
            result[key] = t[(t.dt.time >= time_flag_list[i].time()) & (t.dt.time < time_flag_list[i+1].time())].index

    return result

    


def cluster_days_by_weather(weather, num_days=30, num_points_per_hour=4, n_clusters=2):
    """
        weather should be (day_points, features)
    """

    num_hours_per_day = 24

    reshape_weather = np.reshape(weather, (num_days, num_points_per_hour*num_hours_per_day, -1))
    km = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw")

    day_groups = km.fit_predict(reshape_weather)

    # index dict for each day
    days_index = {}
    for i in range(num_days):
        days_index[i] = np.arange(i*(num_points_per_hour*num_hours_per_day), (i+1)*(num_points_per_hour*num_hours_per_day))

    result = []
    for i in range(len(day_groups)):
        day_idxs = np.where(day_groups == day_groups[i])[0]
        result.append(np.empty((day_idxs.shape[0] * (num_points_per_hour*num_hours_per_day), ), dtype=int))
        for j in range(day_idxs.shape[0]):
            result[i][j*(num_points_per_hour*num_hours_per_day):(j+1)*(num_points_per_hour*num_hours_per_day)] = days_index[day_idxs[j]]

    return result

def cluster_days_by_weather_v2(weather, num_days=30, num_points_per_hour=4, n_clusters=2):
    num_hours_per_day = 24

    reshape_weather = np.reshape(weather, (num_days, num_points_per_hour*num_hours_per_day, -1))
    km = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw")
    day_groups = km.fit_predict(reshape_weather)

    return day_groups

def time_increment(start_time, delta_day):
    start_obj = datetime.strptime(start_time, '%Y-%m-%d')
    end_obj = start_obj + dt.timedelta(days=delta_day)

    return end_obj.strftime("%Y-%m-%d")


def triangular_value_diagonal_matrix(freq, func, length, shift=0.0):
    if func == "sin":
        f = np.sin
    elif func == "cos":
        f = np.cos

    vector = np.ones((length, )) * np.nan
    for i in range(length):
        vector[i] = f((2*np.pi*i / freq) + shift)

    return np.diag(vector)




def neigh_list_segmentation(neigh_list, segment_length, s_i, weather_cluster_result, sample_from_distribution):
    seg_neigh_list = []
    
    if sample_from_distribution or (weather_cluster_result == None):
        for nei_idx in range(len(neigh_list)):
            num_seg = int(neigh_list[nei_idx].shape[0] / segment_length)
            for seg_idx in range(num_seg):
                seg_neigh_list.append(neigh_list[nei_idx][seg_idx*segment_length:(seg_idx+1)*segment_length])
    
    else:
        for nei_idx in range(len(neigh_list)):
            weather_group_idxs = weather_cluster_result[s_i]
            num_seg = int(neigh_list[nei_idx][weather_group_idxs].shape[0] / segment_length)
            for seg_idx in range(num_seg):
                seg_neigh_list.append(neigh_list[nei_idx][weather_group_idxs][seg_idx*segment_length:(seg_idx+1)*segment_length])

    return seg_neigh_list


def neigh_list_segmentation_same_time(neigh_list, seg_idxs):
    seg_neigh_list = []

    for nei_idx in range(len(neigh_list)):
        seg_neigh_list.append(neigh_list[nei_idx][seg_idxs])

    return seg_neigh_list


def segmentation_home_data(target_home_data, seg_idxs, features=None):
    seg_target_home_data = {}
    
    if features == None:
        features = list(target_home_data.keys())

    for f in features:

        if f == 'B_list':
            continue
        
        # print("Feature = ", f, " shape = ", target_home_data[f].shape)
        if len(target_home_data[f].shape) > 1:
            seg_target_home_data[f] = target_home_data[f][seg_idxs, :]
        else:
            seg_target_home_data[f] = target_home_data[f][seg_idxs]

    # print(features)
    # print(list(seg_target_home_data.keys()))
    return seg_target_home_data

def cv_error(real, estimate):
    numerator = np.sqrt(mean_squared_error(real, estimate))
    denominator = np.mean(real)

    try:
        cv = numerator / denominator
    except RuntimeWarning:
        cv = np.nan

    return cv

def rmse_error(real, estimate):
    return np.sqrt(mean_squared_error(real, estimate))

def mape_error(real, estimate):
    light_idx = np.where(real >= 0.1)
    return np.mean(np.abs((real[light_idx] - estimate[light_idx]) / real[light_idx]))


def most_frequent(List):
    counts = np.bincount(List)
    return np.argmax(counts)


###################################

def decompose_to_daily(signal, num_points_per_hour=2):
    num_day = int(signal.shape[0]) // (num_points_per_hour*24)
    num_daily_points = int(num_points_per_hour * 24)
    
    decomposed_signal = np.ones((num_day, num_daily_points)) * np.nan
    
    for i in range(num_day):
        start = num_daily_points * i
        end = num_daily_points * (i+1)
        
        decomposed_signal[i, :] = signal[start:end]
        
    return decomposed_signal


def summation_over_interval_nonduplicate(signal, start_idx, w):
    length = signal.shape[0]
    output = []
    
    start = start_idx
    end = start + w
    
    while(end <= length):
        output.append(np.sum(signal[start:end]))
        
        start = end
        end = start + w
    
    return np.array(output)


def coefficient_matrix_summation_over_interval_nonduplicate(length, start_idx, w):
    start = start_idx
    end = start + w

    item = np.zeros((1, length))
    item[:, start:end] = 1.0
    A = np.copy(item)

    while(True):
        start = end
        end = start + w
        
        if (end > length):
            break
        
        item = np.zeros((1, length))
        item[:, start:end] = 1.0
        
        A = np.append(A, item, axis=0)

    return A



