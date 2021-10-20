"""
    Extension based on v3, change load centroid proxies to dynamic load proxy
"""

import random
import copy
import re
import sys
import os.path
import pickle
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from incremental_trees.trees import StreamingRFR
import cvxpy as cp
from sklearn import preprocessing
from sklearn.cluster import KMeans

import utils
from variables import *
from solarTK_tool import *

from solar_modeling import solar_model
from battery_modeling import battery_model
from load_modeling import init_load_withBattery, load_model_withBattery

# import warnings
# warnings.filterwarnings('ignore')

def read_data(params, data_path, resolution, month, synthetic_proxy=True):
    data = {}
    
    weather_data_path = data_path + resolution + "/" + resolution + "_weather_" + month + ".csv"
    data['weather'] = pd.read_csv(weather_data_path, index_col='time')

    solar_data_path = data_path + resolution + "/" + resolution + "_real_solar_" + month + ".csv"
    data['solar'] = pd.read_csv(solar_data_path, index_col='time')

    net_loads_data_path = data_path + resolution + "/" + resolution + "_real_net_load_" + month + ".csv"
    data['net_load'] = pd.read_csv(net_loads_data_path, index_col='time')

    loads_data_path = data_path + resolution + "/" + resolution + "_real_load_" + month + ".csv"
    data['load'] = pd.read_csv(loads_data_path, index_col='time')

    exp_vars_data_path = data_path + resolution + "/" + resolution + "_explanatory_vars_" + month + ".csv"
    col_names = ['c', 'c_2', 'c_3', 'h', 'h_2', 'h_3', 'c_wmv', 'd', 'c_h']
    data['exp_vars'] = pd.read_csv(exp_vars_data_path, header=None, names=col_names)

    battery_data_path = data_path + resolution + "/" + resolution + "_real_{}_".format(params['battery_model']) + month + ".csv"
    data['battery'] = pd.read_csv(battery_data_path, index_col='time')

    if synthetic_proxy:
        sythetic_proxy_data_path = data_path + resolution + "/" + resolution + "_synthetic_proxy_" + month + ".csv"
        df_synthetic_proxy = pd.read_csv(sythetic_proxy_data_path)
        df_synthetic_proxy = df_synthetic_proxy.set_index(data['solar'].index)
        data['solar'] = pd.concat([data['solar'], df_synthetic_proxy], axis=1)

    return data


def update_net_load_with_battery(params, data):
    for home in params['PV_HOMES']:
        data['net_load'][str(home)] = data['net_load'][str(home)] - data['battery'][str(home)]

def read_process_max_solar_generation(params, home, affix):
    max_solar_gen = pd.read_csv(params["max_gen_file_path"].format(str(home), 
                                params['resolution'], params['month'], affix))
    max_solar_gen = max_solar_gen.interpolate(method='linear')
    max_solar_gen = max_solar_gen['max_generation'].to_numpy() / 1000
    max_solar_gen = np.clip(max_solar_gen, a_min=0.0, a_max=None)

    return max_solar_gen


def load_disaggregation_withoutbattery(params, Y, 
                                        proxy_B_list, exp_vars_feature, 
                                        init_k_list,
                                        B, X):
    """
    Load disaggregation without considering battery

    @params:
        params: experiment context info
        Y: real net load
        proxy_B_list: real measurment of solar proxies
        exp_vars_feature: ambient features
        init_k_list: inital weight vector of solar mixture model
        B: real solar (only for checking error purpose)
        X: real load (only for checking error purpose)

    @return:
        result: dict contains all estimated information
    """

    assert len(proxy_B_list) == len(init_k_list)

    ### solar init (start) ###
    # initlization
    k_list = copy.deepcopy(init_k_list)
    last_k_list = k_list

    # trasposed B_list
    B_list = np.array(copy.deepcopy(proxy_B_list)).T
    solar = np.clip(np.matmul(B_list, k_list), a_min=0, a_max=None)
    x_t = np.clip(Y + solar, a_min=0, a_max=None)
    ### solar init (end) ###
    
    exp_vars_scaled = preprocessing.scale(exp_vars_feature[:, :])
    load_model = StreamingRFR(n_estimators_per_chunk=1, dask_feeding=False, max_n_estimators=params['num_iterations'])
    for iter_idx in range(params['num_iterations']):

        # load model
        load_model.partial_fit(exp_vars_scaled, x_t)
        x_t = load_model.predict(exp_vars_scaled)

        # solar model
        solar = np.clip(x_t - Y, a_min=0, a_max=None)
        k_list = utils.compute_weight_between_target_neighbor(solar, proxy_B_list, print_info=False)
        solar = np.clip(np.matmul(B_list, k_list), a_min=0, a_max=None)
        diff = np.sum(np.abs(np.array(last_k_list) - np.array(k_list)))
        if diff < 0.001:
            break
        last_k_list = k_list
        x_t = np.clip(Y + solar, a_min=0, a_max=None)

    min_error_solar = solar
    min_error_load = np.clip(Y + solar, a_min=0, a_max=None)

    print("SOLAR DISAGGREGATION WITHOUT BATTERY, solar RMSE = {}".format(utils.rmse_error(min_error_solar, B)))
    result = {
        "type": "without_battery",
        "solar_estimation": min_error_solar,
        "solar_params": k_list,
        "load_estimation": min_error_load,
        "load_params": None
        }
    return result


def load_disaggregation_withbattery(params, Y, 
                            proxy_B_list, proxy_X_list,
                            init_k_list,
                            B, X, Batt):
    """
    Load disaggregation with considering battery

    @params:
        params: experiment context info
        Y: real net load
        proxy_B_list: real measurment of solar proxies
        proxy_Y_list: real measurment of load proxies (neighboring homes)
        init_k_list: inital weight vector of solar mixture model
        B: real solar (only for checking error purpose)
        X: real load (only for checking error purpose)
        Batt: real battery (only for checking error purpose)

    @return:
        result: dict contains all estimated information
    """
    
    assert len(proxy_B_list) == len(init_k_list)
    
    error_fn = utils.rmse_error
    
    length = Y.shape[0]
    num_points_in_one_day = int(24 * params['num_points_per_hour'])
    
    
    ########### INITIALIZATION ###########
    
    ### solar init (start) ###
    k_list = copy.deepcopy(init_k_list)
    B_list = np.array(copy.deepcopy(proxy_B_list)).T
    solar = np.matmul(B_list, k_list)
    last_k_list = k_list
    ### solar init (end) ###
    
    ### load init (start) ###
    load = init_load_withBattery(Y, solar, proxy_X_list, params['num_days'])
    ### load init (end) ###

    ### battery init (start) ###
    batt = load - solar - Y
    ### battery init (end) ###
    
    print("iter:{}, Weight diff:{:.6f}, solar:{:.4f}, load:{:.4f}".format(-1, -1, error_fn(B, solar), error_fn(X, load)))

    k = 0
    for solar_iter_idx in range(params['num_iterations']):
        n = 0
        for iter_idx in range(params['num_iterations']):
            
            ########## iterative ###########

            ### update load (start) ###
            load, load_params = load_model_withBattery(Y, solar, batt, proxy_X_list, params['num_days'])
            ### update load (end) ###
            
            ### update battery (start) ###
            batt, batt_params = battery_model(Y, solar, load, params['num_days'])
            ### update battery (end) ###

            if iter_idx == 0 and solar_iter_idx == 0:
                last_k_list = k_list
                last_batt_params = batt_params
                continue
            
            diff = np.sum(np.abs(last_batt_params - batt_params)) / batt_params.shape[0]
            # msg = "inner iter:{}, battery model params:{:.6f}, solar:{:.4f}, load:{:.4f}"
            # print(msg.format(iter_idx, diff, error_fn(B, solar), error_fn(X, load)))
            
            if diff < 0.001:
                break

            last_batt_params = batt_params

        ### update solar (start) ###
        solar, k_list = solar_model(Y, load, batt, proxy_B_list)
        ### update solar (end) ###
        
        solar_diff = np.sum(np.abs(np.array(last_k_list) - np.array(k_list)))
        print("outer iter:{}, solar model params diff = {:.6f}, solar error = {:.4f}, load error = {:.4f}".format(solar_iter_idx, solar_diff, error_fn(B, solar), error_fn(X, load)))
        if solar_diff < 0.01:
            break
        last_k_list = k_list


    min_error_solar, min_error_load, min_error_batt  = solar, load, batt

    result = {
        "type": "with_battery",
        "solar_estimation": min_error_solar,
        "solar_params": k_list,
        "load_estimation": min_error_load,
        "load_params": None,
        "battery_estimation": min_error_batt,
        "battery_params": batt_params
    }
    
    return result


def load_disaggregation(params, target_home_data, 
                        init_k_list, 
                        mode=""): 
    """
        Do the load disaggregation in segment unit.

        @params:
            params: experiment context info
            target_home_data: the data for target customer
            init_k_list: initial weight vector value of solar model
            mode: current mode

        @return:
            estimated_solar: disaggregated solar
            estimated_load: disaggregated home demand
            estimated_battery: disaggregated battery acitivty
    """

    ## experiment configurations
    segment_length = params['segment_length']
    num_seg = int(target_home_data['length'] / segment_length)
    
    ## experiment data
    B_list = copy.deepcopy(target_home_data['B_list'])
    X_list = copy.deepcopy(target_home_data['X_list'])

    # variables for storing final result
    final_estimated_b = np.empty((target_home_data['length'], ))
    final_estimated_x = np.empty((target_home_data['length'], ))
    final_estimated_batt = np.empty((target_home_data['length'], ))

    # load disaggregation by segment
    seg_end_time = params['start_time']
    for s_i in range(num_seg):
        print("Section {} ---------- seg_idx = {} ----------".format(mode, s_i), end='\r')
        current_seg = "seg_{}".format(s_i)

        seg_idxs = np.arange(s_i * segment_length, (s_i+1) * segment_length)
        
        seg_Y = target_home_data['Y'][seg_idxs] # real net load
        seg_target_B = target_home_data['target_B'][seg_idxs] # real solar
        seg_B_list = utils.neigh_list_segmentation_same_time(B_list, seg_idxs) # solar proxies
        seg_X = target_home_data['X'][seg_idxs] # real load 
        seg_X_list = X_list[:, seg_idxs] # load proxies
        seg_Batt = target_home_data['Batt'][seg_idxs] # real battery
        
        # seg_weather = target_home_data['weather'][seg_idxs, :] # weather data
        seg_exp_vars = target_home_data['exp_vars'][seg_idxs, :] # ambient data
        # seg_start_time = seg_end_time 
        # seg_end_time =  utils.time_increment(seg_start_time, delta_day=int(segment_length/(params['num_points_per_hour']*24)))

        if params['is_battery'] is True:
            # do the disaggregation assuming isBattery = 1
            result = load_disaggregation_withoutbattery(params, 
                                                        seg_Y, 
                                                        seg_B_list, seg_exp_vars,
                                                        init_k_list,
                                                        seg_target_B, seg_X)
        
            result = load_disaggregation_withbattery(params, 
                                                    seg_Y, 
                                                    seg_B_list, seg_X_list,
                                                    result['solar_params'],
                                                    seg_target_B, seg_X, seg_Batt)

            final_estimated_b[seg_idxs] = result['solar_estimation']
            final_estimated_x[seg_idxs] = result['load_estimation']
            final_estimated_batt[seg_idxs] = result['battery_estimation']
        else:
            # do the disaggregation assuming isBattery = 0
            result = load_disaggregation_withoutbattery(params, 
                                                        seg_Y, 
                                                        seg_B_list, seg_exp_vars,
                                                        init_k_list,
                                                        seg_target_B, seg_X)
            final_estimated_b[seg_idxs] = result['solar_estimation']
            final_estimated_x[seg_idxs] = result['load_estimation']
        
    # set solar generation to zero at night time
    # final_estimated_b[np.where(target_home_data['weather'][:, 0] == 0.0)] = 0

    return final_estimated_b, final_estimated_x, final_estimated_batt               


def estimate_k(params, target_home, target_home_data, proxies):
    """
        Method for estimating K which is used in proxy transposition model

        Return:
            initial weight vector value of solar model
    """
    
    # roughly guess K using maximum generation
    max_solar_gen = read_process_max_solar_generation(params, target_home, affix='modifiedNetLoad')

    max_solar_gen_nei_list = []
    for nei in proxies:
        max_solar_gen_nei_list.append(read_process_max_solar_generation(params, nei, affix='solar'))

    k_list = utils.compute_ratio_between_target_neighbor(max_solar_gen, max_solar_gen_nei_list, 
                                                 print_info=False)

    return np.array(k_list).T / len(k_list)


def select_proxy(target_home, num_proxy, proxy_pool, pv_system_params, homes_in_same_clusters):

    # exclude target home as proxy
    pool = list(set(proxy_pool) - set([target_home]))

    # default choice: random select from pool
    if (pv_system_params is None) and (homes_in_same_clusters is None):
        proxy_result = random.sample(pool, k=num_proxy)
    elif not homes_in_same_clusters is None:
        candidates = list(set(pool).intersection(set(homes_in_same_clusters)))
        proxy_result = random.sample(candidates, k=num_proxy)

    return proxy_result

def find_home_in_same_cluster(target_home, customer_info_file, n_clusters):
    customer_df = pd.read_csv(customer_info_file)

    # find index of target home
    target_home_idx = customer_df.loc[(customer_df['Customer'] == target_home)].index.values[0]

    geos = customer_df[['latitude', 'longitude']].to_numpy()
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(geos)

    target_home_cluster_idx = kmeans.labels_[target_home_idx]
    same_cluster_homes_idxs = np.where(kmeans.labels_ == target_home_cluster_idx)
    result = customer_df.iloc[same_cluster_homes_idxs]['Customer'].to_numpy()

    return [home for home in result if home != target_home]

def estimate_maximum_generation(params, target_home, proxies, data, target_home_data):
    
    input_file_list = generate_input_file(params, target_home, proxies, data, GHI_cutoff=params['GHI_cutoff'])
    
    # pv system estimation
    for input_file in input_file_list:
        # print(input_file)

        components = re.split(r"[_/.]+", input_file)
        # print("input_file", input_file, components)
        # print(components)
        max_file_output = "{}max_solar_gen/{}_phase_2_{}_{}_{}.csv"\
                            .format(params['max_gen_store_path'], components[6], components[9], components[10], components[11])
        # print("max_file_output = ", max_file_output)
        if not os.path.isfile(max_file_output):
            pv_params = get_pv_system_parameters(lat=params['lat'], lon=params['lon'], hemisphere=params['hemisphere'], data_file=input_file)
            print(max_file_output)
            print(pv_params)
            output_maximum_solar_generation(pv_params, params["start"], params["end"], params["resolution_second"], max_file_output)


def run_experiments(params, PV_homes):

    result_store = {}

    ### before experiment
    np.random.seed(0)

    # read all data as dataframe format (e.x. weather_df)
    data = read_data(params=params,
                    data_path=params['data_path'], 
                    resolution=params['resolution'], 
                    month=params['month'],
                    synthetic_proxy=True)
    
    # imply battery activity into net load, thus NL = L - G - B
    if params['is_battery']:
        update_net_load_with_battery(params, data)
    
    load_proxy = np.transpose(data['load'][[str(x) for x in params['load_proxy']]].to_numpy())
    print("shape of load_proxy = ", load_proxy.shape) # 20 * 1440
    ###

    home_candidates = params['PV_HOMES'] * 10  
    for exp_idx in range(params['num_experiments']):
        print("------ EXPERIMENT {} ------".format(exp_idx))

        result_store["exp_{}".format(exp_idx)] = {}

        # random pick target home
        # if len(max_K_estimation_result_store["exp_{}".format(exp_idx)].keys()) > 0:
        #     target_home_list = [int(list(max_K_estimation_result_store["exp_{}".format(exp_idx)].keys())[0])]
        # else:
        #     continue
        target_home_list = [home_candidates[exp_idx]]
        print(target_home_list)


        for home_idx, target_home in enumerate(target_home_list):
            print("--------- Disaggregate target home {} ---------".format(str(target_home)))

            if params['data_set']['abbr'] == "Ausgrid":
                homes_in_same_clusters = find_home_in_same_cluster(target_home, 
                                            customer_info_file=params['data_path']+params['resolution']+"/"+"customer_info.csv", 
                                            n_clusters=params['n_clusters'])
                # print(homes_in_same_clusters)
            
            elif params['data_set']['abbr'] == "Pecan":
                homes_in_same_clusters = None
            
            try:
                proxies = select_proxy(target_home, num_proxy=params['num_proxy'], proxy_pool=PV_homes, 
                                        pv_system_params=None, homes_in_same_clusters=homes_in_same_clusters)
            except Exception as e:
                print("Occured error in select_proxy(): ", e)
                continue
            # proxies = max_K_estimation_result_store["exp_{}".format(exp_idx)][str(target_home)]["proxy"]
            # proxies = [proxies[0], 10000, 10090, 10270]
            print("Proxy = ", proxies)

            ### sepcific data for target home
            target_home_data = {
                "target_home_id": target_home,
                "target_B": data['solar'][str(target_home)].to_numpy(),
                "proxies": proxies,
                "B_list": [data['solar'][str(nei)].to_numpy() for nei in proxies],
                "Y": data['net_load'][str(target_home)].to_numpy(),
                "X": data['load'][str(target_home)].to_numpy(),
                "X_list": load_proxy,
                "Batt": data['battery'][str(target_home)].to_numpy(),
                "weather": data['weather'][params['weather_feature']].to_numpy(),
                "exp_vars": data['exp_vars'].to_numpy(),
            }
            target_home_data['length'] = target_home_data['X'].shape[0]
            ###

            ####### estimate pv params and maximum solar generation
            # try:
            estimate_maximum_generation(params, target_home, proxies, data, target_home_data)
            # except Exception as e:
            #     print("Occured error in estimate_maximum_generation(): ", e)
            #     continue
            
            
            result_store["exp_{}".format(exp_idx)][str(target_home)] = {}

            #### solar init ######
            try:
                k_list = estimate_k(params, target_home, target_home_data, proxies) 
            except Exception as e:
                print("Occured error in read_maximum_generation(): ", e)
                continue
            #####################

            # try:
            estimated_solar, estimated_load, estimated_batt = load_disaggregation(params, target_home_data, 
                                                                                k_list, 
                                                                                mode="with battery")
            # except Exception as e:
            #     print("Occcured error in solar disaggregation(): ", e)
            #     print("target home is {}, proxies are {}".format(target_home, proxies))
            #     result_store["exp_{}".format(exp_idx)][str(target_home)]['proxy'] = proxies
            #     continue

            # store the result
            result_store["exp_{}".format(exp_idx)][str(target_home)]['proxy'] = proxies
            result_store["exp_{}".format(exp_idx)][str(target_home)]['estimated_solar'] = estimated_solar
            result_store["exp_{}".format(exp_idx)][str(target_home)]['estimated_load'] = estimated_load
            if params['is_battery']:
                result_store["exp_{}".format(exp_idx)][str(target_home)]['estimated_batt'] = estimated_batt

            error_fn = utils.rmse_error
            error_name = "RMSE"
            print("------ {} --------".format(error_name))
            print("Solar {} error for baseline method = {}".format(error_name, error_fn(target_home_data['target_B'], estimated_solar)))
            print("Load {} error for baseline method = {}".format(error_name, error_fn(target_home_data['X'], estimated_load)))
            if params['is_battery']:
                print("Batt {} error for baseline method = {}".format(error_name, error_fn(target_home_data['Batt'], estimated_batt)))

    return result_store
    


if __name__ == '__main__':

    ########## Parameters ###########
    params = EXPERIMENT_DATASET["ausgrid_summer"]

    params['weather_feature'] = ['GHI', 'Temperature']

    params['segment_length'] = 1440  # length for separate training
    params['num_experiments'] = 1
    params['num_proxy'] = 3
    params['num_iterations'] = 200
    params['n_clusters'] = 3  # specific to ausgrid dataset

    params['is_battery'] = False
    params['battery_model'] = 'batt1'
    params['max_gen_store_path'] = 'modified_net_meter_batt_v{}/'.format(params['battery_model'][-1])
    params['max_gen_file_path'] =  params['max_gen_store_path'] + "max_solar_gen/{}_phase_2_{}_{}_{}.csv"

    params['load_proxy'] = [12, 27, 31, 36, 43, 44, 76, 78, 82, 100, 114, 134, 147, 161, 180, 185, 190, 235, 241, 295]
    ###########################################

    # ### TEMPORARY conduct a comparable experiment ####
    # max_K_estimation_result_store_file = "result_store_AusgridData_experiment_choices.pkl"
    # f = open(max_K_estimation_result_store_file , "rb")
    # max_K_estimation_result_store = pickle.load(f)
    # f.close()
    # #########

    params['PV_HOMES'] =  list(set(params['PV_HOMES']) - set(params['load_proxy']))
    result_store = run_experiments(params=params, PV_homes=params['PV_HOMES'])


