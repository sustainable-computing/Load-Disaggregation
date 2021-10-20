import sys
sys.path.insert(1, '../solar-tk/solartk')
import os.path
import datetime
import csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from parameters import ParameterModeling
from maximum_generation import GenerationPotential
from weather_adjusted_ray import WeatherAdjustedGeneration


def generate_input_file(params, target_home, proxies, data_dict, GHI_cutoff=500):

    one_day_length = params['num_points_per_hour'] * 24

    input_file_list = []
    
    # directly using solar generation for proxies to estimate pv_params
    for nei in proxies:
        output_file = params['max_gen_store_path'] + "input/" + str(nei) + \
                        "_phase_1_" + params['resolution'] + "_" + params['month'] + "_solar" + ".csv"
        # print(output_file)
        input_file_list.append(output_file)

        df = pd.concat([data_dict['solar'][str(nei)], data_dict['weather']], axis=1)
        df['idx'] = pd.Series(np.arange(0, len(df), 1), index=df.index)
        max_solar_day_idx = df['idx'][df[str(nei)].idxmax()] // one_day_length
        max_solar_t_idx = df['idx'][df[str(nei)].idxmax()]

        while True:
            bound_GHI_cut_points_idxs = df['idx'][(df['GHI'] >= GHI_cutoff) & (df['idx'] >= (max_solar_day_idx*one_day_length)) & (df['idx'] < ((max_solar_day_idx+1)*one_day_length))].to_numpy()
            # print("nei = ", nei)
            # print(bound_GHI_cut_points_idxs)
            lower_bound_GHI_cut_points_idx = bound_GHI_cut_points_idxs[0]
            upper_bound_GHI_cut_points_idx = bound_GHI_cut_points_idxs[-1]
            
            df_one_day = df[(df['idx'] >= lower_bound_GHI_cut_points_idx) & (df['idx'] <= upper_bound_GHI_cut_points_idx)]
            # print("nei{}, len{}".format(str(nei), len(df_one_day)))

            if len(df_one_day) > 3:
                break

            GHI_cutoff = GHI_cutoff - 50

        df_one_day.to_csv(output_file, columns=[str(nei)], header=['solar'], index=True, index_label="time")

    df = pd.concat([data_dict['net_load'][str(target_home)], data_dict['weather']], axis=1)


    base_consume = df[df['GHI'] == 0.0][str(target_home)].min()

    df['modified_net'] = base_consume - df[str(target_home)]
    df['idx'] = pd.Series(np.arange(0, len(df), 1), index=df.index)
    max_solar_day_idx = df['idx'][df['modified_net'].idxmax()] // one_day_length

    while True:
        bound_GHI_cut_points_idxs = df['idx'][(df['GHI'] >= GHI_cutoff) & (df['idx'] >= (max_solar_day_idx*one_day_length)) & (df['idx'] < ((max_solar_day_idx+1)*one_day_length))].to_numpy()
        lower_bound_GHI_cut_points_idx = bound_GHI_cut_points_idxs[0]
        upper_bound_GHI_cut_points_idx = bound_GHI_cut_points_idxs[-1]
        
        df_one_day = df[(df['idx'] >= lower_bound_GHI_cut_points_idx) & (df['idx'] <= upper_bound_GHI_cut_points_idx)]


        if len(df_one_day) > 3 and df_one_day['modified_net'].gt(0.0).sum() > 1:
            break

        if GHI_cutoff <= 0:
            break

        GHI_cutoff = GHI_cutoff - 50
    
    output_file = params['max_gen_store_path'] + "input/" + str(target_home) + \
                        "_phase_1_" + params['resolution'] + "_" + params['month'] + "_modifiedNetLoad" + ".csv"
    df_one_day.to_csv(output_file, columns=['modified_net'], header=['solar'], index=True, index_label="time")
    input_file_list.append(output_file)

    # exit()

    return input_file_list



def get_pv_system_parameters(lat, lon, hemisphere, data_file):

    # initialize the file name, latitude, and longitude
    parameters = ParameterModeling(latitude=lat, longitude=lon, data_file=data_file)

    # gather sun position, clearsky, and temperature data at the start
    parameters.get_onetime_data()

    # preprocess data to remove night times and first/last hours
    parameters.preprocess_data()

    # find paramaters
    k_, tilt_, ori_ = parameters.find_parameters(hemisphere=hemisphere)
    # k_, tilt_, ori_ = parameters.find_parameters_by_fixed_tilt_orientation(hemisphere=hemisphere)

    t_base, c_ = parameters.find_temp_coefficients(k_, tilt_, ori_)

    pv_params = {
        "latitude": lat,
        "longitude": lon,
        "module_area": k_/0.18,
        "tilt": abs(tilt_),
        "orientation": ori_,
        "temperature_coefficient": c_,
        "baseline_temperature": t_base
    }

    return pv_params



def output_maximum_solar_generation(pv_params, start_time, end_time, resolution, output_file):

    # create an object of GenerationPotential class
    gen = GenerationPotential(k=pv_params['module_area'], 
                            tilt=pv_params['tilt'], 
                            orientation=pv_params['orientation'], 
                            temperature_coefficient=pv_params['temperature_coefficient'], 
                            latitude=pv_params['latitude'], 
                            longitude=pv_params['longitude'], 
                            baseline_temperature=pv_params['baseline_temperature'])


    start_time_ = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    end_time_ = datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")

    gen.maximum_generation_v2(start_time=start_time_, end_time=end_time_, granularity=float(resolution), output_file=output_file)



def output_weather_adjusted_solar_generation(lat, lon, maximum_gen_file, weather_source, output_file):

    # read data from file, split by line, and split each line by comma
    fp = open(maximum_gen_file, "r") 
    data = pd.DataFrame([line for line in csv.reader(fp)])
    fp.close()

    data = data.reset_index(drop=True)

    # set first row as column which contain #time, max_generation
    data.columns = data.iloc[0]
    data = data.reindex(data.index.drop(0)).reset_index(drop=True)
    data.columns.name = None
    data = data.replace(to_replace='None', value=np.nan).dropna()

    # convert time column to datetime
    data['time'] = pd.to_datetime(data['#time'])
    data = data[['time', 'max_generation']]

    # create an object of GenerationPotential class    
    weather = WeatherAdjustedGeneration(latitude=lat, longitude=lon)
    weather.set_data_sources(weather_source=weather_source)

    # compute weather adjusted generation
    weather.adjusted_weather_generation_v2(max_generation=data, output_file=output_file)


