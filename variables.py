import utils

DATASETS_DICT = {
    "Pecan":{
        "name": "PecanStreet Dataset",
        "abbr": "Pecan",
        "location": "Austin, Texas, USA",
        "number_homes": 19,
        "homes": [661, 1642, 2335, 2361, 2818, 3039, 3456, 3538, 4031,
                    4373, 4767, 6139, 7536, 7719, 7800, 8156, 9019, 9278, 9160]
    },

    "Ausgrid":{
        "name": "Ausgrid Utility Dataset",
        "abbr": "Ausgrid",
        "location": "Sydney, New South Wales, Australia",
        "number_homes": 150, 
        "homes": [1, 4, 6, 8, 11, 12, 16, 17, 18, 27, 31, 32, 34, 
                    36, 40, 43, 44, 45, 46, 48, 53, 55, 56, 60, 62, 
                    63, 65, 66, 71, 74, 76, 78, 79, 80, 81, 82, 83, 
                    89, 91, 92, 94, 95, 96, 97, 98, 99, 100, 102, 105, 
                    109, 112, 113, 114, 115, 116, 117, 118, 120, 123, 
                    126, 128, 129, 131, 132, 134, 135, 136, 139, 140, 
                    141, 142, 147, 149, 150, 154, 156, 159, 161, 162, 
                    164, 166, 172, 173, 174, 179, 180, 181, 182, 183, 
                    185, 187, 190, 191, 192, 193, 194, 195, 197, 198, 
                    199, 204, 205, 208, 209, 210, 213, 216, 217, 219, 
                    220, 221, 226, 228, 229, 233, 234, 235, 236, 237, 
                    241, 242, 243, 244, 247, 250, 251, 254, 255, 257, 
                    258, 260, 262, 264, 268, 274, 275, 277, 278, 279, 
                    280, 282, 287, 288, 291, 294, 295, 296, 298, 299, 300],
    }
}

EXPERIMENT_DATASET = {
    "pecan_summer":{
        "data_set":             DATASETS_DICT["Pecan"],
        "data_path":            utils.read_data_path("data_path.txt").rstrip(),
        "max_gen_file_path":    "modified_net_meter_v3/max_solar_gen/{}_phase_2_{}_{}_{}.csv",
        "max_gen_store_path":   "modified_net_meter_v3/",
        "resolution":           "15min",
        "month":                "jun",
        "start_time":           "2018-06-01",
        "end_time":             "2018-07-01",
        "num_days":             30,
        "num_points_per_hour":  4,
        "GHI_cutoff":           500,

        "lat":                  30.292432,
        "lon":                  -97.699662,
        "hemisphere":           "North",
        "start":                "2018-06-01 00:00:00",
        "end":                  "2018-06-30 23:45:00",
        "resolution_second":    "900",
        "length":               2880,
        

        "PV_HOMES":             [661,1642,2335,2361,2818,3039,3456,3538,4031,4373,4767, 
                                6139,7536,7719,7800,8156,9278,9160]

        # # due to 9019 get bad performance
        # "PV_HOMES":             [661,1642,2335,2361,2818,3039,3456,3538,4031,4373,4767, 
        #                         6139,7536,7719,7800,8156,9019,9278,9160]
    },

    "pecan_summer_1min":{
        "data_set":             DATASETS_DICT["Pecan"],
        "data_path":            utils.read_data_path("data_path.txt").rstrip(),
        "max_gen_file_path":    "modified_net_meter_v3/max_solar_gen/{}_phase_2_{}_{}_{}.csv",
        "max_gen_store_path":   "modified_net_meter_v3/",
        "resolution":           "1min",
        "month":                "jun",
        "start_time":           "2018-06-01",
        "end_time":             "2018-07-01",
        "num_days":             30,
        "num_points_per_hour":  60,
        "GHI_cutoff":           500,

        "lat":                  30.292432,
        "lon":                  -97.699662,
        "hemisphere":           "North",
        "start":                "2018-06-01 00:00:00",
        "end":                  "2018-06-30 23:59:00",
        "resolution_second":    "60",
        "length":               43200,
        

        "PV_HOMES":             [661,1642,2335,2361,2818,3039,3456,3538,4031,4373,4767, 
                                6139,7536,7719,7800,8156,9278,9160]

        # # due to 9019 get bad performance
        # "PV_HOMES":             [661,1642,2335,2361,2818,3039,3456,3538,4031,4373,4767, 
        #                         6139,7536,7719,7800,8156,9019,9278,9160]
    },

    "pecan_winter":{
        "data_set":             DATASETS_DICT["Pecan"],
        "data_path":            utils.read_data_path("data_path.txt").rstrip(),
        "max_gen_file_path":    "modified_net_meter_v3/max_solar_gen/{}_phase_2_{}_{}_{}.csv",
        "max_gen_store_path":   "modified_net_meter_v3/",
        "resolution":           "15min",
        "month":                "dec",
        "start_time":           "2018-12-03",
        "end_time":             "2018-12-31",
        "num_days":             28,
        "num_points_per_hour":  4,
        "GHI_cutoff":           300,

        "lat":                  30.292432,
        "lon":                  -97.699662,
        "hemisphere":           "North",
        "start":                "2018-12-03 00:00:00",
        "end":                  "2018-12-30 23:45:00",
        "resolution_second":    "900",
        "length":               2688,
        

        "PV_HOMES":             [661,1642,2335,2361,2818,3039,3456,3538,4031,4373,4767, 
                                6139,7536,7719,7800,8156,9278,9160]

        # # due to 9019 get bad performance
        # "PV_HOMES":             [661,1642,2335,2361,2818,3039,3456,3538,4031,4373,4767, 
        #                         6139,7536,7719,7800,8156,9019,9278,9160]

    },

    "ausgrid_summer":{
        "data_set":             DATASETS_DICT["Ausgrid"],
        "data_path":            utils.read_data_path("data_path_2.txt").rstrip(),
        "max_gen_file_path":    "modified_net_meter_v3/max_solar_gen/{}_phase_2_{}_{}_{}.csv",
        "max_gen_store_path":   "modified_net_meter_v3/",
        "resolution":           "30min",
        "month":                "nov",
        "start_time":           "2012-11-01",
        "end_time":             "2012-12-01",
        "num_days":             30,
        "num_points_per_hour":  2,
        "GHI_cutoff":           500,

        "lat":                  -33.888575,
        "lon":                  151.187349,
        "hemisphere":           "South",
        "start":                "2012-11-01 00:00:00",
        "end":                  "2012-11-30 23:30:00",
        "resolution_second":    "1800",
        "length":               1440,

        "num_homes":            136,
        "PV_HOMES":             [  1,   4,   8,  11,  12,  16,  17,  18,  27,  31,  32,  34,
                                36,  40,  43,  44,  45,  46,  48,  53,  55,  56,  60,  62,  63,
                                65,  66,  71,  74,  76,  78,  80,  81,  82,  83,  89,  91,
                                94,  95,  96,  98,  99, 100, 102, 105, 109, 112, 113,
                               114, 115, 117, 118, 120, 123, 126, 128, 129, 131, 132, 134,
                               135, 136, 139, 140, 141, 142, 147, 149, 150, 154, 156, 159, 161,
                               162, 164, 172, 173, 179, 180, 181, 182, 185, 187,
                               190, 191, 192, 193, 195, 197, 198, 199, 204, 205, 208, 209,
                               210, 213, 216, 217, 219, 220, 226, 228, 229, 233, 235,
                               236, 237, 241, 242, 243, 244, 250, 251, 254, 255, 257, 258,
                               260, 264, 268, 274, 275, 277, 278, 279, 280, 282, 287, 288,
                               291, 294, 295, 296, 298, 300],

    },

    "ausgrid_winter":{
        "data_set":             DATASETS_DICT["Ausgrid"],
        "data_path":            utils.read_data_path("data_path_2.txt").rstrip(),
        "max_gen_file_path":    "modified_net_meter_v3/max_solar_gen/{}_phase_2_{}_{}_{}.csv",
        "max_gen_store_path":   "modified_net_meter_v3/",
        "resolution":           "30min",
        "month":                "may",
        "start_time":           "2013-05-01",
        "end_time":             "2013-05-31",
        "num_days":             30,
        "num_points_per_hour":  2,
        "GHI_cutoff":           300,

        "lat":                  -33.888575,
        "lon":                  151.187349,
        "hemisphere":           "South",
        "start":                "2013-05-01 00:00:00",
        "end":                  "2013-05-30 23:30:00",
        "resolution_second":    "1800",
        "length":               1440,

        "num_homes":            142,
        "PV_HOMES":             [1, 4, 6, 8, 11, 12, 16, 17, 18, 27, 31, 32, 34, 
                                    36, 40, 43, 44, 45, 46, 48, 53, 55, 56, 60, 62, 
                                    63, 65, 66, 71, 74, 76, 78, 79, 80, 81, 82, 83, 
                                    89, 92, 94, 95, 96, 98, 99, 100, 102, 105, 
                                    109, 112, 113, 114, 115, 116, 117, 118, 120, 123, 
                                    126, 128, 129, 131, 132, 134, 135, 136, 139, 140, 
                                    141, 142, 147, 149, 150, 154, 156, 159, 161, 162, 
                                    164, 166, 172, 173, 174, 179, 180, 181, 182, 183, 
                                    185, 190, 191, 192, 193, 194, 195, 197, 198, 
                                    199, 204, 205, 208, 209, 210, 213, 216, 217, 219, 
                                    220, 226, 228, 233, 234, 235, 236, 237, 
                                    241, 242, 243, 244, 247, 251, 254, 255, 257, 
                                    258, 260, 262, 264, 268, 274, 277, 278, 279, 
                                    280, 282, 287, 288, 291, 294, 295, 296, 298, 299],
    }
}



