import cvxpy as cp
import numpy as np
import utils

def battery_model(Y, solar, load, num_days):
    """
        load mixture model for with battery load disaggregation

        @params:
            Y: real net load
            solar: estimated solar in current iteration
            load: estimated load in current iteration
            num_days

        @return: 
            batt: updated batt estimation
            batt_parmas: params value of battery model

    """

    length = Y.shape[0]
    num_points_in_one_day = int(length / num_days)

    batt = load - solar - Y
    estimated_batt_rate = np.max(-batt)

    segment_per_day = 12 # 24H -> 4, 12H -> 8, 8H -> 12, 4H -> 24, 2H -> 48
    dividend = int(num_points_in_one_day/segment_per_day)
    
    alpha_day_list = cp.Variable(int(num_days*segment_per_day))
    beta_day_list = cp.Variable(int(num_days*segment_per_day))
    
    est_batt = cp.Variable(length)
    constraints = []
    for one_day_freq_i in range(num_days):
        for t in range(num_points_in_one_day):

            sector_idx = int(t // dividend)

            #### 8H-sin + 8H-cos
            constraints += [est_batt[(one_day_freq_i*num_points_in_one_day)+t] == np.sin(2*np.pi*t / (num_points_in_one_day/3 - (1/3))) * alpha_day_list[one_day_freq_i*segment_per_day + sector_idx] +
                        np.cos(2*np.pi*t / (num_points_in_one_day/3 - (1/3))) * beta_day_list[one_day_freq_i*segment_per_day + sector_idx]]


    mvs_A_one_day = utils.coefficient_matrix_summation_over_interval_nonduplicate(
                    length=length,
                    start_idx=0,
                    w=int(num_points_in_one_day))


    error = cp.sum_squares(est_batt - batt) / batt.shape[0]
    error2 = cp.sum_squares(mvs_A_one_day @ est_batt) / est_batt.shape[0]
    constraints += [cp.max(cp.abs(est_batt)) <= estimated_batt_rate]

    objective = cp.Minimize(error + 0.01*error2)
    prob = cp.Problem(objective, constraints)
    result = prob.solve(solver=cp.GUROBI, verbose=False, MIPGap=1e-4)


    return est_batt.value, np.hstack((alpha_day_list.value, beta_day_list.value))












