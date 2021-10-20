import cvxpy as cp
import numpy as np
import utils

def init_load_withBattery(Y, solar, proxy_X_list, num_days):
    """
        load initialization for with battery load disaggregation

        @params:
            Y: real net load
            solar: intialized solar 
            proxy_X_list: real home demand of load proxies
            num_days

        @return: 
            load: updated load estimation
            load_params: params value of load model
    """

    length = Y.shape[0]
    num_points_in_one_day = int(length / num_days)

    load_mvs_one_day = utils.summation_over_interval_nonduplicate(
                signal=np.clip(Y + solar, a_min=0, a_max=None), 
                start_idx=0,
                w=int(num_points_in_one_day))
    mvs_A_one_day = utils.coefficient_matrix_summation_over_interval_nonduplicate(
                length=length,
                start_idx=0,
                w=int(num_points_in_one_day))

    # calculate summation
    proxy_load_mvs_one_day_list = np.ones((proxy_X_list.shape[0], mvs_A_one_day.shape[0]))
    for i in range(proxy_X_list.shape[0]):
        load_proxy_mvs = np.matmul(mvs_A_one_day, np.transpose(proxy_X_list[i, :]))
        proxy_load_mvs_one_day_list[i, :] = load_proxy_mvs
        
    chosen_proxy_X_list = proxy_X_list
    chosen_porxy_X_mvs = proxy_load_mvs_one_day_list

    cp_theta_list = cp.Variable(proxy_X_list.shape[0], nonneg=True)
    est_load = chosen_proxy_X_list.T @ cp_theta_list
    est_load_mvs = chosen_porxy_X_mvs.T @ cp_theta_list
    objective = cp.Minimize(cp.sum_squares(est_load_mvs - load_mvs_one_day))
    prob = cp.Problem(objective)
    prob.solve(solver=cp.GUROBI, verbose=False, MIPGap=1e-4)

    load = np.matmul(chosen_proxy_X_list.T, cp_theta_list.value)

    return load

def load_model_withBattery(Y, solar, batt, proxy_X_list, num_days):
    """
        load mixture model for with battery load disaggregation

        @params:
            Y: real net load
            solar: estimated solar in current iteration
            batt: estimated battery in current iteration
            proxy_X_list: real home demand of load proxies
            num_days

        @return: 
            load: updated load estimation
            load_params: params value of load model
    """
    length = Y.shape[0]
    num_points_in_one_day = int(length / num_days)
    load_mvs_one_day = utils.summation_over_interval_nonduplicate(
                        signal=np.clip(Y + solar, a_min=0, a_max=None), 
                        start_idx=0,
                        w=int(num_points_in_one_day))
    mvs_A_one_day = utils.coefficient_matrix_summation_over_interval_nonduplicate(
                length=length,
                start_idx=0,
                w=int(num_points_in_one_day))

    chosen_proxy_X_list = proxy_X_list

    ### daily
    cp_theta_list = cp.Variable((num_days, chosen_proxy_X_list.shape[0]), nonneg=True)
    est_load = cp.Variable(length)
    constraints = []
    for day_i in range(num_days):
        start_idx = day_i * num_points_in_one_day
        end_idx = start_idx + num_points_in_one_day

        chosen_proxy_X_list_segment = chosen_proxy_X_list[:, start_idx:end_idx]
        constraints += [est_load[start_idx:end_idx] == cp.reshape(cp_theta_list[day_i, :] @ chosen_proxy_X_list_segment, (num_points_in_one_day, ))]

    ### whole periods
    error1 = cp.sum_squares((est_load - solar - batt) - Y) / Y.shape[0]
    error2 = cp.sum_squares(mvs_A_one_day @ est_load - load_mvs_one_day) / load_mvs_one_day.shape[0]
    objective = cp.Minimize(error1 + 0.1*error2)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.GUROBI, verbose=False, MIPGap=1e-4)
    load = est_load.value

    return load, None
