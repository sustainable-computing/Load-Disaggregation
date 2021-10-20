import cvxpy as cp
import numpy as np

def solar_model(Y, load, batt, proxy_B_list):
    """
        load mixture model for with battery load disaggregation

        @params:
            Y: real net load
            load: estimated load in current iteration
            batt: estimated battery in current iteration
            proxy_B_list: generation of solar proxies

        @return: 
            solar: updated solar estimation
            solar_params: params value of solar model
    """

    solar = np.clip(load - batt - Y, a_min=0, a_max=None)
    b_array = np.array(proxy_B_list).T
    K = cp.Variable(len(proxy_B_list), nonneg=True)
    est_solar = b_array@K
    
    constraints = []
    objective = cp.Minimize(cp.sum_squares(est_solar - solar) + 0.01*cp.norm(K, 1))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.GUROBI, verbose=False, MIPGap=1e-4)
    est_solar = np.clip(np.matmul(b_array, K.value), a_min=0, a_max=None)

    return est_solar, K.value


