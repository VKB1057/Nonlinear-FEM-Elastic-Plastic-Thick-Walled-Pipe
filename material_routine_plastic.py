import numpy as np
from material_routine_elastic import material_stiffness_matrix

def plastic_material_routine(epsilon, epsilon_pl_old, data):

    E = data["E"]
    nu = data["nu"]
    sigma_y = data["sigma_y"]
    plane_condition = data["plane_condition"]

    C = material_stiffness_matrix(E, nu, plane_condition)
    m_vec = np.array([1.0, -1.0])
    epsilon_e_trial = epsilon - epsilon_pl_old
    sigma_trial = C @ epsilon_e_trial
    stress_difference_trial = m_vec @ sigma_trial
    phi_trial = abs(stress_difference_trial) - sigma_y

    if phi_trial <= 0.0:

        sigma = sigma_trial
        C_t = C
        epsilon_pl_new = epsilon_pl_old.copy()
        is_plastic = False

    else:

        sign_value = np.sign(stress_difference_trial)
        n_vec = sign_value * m_vec
        denominator = n_vec.T @ C @ n_vec
        delta_lambda = phi_trial / denominator
        epsilon_pl_new = epsilon_pl_old + delta_lambda * n_vec
        sigma = sigma_trial - delta_lambda * (C @ n_vec)
        C_t = C - np.outer(C @ n_vec, n_vec @ C) / denominator
        is_plastic = True

    return sigma, C_t, epsilon_pl_new, is_plastic