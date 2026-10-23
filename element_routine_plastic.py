import numpy as np

from kinematics import gauss_rule
from kinematics import B_matrix
from kinematics import strain_function
from material_routine_plastic import plastic_material_routine

def plastic_element_routine(u_e, r_e, epsilon_pl_old_e, data):

    xi, w = gauss_rule(data["n_gp"])
    B, J, r, N = B_matrix(xi, r_e)
    epsilon = strain_function(B, u_e)
    epsilon_pl_old_gp = epsilon_pl_old_e[0]

    sigma, C_t, epsilon_pl_new_gp, is_plastic = plastic_material_routine(epsilon, epsilon_pl_old_gp, data)
    F_int_e = (B.T @ sigma) * r * J * w
    K_t_e = ((B.T @ C_t) @ B) * r * J * w

    epsilon_pl_new_e = epsilon_pl_new_gp.reshape(1, 2)
    stress_e = sigma.reshape(1, 2)
    r_gp_e = np.array([r])
    plastic_flag_e = np.array([is_plastic], dtype=bool)

    return F_int_e, K_t_e, epsilon_pl_new_e, stress_e, r_gp_e, plastic_flag_e