import numpy as np

from mesh import Assignment_Matrix
from element_routine_plastic import plastic_element_routine


def assemble_plastic(u_hat, r_nodes, connectivity, epsilon_pl_old, data):

    n_nodes = len(r_nodes)
    n_elements = len(connectivity)
    n_gp = epsilon_pl_old.shape[1]
    F_int = np.zeros(n_nodes)
    K_t = np.zeros((n_nodes, n_nodes))
    epsilon_pl_new = np.zeros_like(epsilon_pl_old)
    stress = np.zeros((n_elements, n_gp, 2))
    r_gp = np.zeros((n_elements, n_gp))
    plastic_flag = np.zeros((n_elements, n_gp), dtype=bool)

    for e in range(n_elements):

        element_nodes = connectivity[e]
        A_e = Assignment_Matrix(e, n_nodes)
        u_e = A_e @ u_hat
        r_e = r_nodes[element_nodes]
        epsilon_pl_old_e = epsilon_pl_old[e]

        F_int_e, K_t_e, epsilon_pl_new_e, stress_e, r_gp_e, plastic_flag_e = plastic_element_routine(u_e, r_e, epsilon_pl_old_e, data)
        F_int += A_e.T @ F_int_e
        K_t += A_e.T @ K_t_e @ A_e

        epsilon_pl_new[e] = epsilon_pl_new_e
        stress[e] = stress_e
        r_gp[e] = r_gp_e
        plastic_flag[e] = plastic_flag_e

    return F_int, K_t, epsilon_pl_new, stress, r_gp, plastic_flag