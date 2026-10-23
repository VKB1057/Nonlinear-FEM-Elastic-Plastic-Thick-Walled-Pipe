import numpy as np

from mesh import Assignment_Matrix
from element_routine_elastic import elastic_element_routine


def assemble_elastic(u_hat, r_nodes, connectivity, data):

    n_nodes = len(r_nodes)
    n_elements = len(connectivity)

    F_int = np.zeros(n_nodes)
    K_t = np.zeros((n_nodes, n_nodes))

    for e in range(n_elements):

        element_nodes = connectivity[e]
        A_e = Assignment_Matrix(e, n_nodes)
        u_e = A_e @ u_hat
        r_e = r_nodes[element_nodes]
        F_int_e, K_t_e = elastic_element_routine(u_e, r_e, data)
        F_int += A_e.T @ F_int_e
        K_t += A_e.T @ K_t_e @ A_e

    return F_int, K_t


def F_external(n_nodes, a, p):

    F_ext = np.zeros(n_nodes)
    F_ext[0] = a * p

    return F_ext 