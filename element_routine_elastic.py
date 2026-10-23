from kinematics import gauss_rule
from kinematics import B_matrix
from kinematics import strain_function
from material_routine_elastic import elastic_material_routine


def elastic_element_routine(u_e, r_e, data):

    xi, w = gauss_rule(data["n_gp"])
    B, J, r, N = B_matrix(xi, r_e)
    epsilon = strain_function(B, u_e)
    sigma, C_t = elastic_material_routine(epsilon, data)
    F_int_e = B.T @ sigma * r * J * w
    K_t_e = B.T @ C_t @ B * r * J * w

    return F_int_e, K_t_e 