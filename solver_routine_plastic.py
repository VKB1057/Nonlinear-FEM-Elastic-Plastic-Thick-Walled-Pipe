import numpy as np

from assembly_routine_elastic import F_external
from assembly_routine_plastic import assemble_plastic


def Newton_Raphson_solver_plastic(u_start, r_nodes, connectivity, epsilon_pl_old, data, p_current):

    u_hat = u_start.copy()
    n_nodes = len(r_nodes)
    a = r_nodes[0]

    F_ext = F_external(n_nodes, a, p_current)
    converged = False

    for k in range(data["k_max"]):

        F_int, K_t, epsilon_pl_new, stress, r_gp, plastic_flag = assemble_plastic(u_hat, r_nodes, connectivity, epsilon_pl_old, data)

        R = F_int - F_ext
        delta_u = np.linalg.solve(K_t, -R)
        u_hat = u_hat + delta_u

        F_int_check, K_t_check, epsilon_pl_new, stress, r_gp, plastic_flag = assemble_plastic(u_hat, r_nodes, connectivity, epsilon_pl_old, data)
        R_check = F_int_check - F_ext

        norm_R = np.linalg.norm(R_check, np.inf)
        norm_F_int = np.linalg.norm(F_int_check, np.inf)
        norm_delta_u = np.linalg.norm(delta_u, np.inf)
        norm_u = np.linalg.norm(u_hat, np.inf)
        
        residual_converged = norm_R < data["eps_R"] * norm_F_int
        displacement_converged = norm_delta_u < data["eps_u"] * norm_u

        print("Newton iteration:", k + 1)
        print("p_current =", p_current)
        print("||R||_inf =", norm_R)
        print("||F_int||_inf =", norm_F_int)
        print("||delta_u||_inf =", norm_delta_u)
        print("||u||_inf =", norm_u)

        if residual_converged and displacement_converged:

            converged = True
            break

    if converged == False:

        raise RuntimeError("Plastic Newton-Raphson did not converge.")

    solution_increment = {
        "u_hat": u_hat,
        "F_int": F_int_check,
        "F_ext": F_ext,
        "R": R_check,
        "K_t": K_t_check,
        "epsilon_pl_new": epsilon_pl_new,
        "stress": stress,
        "r_gp": r_gp,
        "plastic_flag": plastic_flag,
        "n_iterations": k + 1,
        "p_current": p_current
    }

    return solution_increment