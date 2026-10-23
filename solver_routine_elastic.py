import numpy as np

from assembly_routine_elastic import assemble_elastic
from assembly_routine_elastic import F_external


def Newton_Raphson_solver_elastic(u_start, r_nodes, connectivity, data):

    u_hat = u_start.copy()
    n_nodes = len(r_nodes)
    a = r_nodes[0]
    F_ext = F_external(n_nodes, a, data["p_elastic"])
    converged = False

    for k in range(data["k_max"]):

        F_int, K_t = assemble_elastic(u_hat, r_nodes, connectivity, data)
        R = F_int - F_ext
        delta_u = np.linalg.solve(K_t, -R)
        u_hat = u_hat + delta_u

        F_int_check, K_t_check = assemble_elastic(u_hat, r_nodes, connectivity, data)
        R_check = F_int_check - F_ext

        norm_R = np.linalg.norm(R_check, np.inf)
        norm_F_int = np.linalg.norm(F_int_check, np.inf)
        norm_delta_u = np.linalg.norm(delta_u, np.inf)
        norm_u = np.linalg.norm(u_hat, np.inf)
        residual_converged = norm_R < data["eps_R"] * norm_F_int
        displacement_converged = norm_delta_u < data["eps_u"] * norm_u

        print("Newton iteration:", k + 1)
        print("||R||_inf =", norm_R)
        print("||F_int||_inf =", norm_F_int)
        print("||delta_u||_inf =", norm_delta_u)
        print("||u||_inf =", norm_u)

        if residual_converged and displacement_converged:

            converged = True
            break

    if converged == False:

        raise RuntimeError("Newton-Raphson did not converge.")

    solution_increment = {
        "u_hat": u_hat,
        "F_int": F_int_check,
        "F_ext": F_ext,
        "R": R_check,
        "K_t": K_t_check,
        "n_iterations": k + 1,
        "p_current": data["p_elastic"]
    }

    return solution_increment