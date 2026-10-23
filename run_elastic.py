import numpy as np

from mesh import create_radial_mesh
from mesh import create_connectivity
from solver_routine_elastic import Newton_Raphson_solver_elastic

def solve_elastic_problem(data, n_elements):

    r_nodes = create_radial_mesh(data["a"], data["b"], n_elements, data["mesh_refinement_factor"])
    connectivity = create_connectivity(n_elements)
    n_nodes = len(r_nodes)
    u_start = np.zeros(n_nodes)

    solution_increment = Newton_Raphson_solver_elastic(u_start, r_nodes, connectivity, data)
    u_fem = solution_increment["u_hat"]

    u_exact = (data["p_elastic"] / data["E"]) * (data["a"]**2 / (data["b"]**2 - data["a"]**2)) * ((1.0 - data["nu"]) * r_nodes + (1.0 + data["nu"]) * data["b"]**2 / r_nodes)

    error_vector = u_fem - u_exact
    error_inf = np.linalg.norm(error_vector, np.inf)
    exact_inf = np.linalg.norm(u_exact, np.inf)
    relative_error_inf = error_inf / exact_inf

    elastic_result = {
        "r_nodes": r_nodes,
        "connectivity": connectivity,
        "u_fem": u_fem,
        "u_exact": u_exact,
        "error_vector": error_vector,
        "error_inf": error_inf,
        "relative_error_inf": relative_error_inf,
        "p_elastic": data["p_elastic"],
        "n_elements": n_elements,
        "n_iterations": solution_increment["n_iterations"],
        "solution_increment": solution_increment
    }

    return elastic_result


def run_elastic_mesh_convergence(data):

    n_elements_list = np.array([2, 5, 10, 20, 30, 40, 60, 100])
    relative_error_list = []
    outer_displacement_fem_list = []
    outer_displacement_exact_list = []
    iteration_list = []
    result_list = []

    for n_elements in n_elements_list:

        elastic_result = solve_elastic_problem(data, n_elements)
        relative_error_list.append(elastic_result["relative_error_inf"])
        outer_displacement_fem_list.append(elastic_result["u_fem"][-1])
        outer_displacement_exact_list.append(elastic_result["u_exact"][-1])
        iteration_list.append(elastic_result["n_iterations"])
        result_list.append(elastic_result)

        print("Elastic mesh convergence")
        print("n_elements =", n_elements)
        print("relative_error_inf =", elastic_result["relative_error_inf"])
        print("u_fem at r=b =", elastic_result["u_fem"][-1])
        print("u_exact at r=b =", elastic_result["u_exact"][-1])
        print("Newton iterations =", elastic_result["n_iterations"])

    convergence_result = {
        "n_elements_list": n_elements_list,
        "relative_error_list": np.array(relative_error_list),
        "outer_displacement_fem_list": np.array(outer_displacement_fem_list),
        "outer_displacement_exact_list": np.array(outer_displacement_exact_list),
        "iteration_list": np.array(iteration_list),
        "p_elastic": data["p_elastic"],
        "p_init": data["p_init"],
        "result_list": result_list
    }

    return convergence_result