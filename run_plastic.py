import numpy as np

from mesh import create_radial_mesh
from mesh import create_connectivity
from solver_routine_plastic import Newton_Raphson_solver_plastic


def solve_plastic_problem(data, n_elements, n_increments):

    r_nodes = create_radial_mesh(data["a"], data["b"], n_elements, data["mesh_refinement_factor"])
    connectivity = create_connectivity(n_elements)
    n_nodes = len(r_nodes)
    n_gp = data["n_gp"]
    u_hat = np.zeros(n_nodes)
    epsilon_pl_old = np.zeros((n_elements, n_gp, 2))
    p_history = [0.0]
    u_b_history = [u_hat[-1]]
    iteration_history = []
    plastic_point_history = [0]
    u_history = [u_hat.copy()]
    solution_increment_history = []

    for increment in range(1, n_increments + 1):

        p_current = increment / n_increments * data["p_max"]

        print("Load increment:", increment)
        print("p_current =", p_current)

        solution_increment = Newton_Raphson_solver_plastic(u_hat, r_nodes, connectivity, epsilon_pl_old, data, p_current)
        u_hat = solution_increment["u_hat"].copy()
        epsilon_pl_old = solution_increment["epsilon_pl_new"].copy()
        number_of_plastic_points = np.sum(solution_increment["plastic_flag"])
        p_history.append(p_current)
        u_b_history.append(u_hat[-1])
        iteration_history.append(solution_increment["n_iterations"])
        plastic_point_history.append(number_of_plastic_points)
        u_history.append(u_hat.copy())
        solution_increment_history.append(solution_increment)

    final_increment = solution_increment_history[-1]

    plastic_result = {
        "r_nodes": r_nodes,
        "connectivity": connectivity,
        "u_hat": u_hat,
        "epsilon_pl": epsilon_pl_old,
        "stress": final_increment["stress"],
        "r_gp": final_increment["r_gp"],
        "plastic_flag": final_increment["plastic_flag"],
        "p_history": np.array(p_history),
        "u_b_history": np.array(u_b_history),
        "iteration_history": np.array(iteration_history),
        "plastic_point_history": np.array(plastic_point_history),
        "u_history": np.array(u_history),
        "solution_increment_history": solution_increment_history,
        "final_increment": final_increment,
        "n_elements": n_elements,
        "n_increments": n_increments,
        "p_max": data["p_max"]
    }

    return plastic_result


def run_plastic_mesh_convergence(data):

    n_elements_list = np.array([2, 5, 10, 20, 30, 40, 60, 100])
    n_increments = data["n_increments"]
    final_u_b_list = []
    final_plastic_point_list = []
    average_iteration_list = []
    maximum_iteration_list = []
    result_list = []

    for n_elements in n_elements_list:

        print("Plastic mesh convergence")
        print("n_elements =", n_elements)
        print("n_increments =", n_increments)

        plastic_result = solve_plastic_problem(data, n_elements, n_increments)
        final_u_b = plastic_result["u_b_history"][-1]
        final_plastic_points = plastic_result["plastic_point_history"][-1]
        average_iterations = np.mean(plastic_result["iteration_history"])
        maximum_iterations = np.max(plastic_result["iteration_history"])
        final_u_b_list.append(final_u_b)
        final_plastic_point_list.append(final_plastic_points)
        average_iteration_list.append(average_iterations)
        maximum_iteration_list.append(maximum_iterations)
        result_list.append(plastic_result)

        print("Final u_r(b) =", final_u_b)
        print("Final plastic points =", final_plastic_points)
        print("Average Newton iterations =", average_iterations)
        print("Maximum Newton iterations =", maximum_iterations)

    mesh_convergence_result = {
        "n_elements_list": n_elements_list,
        "n_increments": n_increments,
        "final_u_b_list": np.array(final_u_b_list),
        "final_plastic_point_list": np.array(final_plastic_point_list),
        "average_iteration_list": np.array(average_iteration_list),
        "maximum_iteration_list": np.array(maximum_iteration_list),
        "result_list": result_list
    }

    return mesh_convergence_result


def run_plastic_time_convergence(data):

    n_increments_list = np.array([10, 20, 40, 80])
    n_elements = data["n_elements"]
    final_u_b_list = []
    final_plastic_point_list = []
    average_iteration_list = []
    maximum_iteration_list = []
    result_list = []

    for n_increments in n_increments_list:

        print("Plastic time convergence")
        print("n_elements =", n_elements)
        print("n_increments =", n_increments)

        plastic_result = solve_plastic_problem(data, n_elements, n_increments)
        final_u_b = plastic_result["u_b_history"][-1]
        final_plastic_points = plastic_result["plastic_point_history"][-1]
        average_iterations = np.mean(plastic_result["iteration_history"])
        maximum_iterations = np.max(plastic_result["iteration_history"])
        final_u_b_list.append(final_u_b)
        final_plastic_point_list.append(final_plastic_points)
        average_iteration_list.append(average_iterations)
        maximum_iteration_list.append(maximum_iterations)
        result_list.append(plastic_result)

        print("Final u_r(b) =", final_u_b)
        print("Final plastic points =", final_plastic_points)
        print("Average Newton iterations =", average_iterations)
        print("Maximum Newton iterations =", maximum_iterations)

    time_convergence_result = {
        "n_elements": n_elements,
        "n_increments_list": n_increments_list,
        "final_u_b_list": np.array(final_u_b_list),
        "final_plastic_point_list": np.array(final_plastic_point_list),
        "average_iteration_list": np.array(average_iteration_list),
        "maximum_iteration_list": np.array(maximum_iteration_list),
        "result_list": result_list
    }

    return time_convergence_result