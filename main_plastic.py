import os
import numpy as np
import matplotlib.pyplot as plt

from input_data import input_data
from run_plastic import solve_plastic_problem
from run_plastic import run_plastic_mesh_convergence
from run_plastic import run_plastic_time_convergence


data = input_data()

plot_folder = "plastic_plots"
os.makedirs(plot_folder, exist_ok=True)

run_convergence_studies = True

plastic_result = solve_plastic_problem(data, data["n_elements"], data["n_increments"])

print("Plastic simulation completed.")
print("p_max =", plastic_result["p_max"], "MPa")
print("n_elements =", plastic_result["n_elements"])
print("n_increments =", plastic_result["n_increments"])

print("Final outer displacement u_r(b) =")
print(plastic_result["u_b_history"][-1], "mm")

print("Final number of plastic Gauss points =")
print(plastic_result["plastic_point_history"][-1])

print("Final displacement vector u_hat =")
print(plastic_result["u_hat"])

print("Final Gauss point radii r_gp =")
print(plastic_result["r_gp"])

print("Final stress values =")
print(plastic_result["stress"])

print("Newton iteration history =")
print(plastic_result["iteration_history"])

print("Plastic point history =")
print(plastic_result["plastic_point_history"])


r_nodes = plastic_result["r_nodes"]
u_hat = plastic_result["u_hat"]

r_gp = plastic_result["r_gp"].reshape(-1)
stress = plastic_result["stress"].reshape(-1, 2)

sigma_rr = stress[:, 0]
sigma_phiphi = stress[:, 1]
sigma_eq = np.abs(sigma_rr - sigma_phiphi)

p_history = plastic_result["p_history"]
u_b_history = plastic_result["u_b_history"]
plastic_point_history = plastic_result["plastic_point_history"]


plt.figure()
plt.plot(r_nodes, u_hat, "o-")
plt.xlabel("r [mm]")
plt.ylabel("u_r [mm]")
plt.title("Final radial displacement")
plt.grid(True)
plt.savefig(plot_folder + "/plastic_final_radial_displacement.png", dpi=300, bbox_inches="tight")
plt.show()


plt.figure()
plt.plot(r_gp, sigma_rr, "o-", label="sigma_rr")
plt.plot(r_gp, sigma_phiphi, "o-", label="sigma_phiphi")
plt.xlabel("r [mm]")
plt.ylabel("stress [MPa]")
plt.title("Final stress distribution")
plt.grid(True)
plt.legend()
plt.savefig(plot_folder + "/plastic_final_stress_distribution.png", dpi=300, bbox_inches="tight")
plt.show()


plt.figure()
plt.plot(r_gp, sigma_eq, "o-", label="|sigma_rr - sigma_phiphi|")
plt.axhline(data["sigma_y"], linestyle="--", label="sigma_y")
plt.xlabel("r [mm]")
plt.ylabel("Tresca stress [MPa]")
plt.title("Final Tresca stress distribution")
plt.grid(True)
plt.legend()
plt.savefig(plot_folder + "/plastic_final_tresca_stress_distribution.png", dpi=300, bbox_inches="tight")
plt.show()


plt.figure()
plt.plot(p_history, u_b_history, "o-")
plt.xlabel("p [MPa]")
plt.ylabel("u_r(b) [mm]")
plt.title("Pipe widening history")
plt.grid(True)
plt.savefig(plot_folder + "/plastic_pipe_widening_history.png", dpi=300, bbox_inches="tight")
plt.show()


plt.figure()
plt.plot(p_history, plastic_point_history, "o-")
plt.xlabel("p [MPa]")
plt.ylabel("number of plastic Gauss points")
plt.title("Plastic zone development")
plt.grid(True)
plt.savefig(plot_folder + "/plastic_zone_development.png", dpi=300, bbox_inches="tight")
plt.show()


if run_convergence_studies == True:

    plastic_mesh_convergence_result = run_plastic_mesh_convergence(data)

    print("Plastic mesh convergence completed.")

    print("n_elements_list =")
    print(plastic_mesh_convergence_result["n_elements_list"])

    print("final_u_b_list =")
    print(plastic_mesh_convergence_result["final_u_b_list"])

    print("final_plastic_point_list =")
    print(plastic_mesh_convergence_result["final_plastic_point_list"])

    print("average_iteration_list =")
    print(plastic_mesh_convergence_result["average_iteration_list"])

    print("maximum_iteration_list =")
    print(plastic_mesh_convergence_result["maximum_iteration_list"])


    plt.figure()
    plt.plot(
        plastic_mesh_convergence_result["n_elements_list"],
        plastic_mesh_convergence_result["final_u_b_list"],
        "o-"
    )
    plt.xlabel("number of elements")
    plt.ylabel("final u_r(b) [mm]")
    plt.title("Plastic mesh convergence")
    plt.ticklabel_format(axis="y", style="plain", useOffset=False)
    plt.grid(True)
    plt.savefig(plot_folder + "/plastic_mesh_convergence.png", dpi=300, bbox_inches="tight")
    plt.show()


    plastic_time_convergence_result = run_plastic_time_convergence(data)

    print("Plastic load-increment convergence completed.")

    print("n_increments_list =")
    print(plastic_time_convergence_result["n_increments_list"])

    print("final_u_b_list =")
    print(plastic_time_convergence_result["final_u_b_list"])

    print("final_plastic_point_list =")
    print(plastic_time_convergence_result["final_plastic_point_list"])

    print("average_iteration_list =")
    print(plastic_time_convergence_result["average_iteration_list"])

    print("maximum_iteration_list =")
    print(plastic_time_convergence_result["maximum_iteration_list"])


    plt.figure()
    plt.plot(plastic_time_convergence_result["n_increments_list"], plastic_time_convergence_result["final_u_b_list"], "o-")
    plt.xlabel("number of load increments")
    plt.ylabel("final u_r(b) [mm]")
    plt.title("Plastic load-increment convergence")
    plt.ticklabel_format(axis="y", style="plain", useOffset=False)
    plt.grid(True)
    plt.savefig(plot_folder + "/plastic_load_increment_convergence.png", dpi=300, bbox_inches="tight")
    plt.show()