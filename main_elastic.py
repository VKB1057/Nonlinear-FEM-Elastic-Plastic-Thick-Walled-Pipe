import os
import matplotlib.pyplot as plt

from input_data import input_data
from run_elastic import solve_elastic_problem
from run_elastic import run_elastic_mesh_convergence


data = input_data()

elastic_result = solve_elastic_problem(
    data,
    data["n_elements"]
)

print("Elastic verification completed.")
print("p_init =", data["p_init"], "MPa")
print("p_elastic =", data["p_elastic"], "MPa")
print("n_elements =", elastic_result["n_elements"])
print("Newton iterations =", elastic_result["n_iterations"])

print("Inner displacement FEM u_r(a) =")
print(elastic_result["u_fem"][0], "mm")

print("Outer displacement FEM u_r(b) =")
print(elastic_result["u_fem"][-1], "mm")

print("Inner displacement exact u_r(a) =")
print(elastic_result["u_exact"][0], "mm")

print("Outer displacement exact u_r(b) =")
print(elastic_result["u_exact"][-1], "mm")

print("Relative infinity norm error =")
print(elastic_result["relative_error_inf"])

plot_folder = "elastic_plots"
os.makedirs(plot_folder, exist_ok=True)

plt.figure()
plt.plot(elastic_result["r_nodes"], elastic_result["u_fem"], "o-", label="FEM")
plt.plot(elastic_result["r_nodes"], elastic_result["u_exact"], "-", label="Analytical")
plt.xlabel("r [mm]")
plt.ylabel("u_r [mm]")
plt.title("Elastic verification: FEM vs analytical solution")
plt.grid(True)
plt.legend()
plt.savefig(plot_folder + "/elastic_verification_FEM_vs_analytical.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure()
plt.plot(elastic_result["r_nodes"], elastic_result["error_vector"], "o-")
plt.xlabel("r [mm]")
plt.ylabel("u_FEM - u_exact [mm]")
plt.title("Elastic displacement error")
plt.grid(True)
plt.savefig(plot_folder + "/elastic_displacement_error.png", dpi=300, bbox_inches="tight")
plt.show()

convergence_result = run_elastic_mesh_convergence(data)

print("Elastic mesh convergence completed.")

print("n_elements_list =")
print(convergence_result["n_elements_list"])

print("relative_error_list =")
print(convergence_result["relative_error_list"])

print("outer_displacement_fem_list =")
print(convergence_result["outer_displacement_fem_list"])

print("outer_displacement_exact_list =")
print(convergence_result["outer_displacement_exact_list"])

print("iteration_list =")
print(convergence_result["iteration_list"])


plt.figure()
plt.plot(convergence_result["n_elements_list"], convergence_result["relative_error_list"], "o-")
plt.xlabel("number of elements")
plt.ylabel("relative infinity norm error")
plt.title("Elastic mesh convergence")
plt.grid(True)
plt.savefig(plot_folder + "/elastic_mesh_convergence.png", dpi=300, bbox_inches="tight")
plt.show()


plt.figure()
plt.plot(convergence_result["n_elements_list"], convergence_result["outer_displacement_fem_list"], "o-", label="FEM")
plt.plot(convergence_result["n_elements_list"], convergence_result["outer_displacement_exact_list"], "-", label="Analytical")
plt.xlabel("number of elements")
plt.ylabel("u_r(b) [mm]")
plt.title("Outer displacement convergence")
plt.ticklabel_format(axis="y", style="plain", useOffset=False)
plt.grid(True)
plt.legend()
plt.savefig(plot_folder + "/elastic_outer_displacement_convergence.png", dpi=300, bbox_inches="tight")
plt.show()