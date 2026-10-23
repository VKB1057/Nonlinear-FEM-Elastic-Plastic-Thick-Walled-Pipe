def input_data():

    data = {}

    data["E"] = 200000.0
    data["nu"] = 0.30
    data["sigma_y"] = 200.0

    data["a"] = 30.0
    data["b"] = 60.0

    data["p_max"] = 120.0

    data["plane_condition"] = "PS"

    data["n_gp"] = 1

    data["n_elements"] = 20
    data["n_increments"] = 50
    data["k_max"] = 20

    data["eps_R"] = 0.005
    data["eps_u"] = 0.005

    data["mesh_refinement_factor"] = 2.0

    data["p_init"] = data["sigma_y"] * (data["b"]**2 - data["a"]**2) / (2.0 * data["b"]**2)
    data["p_elastic"] = 0.5 * data["p_init"]

    return data