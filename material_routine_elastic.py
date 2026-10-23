import numpy as np

def material_stiffness_matrix(E, nu, plane_condition):

    if plane_condition == "PS":

        C = E / (1.0 - nu**2) * np.array([
            [1.0, nu],
            [nu, 1.0]
        ])

    else:

        raise ValueError("Only plane stress condition PS is implemented for variant 4.")

    return C


def elastic_material_routine(epsilon, data):

    E = data["E"]
    nu = data["nu"]
    plane_condition = data["plane_condition"]
    C_t = material_stiffness_matrix(E, nu, plane_condition)
    sigma = C_t @ epsilon

    return sigma, C_t