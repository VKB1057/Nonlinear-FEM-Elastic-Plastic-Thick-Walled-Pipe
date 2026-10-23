import numpy as np

def create_radial_mesh(a, b, n_elements, mesh_refinement_factor):

    if n_elements == 1:

        r_nodes = [a, b]
        r_nodes = np.array(r_nodes)
        return r_nodes

    q = mesh_refinement_factor ** (1.0 / (n_elements - 1))
    dr = (b - a) * (1.0 - q) / (1.0 - mesh_refinement_factor * q)
    r_node = a
    r_nodes = [a]

    for i in range(1, n_elements + 1):

        r_node = r_node + dr
        r_nodes.append(r_node)
        dr = dr * q

    r_nodes = np.array(r_nodes)
    return r_nodes


def create_connectivity(n_elements):

    connectivity = []

    for e in range(n_elements):

        element_nodes = [e, e + 1]
        connectivity.append(element_nodes)

    connectivity = np.array(connectivity, dtype=int)
    return connectivity


def Assignment_Matrix(e, n_nodes):

    A_e = np.zeros((2, n_nodes))
    A_e[0, e] = 1.0
    A_e[1, e + 1] = 1.0
    return A_e 