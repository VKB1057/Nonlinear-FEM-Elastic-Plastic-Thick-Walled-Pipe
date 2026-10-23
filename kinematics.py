import numpy as np

def gauss_rule(n_gp):

    if n_gp == 1:

        xi = 0.0
        w = 2.0

    else:

        raise ValueError("Only 1 Gauss point is to be used for variant 4.")

    return xi, w


def B_matrix(xi, r_e):

    N1 = 0.5 * (1.0 - xi)
    N2 = 0.5 * (1.0 + xi)
    N = np.array([N1, N2])

    dN1_dxi = -0.5
    dN2_dxi = 0.5
    dN_dxi = np.array([dN1_dxi, dN2_dxi])

    r = N @ r_e
    J = dN_dxi @ r_e
    dN_dr = dN_dxi / J

    B = np.array([
        [dN_dr[0], dN_dr[1]],
        [N[0] / r, N[1] / r]
    ])

    return B, J, r, N


def strain_function(B, u_e):

    epsilon = B @ u_e

    return epsilon