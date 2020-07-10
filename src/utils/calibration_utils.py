import numpy as np


def wrap(a):
    """Short summary.

    Parameters
    ----------
    a : type
        Description of parameter `a`.

    Returns
    -------
    type
        Description of returned object.

    """

    return (a + np.pi) % (2.0 * np.pi) - np.pi


def compute_f_matrix_from_antennas(antennas):

    f = np.zeros(
        shape=(
            len(np.unique(antennas)),
            antennas.shape[0]
        )
    )

    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            if antennas[j, 0] == i:
                f[i,j] = +1.0
            if antennas[j, 1] == i:
                f[i,j] = -1.0

    return f


def compute_A_matrix_from_f_and_C_matrices(f, C):

    A = np.matmul(
        f,
        np.matmul(C, f.T)
    )

    return A


def compute_B_matrix_from_f_and_C_matrices(f, C):

    B = np.matmul(f, C)

    return B


def phase_errors_from_A_and_B_matrices(phases, model_phases, A, B):

    phase_difference = wrap(
        a=np.subtract(
            phases,
            model_phases
        )
    )

    phase_errors = np.linalg.solve(
        A,
        np.matmul(
            B,
            phase_difference
        )
    )

    return phase_errors
