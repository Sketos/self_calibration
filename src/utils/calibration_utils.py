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
    """Short summary.

    Parameters
    ----------
    antennas : type
        Description of parameter `antennas`.

    Returns
    -------
    type
        Description of returned object.

    """

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
    """Short summary.

    Parameters
    ----------
    f : type
        Description of parameter `f`.
    C : type
        Description of parameter `C`.

    Returns
    -------
    type
        Description of returned object.

    """

    A = np.matmul(
        f,
        np.matmul(C, f.T)
    )

    return A


def compute_B_matrix_from_f_and_C_matrices(f, C):
    """Short summary.

    Parameters
    ----------
    f : type
        Description of parameter `f`.
    C : type
        Description of parameter `C`.

    Returns
    -------
    type
        Description of returned object.

    """

    B = np.matmul(f, C)

    return B


def phase_errors_from_A_and_B_matrices(phases, model_phases, A, B):
    """Short summary.

    Parameters
    ----------
    phases : type
        Description of parameter `phases`.
    model_phases : type
        Description of parameter `model_phases`.
    A : type
        Description of parameter `A`.
    B : type
        Description of parameter `B`.

    Returns
    -------
    type
        Description of returned object.

    """

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


def phase_errors(f):
    pass

    # phase_difference = wrap(
    #     a=np.subtract(
    #         phases,
    #         model_phases
    #     )
    # )
    #
    # phase_errors = np.linalg.solve(
    #     A,
    #     np.matmul(
    #         B,
    #         phase_difference
    #     )
    # )

    #return phase_errors
