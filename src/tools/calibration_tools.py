import numpy as np

def compute_C_matrix(visibilities, sigma):

    return np.diag(
        np.power(
            np.divide(
                np.sqrt(
                    np.add(
                        np.multiply(
                            np.square(visibilities.real),
                            np.square(sigma[:, 0])
                        ),
                        np.multiply(
                            np.square(visibilities.imag),
                            np.square(sigma[:, 1])
                        )
                    )
                ),
                visibilities.amplitudes**2.0,
            ),
            -2
        )
    )
