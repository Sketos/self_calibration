import os
import sys
import matplotlib

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from scipy import optimize

from src.utils import random_utils
from src.utils import calibration_utils

from src.tools import calibration_tools

from src.datatypes.visibilities import Visibilities


def generate_phase_errors(antennas):

    antennas_unique = np.unique(antennas)

    if os.path.isfile("./phase_errors.fits"):
        phase_errors = fits.getdata(
            filename="./phase_errors.fits"
        )
    else:
        np.random.seed(seed=random_utils.seed_generator())
        phase_errors = np.random.uniform(
            -np.pi / 10.0,
            +np.pi / 10.0,
            size=antennas_unique.size
        )
        fits.writeto(
            "./phase_errors.fits",
            data=phase_errors
        )

    return phase_errors


def corrupt_visibilities(visibilities, f, phase_errors):

    phases_corrupted = np.add(
        visibilities.phases,
        np.matmul(
            f.T, phase_errors
        )
    )

    return Visibilities.manual(
        array=np.stack(
            arrays=(
                np.multiply(
                    visibilities.amplitudes,
                    np.cos(phases_corrupted)
                ),
                np.multiply(
                    visibilities.amplitudes,
                    np.sin(phases_corrupted)
                )
            ),
            axis=-1
        )
    )


def residuals(phases, model_phases, f, phase_errors):

    return np.square(
        np.subtract(
            calibration_utils.wrap(
                np.subtract(
                    phases, model_phases
                )
            ),
            np.matmul(
                f.T, phase_errors
            )
        )
    )



# def compute_phase_errors(visibilities, model_visibilities, sigma, type="approx"):
#
#     if type == "approx":
#         C = calibration_tools.compute_C_matrix(
#             visibilities=visibilities,
#             sigma=sigma
#         )
#
#         A = calibration_utils.compute_A_matrix_from_f_and_C_matrices(
#             f=f, C=C
#         )
#
#         B = calibration_utils.compute_B_matrix_from_f_and_C_matrices(
#             f=f, C=C
#         )
#
#         phase_errors = calibration_utils.phase_errors_from_A_and_B_matrices(
#             phases=visibilities.phases,
#             model_phases=model_visibilities.phases,
#             A=A,
#             B=B
#         )


antennas = fits.getdata(
    filename="./antennas.fits"
)

phase_errors = generate_phase_errors(
    antennas=antennas
)

if __name__ == "__main__":

    visibilities = Visibilities.from_fits(
        filename="./visibilities.fits"
    )

    # NOTE: ...
    f = calibration_utils.compute_f_matrix_from_antennas(
        antennas=antennas
    )
    # plt.figure()
    # plt.imshow(
    #     f,
    #     cmap="jet",
    #     aspect="auto"
    # )
    # plt.show()
    # exit()

    visibilities_corrupted = corrupt_visibilities(
        visibilities=visibilities,
        f=f,
        phase_errors=phase_errors
    )

    # out = residuals(
    #     phases=visibilities_corrupted.phases,
    #     model_phases=visibilities.phases,
    #     f=f,
    #     phase_errors=phase_errors
    # )
    # print(out)
    # exit()

    # NOTE: ...
    sigma = np.random.normal(
        loc=0.0,
        scale=1.0 * 10**-1.0,
        size=visibilities_corrupted.shape
    )

    visibilities_corrupted_with_added_noise = Visibilities.manual(
        array=np.add(
            visibilities_corrupted,
            sigma
        )
    )
    # # NOTE: ...
    # plt.figure()
    # plt.plot(
    #     visibilities_corrupted.real,
    #     visibilities_corrupted.imag,
    #     linestyle="None",
    #     marker="o",
    #     markersize=10,
    #     color="black"
    # )
    # plt.plot(
    #     visibilities_corrupted_with_added_noise.real,
    #     visibilities_corrupted_with_added_noise.imag,
    #     linestyle="None",
    #     marker="o",
    #     markersize=5,
    #     color="r"
    # )
    # plt.show()
    # exit()



    def cost_function(phase_errors, f, visibilities, model_visibilities, sigma):

        def chi_square(visibilities, model_visibilities, sigma):

            return np.sum(
                np.square(
                    np.divide(
                        np.subtract(
                            visibilities,
                            model_visibilities
                        ),
                        sigma
                    )
                )
            )

        model_visibilities_corrupted = corrupt_visibilities(
            visibilities=model_visibilities,
            f=f,
            phase_errors=phase_errors
        )

        chi_square_real = chi_square(
            visibilities=visibilities[:, 0],
            model_visibilities=model_visibilities_corrupted[:, 0],
            sigma=sigma[:, 0]
        )

        chi_square_imag = chi_square(
            visibilities=visibilities[:, 1],
            model_visibilities=model_visibilities_corrupted[:, 1],
            sigma=sigma[:, 1]
        )

        cost = np.sum(
            a=[chi_square_real, chi_square_imag]
        )

        return cost

    # NOTE: ...
    res = optimize.minimize(
        cost_function,
        x0=phase_errors,
        method='L-BFGS-B',
        options={
            'maxiter':50000,
            'ftol':1e-04
        },
        bounds=tuple([-np.pi, np.pi]
            for i in range(phase_errors.size)
        ),
        args=(
            f,
            visibilities_corrupted_with_added_noise,
            visibilities,
            sigma
        )
    )
    phase_errors_sol = res.x

    # cost_true = cost_function(
    #     phase_errors=phase_errors,
    #     f=f,
    #     visibilities=visibilities_corrupted_with_added_noise,
    #     model_visibilities=visibilities,
    #     sigma=sigma
    # )
    #
    # cost_sol = cost_function(
    #     phase_errors=phase_errors_sol,
    #     f=f,
    #     visibilities=visibilities_corrupted_with_added_noise,
    #     model_visibilities=visibilities,
    #     sigma=sigma
    # )
    # print(cost_true, cost_sol)
    #
    # tol = np.divide(
    #     np.subtract(
    #         cost_true,
    #         cost_sol
    #     ),
    #     np.max([
    #         cost_true,
    #         cost_sol
    #     ])
    # )
    # print(
    #     "tol = {}".format(tol)
    # )

    # # NOTE: ...
    # residuals_true = residuals(
    #     phases=visibilities_corrupted_with_added_noise.phases,
    #     model_phases=visibilities.phases,
    #     f=f,
    #     phase_errors=phase_errors
    # )
    # residuals_sol = residuals(
    #     phases=visibilities_corrupted_with_added_noise.phases,
    #     model_phases=visibilities.phases,
    #     f=f,
    #     phase_errors=phase_errors_sol
    # )
    # print(
    #     np.sum(residuals_true),
    #     np.sum(residuals_sol)
    # )
    # exit()


    C = calibration_tools.compute_C_matrix(
        visibilities=visibilities_corrupted_with_added_noise,
        sigma=sigma
    )

    A = calibration_utils.compute_A_matrix_from_f_and_C_matrices(
        f=f, C=C
    )

    B = calibration_utils.compute_B_matrix_from_f_and_C_matrices(
        f=f, C=C
    )

    phase_errors_sol_approx = calibration_utils.phase_errors_from_A_and_B_matrices(
        phases=visibilities_corrupted_with_added_noise.phases,
        model_phases=visibilities.phases,
        A=A,
        B=B
    )

    # # NOTE: ...
    # residuals_true = residuals(
    #     phases=visibilities_corrupted_with_added_noise.phases,
    #     model_phases=visibilities.phases,
    #     f=f,
    #     phase_errors=phase_errors
    # )
    # residuals_sol = residuals(
    #     phases=visibilities_corrupted_with_added_noise.phases,
    #     model_phases=visibilities.phases,
    #     f=f,
    #     phase_errors=phase_errors_sol
    # )
    # print(
    #     np.sum(residuals_true),
    #     np.sum(residuals_sol)
    # )
    # exit()

    # NOTE: visualise
    plt.figure()
    index = 0
    plt.plot(
        phase_errors,
        color="black",
        linewidth=4,
        alpha=0.75,
        label="input"
    )
    plt.plot(
        np.add(
            phase_errors_sol,
            np.subtract(
                phase_errors[index],
                phase_errors_sol[index]
            )
        ),
        color="r",
        linewidth=1,
        marker="o",
        alpha=0.75,
        label="self-calibration"
    )
    plt.plot(
        np.add(
            phase_errors_sol_approx,
            np.subtract(
                phase_errors[index],
                phase_errors_sol_approx[index]
            )
        ),
        color="b",
        linewidth=1,
        marker="o",
        alpha=0.75,
        label="self-calibration (approx)"
    )
    plt.legend()
    plt.xlabel(
        "# of antennas",
        fontsize=15
    )
    plt.ylabel(
        "Antenna Phase Error (rad)",
        fontsize=15
    )
    plt.show()
    exit()
