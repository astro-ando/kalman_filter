"""
Use least squares and other simple estimation techniques

Author: Andrew Alder
Date created: 8th September 2024
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    mass_gold_bar = 1.024  # [kg]

    number_of_measurements = 100
    measurement_number = np.arange(start=1, stop=number_of_measurements + 1)
    mean = 0  # [kg]
    std = 0.010  # [kg]

    def measure_mass() -> float:
        return mass_gold_bar + np.random.normal(mean, std)

    mass_measured = [measure_mass()]
    mass_estimated_alpha = [mass_measured[0]]  # Use the first measurement as the estimate
    mass_estimated_mean = [mass_measured[0]]  # Use the first measurement as the estimate
    i = 1
    while i < number_of_measurements:

        # Estimate the mass with an alpha filter as per: https://www.kalmanfilter.net/alphabeta.html
        mass_measured.append(measure_mass())
        mass_estimate_new = mass_estimated_alpha[i - 1] + (1 / (i + 1)) * (mass_measured[i] - mass_estimated_alpha[i - 1])
        mass_estimated_alpha.append(mass_estimate_new)
        mass_estimated_mean.append(np.mean(mass_measured))
        i += 1

    final_estimate_filter = mass_estimated_alpha[-1]
    error_filter = final_estimate_filter - mass_gold_bar
    print(f"alpha-beta filter mass estimate: {final_estimate_filter:.5f} kg | Error: {error_filter:.5f} kg")

    # Now perform a batch linear least squares on all the measurements. This is much more computational expensive
    # but should give the same thing
    y_vec = np.array(mass_measured).reshape((number_of_measurements, 1))
    A_matrix = np.ones((number_of_measurements, 1))
    x = np.linalg.inv(A_matrix.transpose() @ A_matrix) @ A_matrix.transpose() @ y_vec
    final_estimate_lls = x[0][0]
    error_lls = final_estimate_lls - mass_gold_bar
    print(f"linear least squares mass estimate: {final_estimate_lls:.5f} kg | Error: {error_lls:.5f} kg")

    # And the average should give the same as the LLS
    final_estimate_mean = np.mean(mass_measured)
    error_mean = final_estimate_mean - mass_gold_bar
    print(f"Mean mass estimate: {final_estimate_mean:.5f} kg | Error: {error_mean:.5f} kg")

    print(mass_measured)

    # Plot the results
    plt.figure()
    plt.scatter(x=measurement_number, y=mass_measured, label="Measurement")
    plt.plot(measurement_number, mass_estimated_mean, label="Estimate mean")
    plt.plot(measurement_number, mass_estimated_alpha, label="Estimate alpha")
    plt.hlines(mass_gold_bar, xmin=0, xmax=number_of_measurements, color="red", linestyles="--", label="truth")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
