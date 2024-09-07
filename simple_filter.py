"""
Filter some noisy data using a linear regression

Author: Andrew Alder
Date created: 7th September 2024
"""

import numpy as np
from matplotlib import pyplot as plt


def left_psuedo_inverse(A: np.array):
    return np.linalg.inv(A.transpose() @ A) @ A.transpose()


def analyse_line() -> None:
    gradient = 3
    y_offset = 4

    num_measurements = 20

    t = np.linspace(-10, 10, num_measurements)
    y = gradient * t + y_offset

    plt.figure()
    plt.title("No noise")
    plt.plot(t, y, color="black")
    plt.scatter(t, y, color="blue", marker="x")

    # Noise
    # Generate random Gaussian numbers
    mean = 0  # Mean (mu)
    std_dev = 5  # Standard deviation (sigma)
    y_noise = y + np.random.normal(mean, std_dev, num_measurements)

    plt.figure()
    plt.title("Noise")
    plt.plot(t, y, color="black")
    plt.scatter(t, y_noise, color="blue", marker="x")

    # Filtered
    # y = A*x + residual -> residual = y - A*x
    # For linear least squares we want to minimise the cost function:
    # J(x) = residual^T * residual = (y - A*x)^T * (y - A*x) = (y^T - x^T * A^T) * (y - A*x)
    # J(x) = y * y^T - y * x^T * A^T - A * x * y^T + x^T * A^T * A * x
    # J(x) = y * y^T - y * x^T * A^T - y * (A*x)^T + x^T * A^T * A * x
    # J(x) = y * y^T - y * x^T * A^T - y * x^T * A^T + x^T * A^T * A * x
    # J(x) = y * y^T - 2 * y * x^T * A^T + x^T * A^T * A * x
    # Differentiate: d J(x) / d x = 0 - 2 * y * A^T + 2 * A^T * A * x = - 2 * A^T (y - A * x)
    # Set to zero to find minimum: - 2 * A^T (y - A * x) = 0 ->  A^T*y - A^T * A * x = 0 -> A^T*y = A^T * A * x
    # -> (A^T * A)^-1 * A^T*y = x

    A = np.concatenate([t.reshape([num_measurements, 1]), np.ones([num_measurements, 1])], axis=1)

    x = left_psuedo_inverse(A) @ y_noise
    y_filtered = A @ x

    plt.figure()
    plt.title("Filtered")
    plt.plot(t, y, color="black")
    plt.scatter(t, y_noise, color="blue", marker="x")
    plt.plot(t, y_filtered, color="red", linestyle="--")
    plt.show()


def analyse_parabola() -> None:
    coefficient = 3
    y_offset = 4

    num_measurements = 20

    t = np.linspace(-10, 10, num_measurements)
    y = coefficient * t**2 + y_offset

    plt.figure()
    plt.title("No noise")
    plt.plot(t, y, color="black")
    plt.scatter(t, y, color="blue", marker="x")

    # Noise
    # Generate random Gaussian numbers
    mean = 0  # Mean (mu)
    std_dev = 10  # Standard deviation (sigma)
    y_noise = y + np.random.normal(mean, std_dev, num_measurements)

    plt.figure()
    plt.title("Noise")
    plt.plot(t, y, color="black")
    plt.scatter(t, y_noise, color="blue", marker="x")

    # Filtered
    # y = A*x + residual -> residual = y - A*x
    # For linear least squares we want to minimise the cost function:
    # J(x) = residual^T * residual = (y - A*x)^T * (y - A*x) = (y^T - x^T * A^T) * (y - A*x)
    # Solve: (A^T * A)^-1 * A^T*y = x

    A = np.concatenate([t.reshape([num_measurements, 1])**2, np.ones([num_measurements, 1])], axis=1)

    x = left_psuedo_inverse(A) @ y_noise
    y_filtered = A @ x

    plt.figure()
    plt.title("Filtered")
    plt.plot(t, y, color="black")
    plt.scatter(t, y_noise, color="blue", marker="x")
    plt.plot(t, y_filtered, color="red", linestyle="--")
    plt.show()


def main() -> None:
    # analyse_line()
    analyse_parabola()


if __name__ == "__main__":
    main()
