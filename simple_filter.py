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
    a_coefficient = -2
    b_coefficient = 3
    c_coefficient = 6

    num_measurements = 20

    t = np.linspace(-10, 10, num_measurements)
    y = a_coefficient * t**2 + b_coefficient * t + c_coefficient

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

    A = np.concatenate(
        [t.reshape([num_measurements, 1]) ** 2, t.reshape([num_measurements, 1]), np.ones([num_measurements, 1])],
        axis=1,
    )

    x = left_psuedo_inverse(A) @ y_noise
    y_filtered = A @ x

    plt.figure()
    plt.title("Filtered")
    plt.plot(t, y, color="black")
    plt.scatter(t, y_noise, color="blue", marker="x")
    plt.plot(t, y_filtered, color="red", linestyle="--")
    plt.show()


def analyse_semicircle() -> None:
    radius = 5
    h = 3
    k = 4

    num_measurements = 20

    x = np.linspace(h - radius, h + radius, num_measurements)
    y_plus = (radius**2 - (x - h)**2)**0.5 + k
    y = y_plus
    x = x.reshape([num_measurements, 1])
    y = y.reshape([num_measurements, 1])

    # plt.figure()
    # plt.title("No noise")
    # plt.plot(x, y, color="black")
    # plt.scatter(x, y, color="blue", marker="x")
    # ax = plt.gca()
    # ax.set_aspect("equal", adjustable="box")
    # plt.show()

    # Noise
    # Generate random Gaussian numbers
    mean = 0  # Mean (mu)
    std_dev = 0.5  # Standard deviation (sigma)
    x_noise = x + np.random.normal(mean, std_dev, size=np.shape(x))
    y_noise = y + np.random.normal(mean, std_dev, size=np.shape(y))

    # plt.figure()
    # plt.title("Noise")
    # plt.plot(x, y, color="black")
    # plt.scatter(x_noise, y_noise, color="blue", marker="x")
    # ax = plt.gca()
    # ax.set_aspect("equal", adjustable="box")

    # Filtered

    x_filtered = x_noise
    y_filtered = y_noise
    # y_guess = y_noise
    h_guess = 0
    k_guess = 0
    r_guess = 0
    for i in range(1):

        A = np.concatenate([x_filtered, y_filtered, np.ones([num_measurements, 1])], axis=1)
        b = -(x_noise ** 2) - (y_noise ** 2)

        theta = left_psuedo_inverse(A) @ b

        D = theta[0][0]
        E = theta[1][0]
        F = theta[2][0]

        h_guess = -D / 2
        k_guess = -E / 2
        r_guess = (h_guess**2 + k_guess**2 - F) ** 0.5

        y_filtered = (r_guess**2 - (x_filtered - h_guess)**2)**0.5 + k_guess

        b_reconstructed = A @ theta

        x_filtered = ((-b_reconstructed - y_filtered**2)**2)**0.25

        # b_reconstructed = A @ theta
        # x_filtered = (radius**2 - (y_guess - k)**2)**0.5 + h

    plt.figure()
    plt.title("Filtered")
    plt.plot(x, y, color="black")
    plt.scatter(x_noise, y_noise, color="blue", marker="x")
    plt.plot(x_filtered, y_filtered, color="red", linestyle="--")
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    plt.show()


# def analyse_ellipse() -> None:
#     semi_major_axis = 5
#     semi_minor_axis = 3
#
#     num_measurements = 20
#
#     x_offset = 0
#     y_offset = 0
#
#     t = np.linspace(0, 2 * np.pi, num_measurements)
#     x_points = x_offset + semi_major_axis * np.cos(t)
#     y_points = y_offset + semi_minor_axis * np.sin(t)
#
#     plt.figure()
#     plt.title("No noise")
#     plt.plot(x_points, y_points, color="black")
#     plt.scatter(x_points, y_points, color="blue", marker="x")
#
#     # Noise
#     # Generate random Gaussian numbers
#     mean = 0  # Mean (mu)
#     std_dev = 0.1  # Standard deviation (sigma)
#     x_points_noise = x_points + np.random.normal(mean, std_dev, num_measurements)
#     y_points_noise = y_points + np.random.normal(mean, std_dev, num_measurements)
#
#     plt.figure()
#     plt.title("Noise")
#     plt.plot(x_points, y_points, color="black")
#     plt.scatter(x_points_noise, y_points_noise, color="blue", marker="x")
#
#     # Filtered
#     # y = A*x + residual -> residual = y - A*x
#     # For linear least squares we want to minimise the cost function:
#     # J(x) = residual^T * residual = (y - A*x)^T * (y - A*x) = (y^T - x^T * A^T) * (y - A*x)
#     # Solve: (A^T * A)^-1 * A^T*y = x
#
#     A = np.concatenate([np.ones([num_measurements, 1]), t.reshape([num_measurements, 1])], axis=1)
#
#     x = left_psuedo_inverse(A) @ y_noise
#     y_filtered = A @ x
#
#     plt.figure()
#     plt.title("Filtered")
#     plt.plot(t, y, color="black")
#     plt.scatter(t, y_noise, color="blue", marker="x")
#     plt.plot(t, y_filtered, color="red", linestyle="--")
#     plt.show()


def main() -> None:
    # analyse_line()
    # analyse_parabola()
    analyse_semicircle()
    # analyse_ellipse()


if __name__ == "__main__":
    main()
