"""
Filter some noisy data using a linear regression

Author: Andrew Alder
Date created: 7th September 2024
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import least_squares


def left_psuedo_inverse(A: np.array):
    return np.linalg.inv(A.transpose() @ A) @ A.transpose()


def analyse_line() -> None:
    gradient = -3
    y_offset = 5

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

    theta = left_psuedo_inverse(A) @ y_noise
    m_guess = theta[0]
    c_guess = theta[1]
    y_filtered = A @ theta

    def residuals(params, t_value, y_value):
        m, c = params
        return y_value - (m*t_value + c)

    result = least_squares(residuals, [m_guess, c_guess], args=(t.reshape(num_measurements), y_noise.reshape(num_measurements)))
    m_refined, c_refined = result.x
    print(f"Initial guess line: gradient={m_guess} y-intercept={c_guess}")
    print(f"Refined line: gradient={m_refined} y-intercept={c_refined}")

    # Generate a plot of the optimised parameters
    y_refined = m_refined * t + c_refined

    plt.figure()
    plt.title("Filtered")
    plt.plot(t, y, color="black")
    plt.scatter(t, y_noise, color="blue", marker="x")
    plt.plot(t, y_filtered, color="red", linestyle="--")
    plt.plot(t, y_refined, color="green", linestyle="--")
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

    # Generate Measurements
    radius = 5
    h = 3
    k = 10
    num_measurements = 50
    x = np.linspace(h - radius, h + radius, num_measurements)
    y_plus = (radius**2 - (x - h)**2)**0.5 + k
    y = y_plus
    x = x.reshape([num_measurements, 1])
    y = y.reshape([num_measurements, 1])

    # Add Gaussian noise to measurements
    mean = 0  # Mean (mu)
    std_dev = 0.5  # Standard deviation (sigma)
    x_noise = x + np.random.normal(mean, std_dev, size=np.shape(x))
    y_noise = y + np.random.normal(mean, std_dev, size=np.shape(y))

    # Filter data
    # Problem is non-linear so reformulate such that it is linear in k, h, and radius
    x_filtered = x_noise
    y_filtered = y_noise

    A = np.concatenate([x_filtered, y_filtered, np.ones([num_measurements, 1])], axis=1)
    b = -(x_noise ** 2) - (y_noise ** 2)

    theta = left_psuedo_inverse(A) @ b

    D = theta[0][0]
    E = theta[1][0]
    F = theta[2][0]

    h_guess = -D / 2
    k_guess = -E / 2
    r_guess = (h_guess**2 + k_guess**2 - F) ** 0.5

    t = np.linspace(0, np.pi * 2, 100)
    x_model = r_guess * np.cos(t) + h_guess
    y_model = r_guess * np.sin(t) + k_guess

    def residuals(params, x, y):
        r, h, k = params
        return ((x - h)**2 + (y - k)**2)**0.5 - r

    result = least_squares(residuals, [r_guess, h_guess, k_guess], args=(x_noise.reshape(num_measurements), y_noise.reshape(num_measurements)))
    r_refined, h_refined, k_refined = result.x
    print(f"Refined circle: center=({h_refined}, {k_refined}), radius={r_refined}")

    # Generate a plot of the optimised parameters
    t = np.linspace(0, np.pi * 2, 100)
    x_refined = r_refined * np.cos(t) + h_refined
    y_refined = r_refined * np.sin(t) + k_refined

    plt.figure()
    plt.title("Filtered")
    plt.plot(x, y, color="black")
    plt.scatter(x_noise, y_noise, color="blue", marker="x")
    plt.plot(x_model, y_model, color="red", linestyle="--")
    plt.plot(x_refined, y_refined, color="green", linestyle="--")
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    plt.show()


def main() -> None:
    analyse_line()
    # analyse_parabola()
    # analyse_semicircle()


if __name__ == "__main__":
    main()
