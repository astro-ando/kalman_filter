"""
Filter some noisy data using a linear regression

Author: Andrew Alder
Date created: 7th September 2024
"""

import numpy as np
from matplotlib import pyplot as plt
from dataclasses import dataclass

NUM_X_DATA_POINTS = 10
NUM_Y_DATA_POINTS = 10


@dataclass
class Point:
    x: float
    y: float


def generate_ellipse_data(semi_major_axis: float, semi_minor_axis: float):
    """
    Generate data for an ellipse
    (x/a)**2 + (y/b)**2 = 1
    :param semi_major_axis: ellipse semi major axis [m]
    :param semi_minor_axis: ellipse semi minor axis [m]
    :return: data representing an ellipse
    """

    x_data_points = np.linspace(-semi_major_axis, semi_major_axis, NUM_X_DATA_POINTS)

    ellipse_data = []
    for x in x_data_points:
        y = semi_minor_axis * (1 - (x / semi_major_axis) ** 2) ** 0.5
        # The y solution has two values, a positive and a negative
        ellipse_data.append(Point(x=x, y=y))
        ellipse_data.append(Point(x=x, y=-y))

    return ellipse_data


def plot_data(data_points: list) -> None:

    plt.figure()
    for point in data_points:
        plt.scatter(x=point.x, y=point.y, color='blue', marker='x')
    plt.show()


def main() -> None:
    ellipse_data = generate_ellipse_data(semi_major_axis=5, semi_minor_axis=3)
    plot_data(data_points=ellipse_data)


if __name__ == "__main__":
    main()
