"""
Filter some noisy data using a linear regression

Author: Andrew Alder
Date created: 7th September 2024
"""

import numpy as np
from matplotlib import pyplot as plt

NUM_DATA_POINTS = 20


def main() -> None:
    semi_major_axis = 5
    semi_minor_axis = 3
    
    x_offset = 0
    y_offset = 0

    ellipse_data_points = []
    t = np.linspace(0, 2 * np.pi, NUM_DATA_POINTS)
    x = y_offset + semi_major_axis * np.cos(t)
    y = x_offset + semi_minor_axis * np.sin(t)

    plt.figure()
    plt.title("No noise")
    plt.plot(x, y, color='black')
    plt.scatter(x=x, y=y, color='blue', marker='x')
    plt.show()

    # Parameters
    mean = 0  # Mean (mu)
    std_dev = 0.2  # Standard deviation (sigma)
    # Generate random Gaussian numbers
    x_noise = x + np.random.normal(mean, std_dev, NUM_DATA_POINTS)
    y_noise = y + np.random.normal(mean, std_dev, NUM_DATA_POINTS)

    plt.figure()
    plt.title("Noise")
    plt.plot(x, y, color='black')
    plt.scatter(x=x_noise, y=y_noise, color='blue', marker='x')
    plt.show()




if __name__ == "__main__":
    main()
