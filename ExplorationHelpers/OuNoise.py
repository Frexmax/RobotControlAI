import numpy as np
from numba import jit


# ----------------------------------- SUPPORTS ONLY 1D ACTIONS ----------------------------------- #


class OuNoise:

    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.mean_shape = mean.shape[0]
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()
        self.x_prev = 0

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

    @staticmethod
    @jit(nopython=True, cache=True)
    def calculate_noise(x_prev, theta, mean, dt, std_dev, mean_shape):
        return x_prev + theta * (mean - x_prev) * dt + std_dev * np.sqrt(dt) * np.random.randn(mean_shape)

    def __call__(self):
        x = self.calculate_noise(self.x_prev, self.theta, self.mean, self.dt, self.std_dev, self.mean_shape)
        self.x_prev = x  # STORE X TO MAKE NOISE DEPENDENT ON PREVIOUS ONE
        return x
