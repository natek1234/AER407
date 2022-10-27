from abc import ABC, abstractmethod

import numpy as np

SECS_PER_DAY = 24 * 60 * 60  # [s/day]


class Model(ABC):
    def __init__(self, sim):
        self.sim = sim

    @abstractmethod
    def step(self, dt: float):
        pass


class Plane:
    def __init__(self, area: float, normal: np.ndarray):
        self.area = area
        self.normal = np.array(normal) / np.linalg.norm(normal)

    def projected_area(self, view_from: np.ndarray) -> float:
        return max(self.area * np.dot(self.normal, view_from), 0)


def degC_to_K(degC: float) -> float:
    return 273.15 + degC


def K_to_degC(K: float) -> float:
    return K - 273.15


def snap_angle_range(deg: float) -> float:
    deg = deg % 360
    deg = (deg + 360) % 360
    if deg > 180:
        deg -= 360
    return deg
