from abc import ABC, abstractmethod


class Model(ABC):
    def __init__(self, sim):
        self.sim = sim

    @abstractmethod
    def step(self, dt: float):
        pass


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
