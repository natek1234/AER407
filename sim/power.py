import numpy as np

from utils import Model, Plane


class Power(Model):
    # [W / m^2]
    MIN_SOLAR_FLUX = 6278  # Minimum at aphelion
    MAX_SOLAR_FLUX = 13000
    SOLAR_FLUX = MIN_SOLAR_FLUX

    GEN_EFFICIENCY = 0.2 * 0.8
    SOLAR_PANELS = [
        Plane(0.2 * 0.3, [1, 0, 0]),
        Plane(0.2 * 0.3, [1, 0, 0]),
        Plane(0.08, [0, 1, 0]),
        Plane(0.08, [0, -1, 0]),
    ]

    def __init__(self, sim):
        super().__init__(sim)
        self.step(0)

    def step(self, dt: float):
        sun_vec = self.sim.models.sun.vec
        tot_area = 0
        for plane in self.SOLAR_PANELS:
            tot_area += plane.projected_area(sun_vec)
        self.received = self.SOLAR_FLUX * tot_area
        self.generated = self.GEN_EFFICIENCY * self.received
