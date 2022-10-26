import numpy as np

from utils import Model, Plane


class Power(Model):
    SOLAR_FLUX = 6278  # [W / m^2], Minimum at aphelion

    GEN_EFFICIENCY = 0.2
    SOLAR_PANELS = [
        Plane(1 * 2.3, [1, 0, 0]),
        # Plane(1 * 1, [0, 1, 0]),
        # Plane(1 * 1, [0, -1, 0]),
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
