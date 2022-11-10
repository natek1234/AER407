import numpy as np

from utils import Model, Plane


class Thermal(Model):
    SOLAR_FLUX = 14462  # [W / m^2], Maximum at perihelion

    X_DIM = 0.3
    Y_DIM = 0.2
    Z_DIM = 0.2

    PANELS = [
        Plane(Y_DIM * Z_DIM, [1, 0, 0]),    # X faces
        Plane(Y_DIM * Z_DIM, [-1, 0, 0]),
        Plane(X_DIM * Z_DIM, [0, 1, 0]),    # Y faces
        Plane(X_DIM * Z_DIM, [0, -1, 0]),
        Plane(X_DIM * Y_DIM, [0, 0, 1]),    # Z faces
        Plane(X_DIM * Y_DIM, [0, 0, -1]),
    ]

    def __init__(self, sim):
        super().__init__(sim)
        self.step(0)

    def step(self, dt: float):
        # Estimate heat from solar radiation
        sun_vec = self.sim.models.sun.vec
        tot_area = 0
        for plane in self.PANELS:
            tot_area += plane.projected_area(sun_vec)
        self.power_sun = self.SOLAR_FLUX * tot_area
