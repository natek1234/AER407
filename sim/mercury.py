import numpy as np

from utils import Model, degC_to_K, K_to_degC, snap_angle_range


class Terminator(Model):
    """Model of terminator movement."""

    SPEED = 360 / (175.94 * 24 * 60 * 60)  # [deg/s]

    def __init__(self, sim):
        super().__init__(sim)
        self.longitude = sim.path.points[0].lon - (90 - 86.5)

    def step(self, dt: float):
        self.longitude += self.SPEED * dt


class Sun(Model):
    def __init__(self, sim):
        super().__init__(sim)
        self.compute()

    def compute(self):
        self.elevation = max(self.sim.models.traverse.alpha, 0)
        self.azimuth = snap_angle_range(90 - self.sim.models.traverse.bearing)

    def step(self, dt: float):
        self.compute()


class SurfaceThermal(Model):
    """Model of Mercury's surface temperature."""

    R_AU = 0.38

    def __init__(self, sim):
        super().__init__(sim)
        self.compute_temp()

    def step(self, dt: float):
        self.compute_temp()

    def compute_temp(self) -> float:
        phi = self.sim.models.traverse.phi
        self.surface_temp = self._compute_surface_temp(phi, self.R_AU)

    def _compute_surface_temp(self, phi: float, r: float) -> float:
        assert 0.3075 <= r <= 0.4667
        T_COLD = 110  # [K]
        if 90 <= phi <= 270 or -270 <= phi <= -90:
            return K_to_degC(T_COLD)
        T_subsolar = 407 + (8 / np.sqrt(r))
        T_K = (
            T_subsolar * np.cos(np.deg2rad(phi)) ** 0.25
            + T_COLD * (np.abs(phi) / 90) ** 3
        )
        return K_to_degC(T_K)

    def _phi_from_surface_temp(self, temp: float, r: float) -> float:
        sol = root_scalar(
            lambda phi: self.compute_surface_temp(phi, r) - temp, bracket=(0, 180)
        )
        assert sol.converged
        return sol.root
