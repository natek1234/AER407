from paths import Location
from utils import Model, snap_angle_range


class Traversal(Model):
    def __init__(self, sim):
        super().__init__(sim)
        self.dist = 0
        self.pos = sim.path.points[0]
        self._compute()

    def step(self, dt: float):
        self.dist += self.sim.models.speed.speed * 1e-3 * dt
        self.pos = self.sim.path.point_at_dist(self.dist)
        self._compute()

    def _compute(self):
        self.alpha = Location.subtract_longitudes(
            self.pos.lon, self.sim.models.term.longitude
        )
        self.phi = snap_angle_range(90 - self.alpha)

        # Compute which direction we're headed
        slighty_ahead = self.sim.path.point_at_dist(self.dist + 0.1)
        self.bearing = self.pos.bearing_to(slighty_ahead)


class SpeedControl(Model):
    TEMP_MAX = 50  # [degC]
    MAX_SPEED = 1.6  # [m/s]

    TEMP_P_RANGE = 5  # [degC]
    TEMP_P_MIN = TEMP_MAX - TEMP_P_RANGE

    def __init__(self, sim):
        super().__init__(sim)
        self.speed = 0

    def step(self, dt: float):
        # Simple proportional speed control based off surface temp
        surf_temp = self.sim.models.surf_temp.surface_temp
        if surf_temp > self.TEMP_MAX:
            self.speed = 0
        elif surf_temp > self.TEMP_P_MIN:
            gain = 1 - (surf_temp - self.TEMP_P_MIN) / self.TEMP_P_RANGE
            self.speed = gain * self.MAX_SPEED
        else:
            self.speed = self.MAX_SPEED
