from paths import Location
from utils import Model, snap_angle_range, SECS_PER_DAY


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
    MAX_SPEED = 1.6  # [m/s]

    TEMP_P_RANGE = 5  # [degC]
    TOO_COLD_DISTS = [  # [km]
        # fmt: off
        (  500,   650),
        ( 1700,  1950),
        ( 4800,  5000),
        ( 6150,  6400),
        ( 9100,  9350),
        ( 9750,  9900),
        (13700, 14200),
        # fmt: on
    ]

    def __init__(self, sim):
        super().__init__(sim)
        self.speed = 0
        self.temp_max = 20
        self.t_excess = 0

    def step(self, dt: float):
        # Simple proportional speed control based on surface temp
        self.temp_max = self.target_temp_max()
        temp_P_min = self.temp_max - self.TEMP_P_RANGE
        surf_temp = self.sim.models.surf_temp.surface_temp
        if surf_temp > self.temp_max:
            self.speed = 0
        elif surf_temp > temp_P_min:
            gain = 1 - (surf_temp - temp_P_min) / self.TEMP_P_RANGE
            self.speed = gain * self.MAX_SPEED
        else:
            self.speed = self.MAX_SPEED
        self.t_excess = (1 - self.speed / self.MAX_SPEED) * dt

    def target_temp_max(self):
        d = self.sim.models.traverse.dist
        for d_start, d_end in self.TOO_COLD_DISTS:
            start_before = (d_end - d_start) * 2
            if (d_start - start_before) < d < d_end:
                return 65
        return 55
