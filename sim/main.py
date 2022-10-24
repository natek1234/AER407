from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt

from paths import PathsImage, Location
from speed import Terminator, SpeedControl

DT = (60 * 5)  # [s]
MAX_SPEED = 2.5  # [m/s]
SECS_PER_DAY = 24 * 60 * 60  # [s/day]

path = PathsImage.get_global_path()

sim = SimpleNamespace()
sim.t = [0]  # [s]
sim.dist = [0]  # [km]

sim.pos = []
sim.speed = []  # [m/s]
sim.term_lon = []  # [deg]
sim.phi = []  # [deg]

sim.models = SimpleNamespace()
sim.models.term = Terminator(path.points[0].lon)
sim.models.speed = SpeedControl(MAX_SPEED)

while True:
    # Compute state values at current time t_i
    sim.pos.append(path.point_at_dist(sim.dist[-1]))
    sim.speed.append(sim.models.speed.speed)
    sim.term_lon.append(sim.models.term.longitude)
    sim.phi.append(Location.subtract_longitudes(sim.pos[-1].lon, sim.term_lon[-1]))

    # Exit only when have traversed entire path
    if sim.dist[-1] >= path.total_distance() or sim.t[-1] > (300 * SECS_PER_DAY):
        break

    # Propogate to next time-step at t_{i+1}
    sim.t.append(sim.t[-1] + DT)
    sim.dist.append(sim.dist[-1] + (sim.speed[-1] * 1e-3 * DT))
    sim.models.term.step(DT, sim)
    sim.models.speed.step(DT, sim)

sim.t = np.array(sim.t)
sim.days = sim.t / SECS_PER_DAY
sim.dist = np.array(sim.dist)

plt.plot(sim.days, sim.phi)
plt.show()
