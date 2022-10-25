from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from paths import PathsImage
from mercury import Terminator, Sun, SurfaceThermal
from traversal import Traversal, SpeedControl

plt.style.use("ggplot")
plt.rcParams["figure.autolayout"] = True

DT = 60 * 10  # [s]
SECS_PER_DAY = 24 * 60 * 60  # [s/day]

path = PathsImage.get_global_path()

sim = SimpleNamespace()
sim.path = path
sim.t = [0]  # [s]

sim.pos = []
sim.dist = []  # [km]
sim.speed = []  # [m/s]
sim.term_lon = []  # [deg]
sim.phi = []  # [deg]
sim.surf_temp = []  # [degC]
sim.sun_elevation = []  # [deg]

sim.models = SimpleNamespace()
sim.models.term = Terminator(sim)
sim.models.traverse = Traversal(sim)
sim.models.speed = SpeedControl(sim)
sim.models.surf_temp = SurfaceThermal(sim)
sim.models.sun = Sun(sim)

with tqdm(total=int(path.total_distance()) + 1, unit="km") as pbar:
    while True:
        # Compute state values at current time t_i
        sim.pos.append(sim.models.traverse.pos)
        sim.dist.append(sim.models.traverse.dist)
        sim.speed.append(sim.models.speed.speed)
        sim.term_lon.append(sim.models.term.longitude)
        sim.phi.append(sim.models.traverse.phi)
        sim.surf_temp.append(sim.models.surf_temp.surface_temp)
        sim.sun_elevation.append(sim.models.sun.elevation)

        # Exit only when have traversed entire path
        if len(sim.dist) > 2:
            pbar.update(sim.dist[-1] - sim.dist[-2])
        if sim.dist[-1] >= sim.path.total_distance() or sim.t[-1] > (
            300 * SECS_PER_DAY
        ):
            break

        # Propogate to next time-step at t_{i+1}
        sim.t.append(sim.t[-1] + DT)
        sim.models.term.step(DT)
        sim.models.surf_temp.step(DT)
        sim.models.speed.step(DT)
        sim.models.traverse.step(DT)
        sim.models.sun.step(DT)

sim.t = np.array(sim.t)
sim.days = sim.t / SECS_PER_DAY
sim.dist = np.array(sim.dist)

plt.plot(sim.days, sim.sun_elevation)
plt.show()
