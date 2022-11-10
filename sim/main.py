from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import SECS_PER_DAY
from paths import PathsImage
from mercury import Terminator, Sun, SurfaceThermal
from traversal import Traversal, SpeedControl
from power import Power
from thermal import Thermal

plt.style.use("ggplot")
plt.rcParams["figure.autolayout"] = True

DT = 60 * 10  # [s]

path = PathsImage.get_global_path()

sim = SimpleNamespace()
sim.path = path
sim.t = [0]  # [s]

sim.pos = []
sim.dist = []  # [km]
sim.speed = []  # [m/s]
sim.bearing = []  # [deg]
sim.term_lon = []  # [deg]
sim.phi = []  # [deg]
sim.surf_temp = []  # [degC]
sim.sun_elevation = []  # [deg]
sim.sun_azimuth = []  # [deg]
sim.power_gen = []  # [W]
sim.heat_sun = []  # [W]

sim.models = SimpleNamespace()
sim.models.term = Terminator(sim)
sim.models.traverse = Traversal(sim)
sim.models.speed = SpeedControl(sim)
sim.models.surf_temp = SurfaceThermal(sim)
sim.models.sun = Sun(sim)
sim.models.power = Power(sim)
sim.models.thermal = Thermal(sim)

PBAR_FORMAT = (
    "{l_bar}{bar}| {n:.3f}/{total:.0f} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
)
with tqdm(total=path.total_distance(), unit="km", bar_format=PBAR_FORMAT) as pbar:
    while True:

        # Compute state values at current time t_i
        sim.pos.append(sim.models.traverse.pos)
        sim.dist.append(sim.models.traverse.dist)
        sim.speed.append(sim.models.speed.speed)
        sim.bearing.append(sim.models.traverse.bearing)
        sim.term_lon.append(sim.models.term.longitude)
        sim.phi.append(sim.models.traverse.phi)
        sim.surf_temp.append(sim.models.surf_temp.surface_temp)
        sim.sun_elevation.append(sim.models.sun.elevation)
        sim.sun_azimuth.append(sim.models.sun.azimuth)
        sim.power_gen.append(sim.models.power.generated)
        sim.heat_sun.append(sim.models.thermal.power_sun)

        # Exit only when have traversed entire path
        if sim.dist[-1] >= sim.path.total_distance():
            break
        if len(sim.dist) > 2:
            pbar.update(sim.dist[-1] - sim.dist[-2])

        # Propogate to next time-step at t_{i+1}
        sim.t.append(sim.t[-1] + DT)
        sim.models.term.step(DT)
        sim.models.surf_temp.step(DT)
        sim.models.speed.step(DT)
        sim.models.traverse.step(DT)
        sim.models.sun.step(DT)
        sim.models.power.step(DT)
        sim.models.thermal.step(DT)

sim.t = np.array(sim.t)
sim.days = sim.t / SECS_PER_DAY
sim.dist = np.array(sim.dist)

PLOT_STUFF = True
plt.ion()
if PLOT_STUFF:
    # Plot a bunch of shit
    fig, axs = plt.subplots(2, 2, sharex="all", figsize=(10, 6))
    fig.suptitle("Traversal")
    axs[0, 0].plot(sim.days, [p.lat for p in sim.pos])
    axs[0, 0].set_title("Latitude")
    axs[0, 0].set_ylabel("[deg]")
    axs[1, 0].plot(sim.days, [p.lon for p in sim.pos])
    axs[1, 0].set_title("Longitude")
    axs[1, 0].set_ylabel("[deg]")
    axs[0, 1].plot(sim.days, sim.speed)
    axs[0, 1].set_title("Speed")
    axs[0, 1].set_ylabel("[m/s]")
    axs[1, 1].plot(sim.days, np.unwrap(sim.bearing, period=360))
    axs[1, 1].set_title("Bearing")
    axs[1, 1].set_ylabel("[deg]")
    axs[-1, 0].set_xlabel("Time [days]")
    fig, axs = plt.subplots(2, 2, sharex="all", figsize=(10, 6))
    fig.suptitle("Thermal")
    axs[0, 0].plot(sim.days, sim.surf_temp)
    axs[0, 0].set_title("Surface Temp")
    axs[0, 0].set_ylabel("[degC]")
    axs[1, 0].plot(sim.days, sim.phi)
    axs[1, 0].set_title("Subsolar Phi Angle")
    axs[1, 0].set_ylabel("[deg]")
    axs[0, 1].plot(sim.days, sim.heat_sun)
    axs[0, 1].set_title("Solar Power")
    axs[0, 1].set_ylabel("[W]")
    fig, axs = plt.subplots(2, 2, sharex="all", figsize=(10, 6))
    axs[0, 0].plot(sim.days, sim.power_gen)
    axs[0, 0].set_title("Generated Solar Power")
    axs[0, 0].set_ylabel("[W]")
    axs[0, 1].plot(sim.days, sim.sun_azimuth)
    axs[0, 1].set_title("Local Sun Azimuth")
    axs[0, 1].set_ylabel("[deg]")
    axs[1, 1].plot(sim.days, sim.sun_elevation)
    axs[1, 1].set_title("Sun Elevation")
    axs[1, 1].set_ylabel("[deg]")
    plt.show()
