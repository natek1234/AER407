from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import SECS_PER_DAY
from paths import PathsImage
from mercury import Terminator, Sun, SurfaceThermal
from traversal import Traversal, SpeedControl
from power import Power

plt.style.use("ggplot")
plt.rcParams.update(
    {
        "figure.autolayout": True,
        "font.family": "sans-serif",
        "font.sans-serif": "PT Sans",
    }
)

DT = 60 * 10  # [s]

path = PathsImage.get_global_path()

sim = SimpleNamespace()
sim.path = path
sim.t = [0]  # [s]

sim.pos = []
sim.dist = []  # [km]
sim.speed = []  # [m/s]
sim.t_excess = []  # [s]
sim.bearing = []  # [deg]
sim.term_lon = []  # [deg]
sim.phi = []  # [deg]
sim.surf_temp = []  # [degC]
sim.sun_elevation = []  # [deg]
sim.sun_azimuth = []  # [deg]
sim.power_gen = []  # [W]

sim.models = SimpleNamespace()
sim.models.term = Terminator(sim)
sim.models.traverse = Traversal(sim)
sim.models.speed = SpeedControl(sim)
sim.models.surf_temp = SurfaceThermal(sim)
sim.models.sun = Sun(sim)
sim.models.power = Power(sim)

PBAR_FORMAT = (
    "{l_bar}{bar}| {n:.3f}/{total:.0f} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
)
with tqdm(total=path.total_distance(), unit="km", bar_format=PBAR_FORMAT) as pbar:
    while True:

        # Compute state values at current time t_i
        sim.pos.append(sim.models.traverse.pos)
        sim.dist.append(sim.models.traverse.dist)
        sim.speed.append(sim.models.speed.speed)
        sim.t_excess.append(sim.models.speed.t_excess)
        sim.bearing.append(sim.models.traverse.bearing)
        sim.term_lon.append(sim.models.term.longitude)
        sim.phi.append(sim.models.traverse.phi)
        sim.surf_temp.append(sim.models.surf_temp.surface_temp)
        sim.sun_elevation.append(sim.models.sun.elevation)
        sim.sun_azimuth.append(sim.models.sun.azimuth)
        sim.power_gen.append(sim.models.power.generated)

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

sim.t = np.array(sim.t)
sim.days = sim.t / SECS_PER_DAY
sim.dist = np.array(sim.dist)


def plot_traversal():
    fig, axs = plt.subplots(2, 2, num="traversal", sharex="all", figsize=(10, 6))
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
    for i in range(axs.shape[-1]):
        axs[-1, i].set_xlabel("Mission Time [days]")


def plot_thermal():
    fig, axs = plt.subplots(2, 1, num="thermal", sharex="all", figsize=(6, 6))
    fig.suptitle("Thermal")
    axs[0].plot(sim.days, sim.surf_temp)
    axs[0].set_title("Surface Temp")
    axs[0].set_ylabel("[degC]")
    axs[1].plot(sim.days, sim.phi)
    axs[1].set_title("Subsolar Phi Angle")
    axs[1].set_ylabel("[deg]")
    axs[1].set_xlabel("Mission Time [day]")


def plot_sun():
    fig, axs = plt.subplots(2, 1, num="sun", sharex="all", figsize=(10, 6))
    axs[0].plot(sim.days, sim.sun_azimuth)
    axs[0].set_title("Local Sun Azimuth")
    axs[0].set_ylabel("[deg]")
    axs[1].plot(sim.days, sim.sun_elevation)
    axs[1].set_title("Sun Elevation")
    axs[1].set_ylabel("[deg]")
    axs[1].set_xlabel("Mission Time [day]")
    plt.show()


def plot_power_gen():
    fig, ax = plt.subplots(num="power-gen", sharex="all", figsize=(10, 6))
    min_power_gen = np.array(sim.power_gen)
    max_power_gen = (
        min_power_gen / sim.models.power.SOLAR_FLUX * sim.models.power.MAX_SOLAR_FLUX
    )
    ax.fill_between(sim.days, min_power_gen, max_power_gen, alpha=0.5)
    ax.plot(sim.days, (min_power_gen + max_power_gen) / 2)
    ax.set_title("Generated Solar Power")
    ax.set_ylabel("[W]")
    ax.set_xlabel("Mission Time [day]")


def plot_stoppage_time():
    """Plot possible stoppage time per mission day."""
    ts, t_excess = [], []
    prev_day = 0
    day_excess = 0
    for i, day in enumerate(sim.days):
        if int(day) != prev_day:
            ts.append(prev_day)
            t_excess.append(day_excess)
            day_excess = 0
            prev_day = day
        else:
            day_excess += sim.t_excess[i]
    ts, t_excess = np.array(ts), np.array(t_excess)
    t_excess /= 60 * 60  # Convert secs to hours

    fig, axs = plt.subplots(2, 1, num="stoppage-time", sharex="all")
    axs[0].plot(sim.days, sim.speed)
    axs[0].set_ylabel("Speed [m/s]")
    axs[0].set_title("Required Speed w/ No Stopping")
    axs[1]._get_lines.get_next_color()
    axs[1].bar(ts, t_excess, align="center", color=axs[1]._get_lines.get_next_color())
    axs[1].set_xlabel("Mission Day")
    axs[1].set_ylabel("Stoppage Time [hour]")
    axs[1].set_title("Maximum Stoppage Time Per Mission Day")


PLOT_STUFF = True
plt.ion()
if PLOT_STUFF:
    # Plot a bunch of shit
    # plot_traversal()
    # plot_thermal()
    # plot_sun()
    plot_power_gen()
    # plot_stoppage_time()
