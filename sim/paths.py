from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import splprep, splev

R_EQUAT = 2440.5  # Equatorial radius (semi-major)
R_POLAR = 2438.3  # Polar radius (semi-minor)
R_CIRC = (R_EQUAT + R_POLAR) / 2  # Radius to use for circular calcs


@dataclass
class Location:
    xyz: np.array
    lat: float
    lon: float


@dataclass
class Path:
    name: str
    xyzs: np.array
    lats: np.array
    lons: np.array

    points: list[Location] = field(init=False, repr=False)
    sections = property(lambda self: len(self.points))

    def __post_init__(self):
        self.points = []
        for xyz, lat, lon in zip(self.xyzs.T, self.lats, self.lons):
            self.points.append(Location(xyz, lat, lon))


class PathsImage:
    """Parsed this shit with some online tool."""

    CENTRE = (303, 225)
    X_DIAMETER = 508 - 97
    Y_DIAMETER = 430 - 19
    RADIUS = (X_DIAMETER + Y_DIAMETER) / 4
    SMOOTH_FACTOR = 300
    POINTS_PER_PATH = 201

    TRAVERSE_PATHS = ["Beta", "Alpha_3", "Alpha_2", "Gam_2", "Delta_2"]

    @classmethod
    def parse_path_from_pixels(cls, name: str) -> Path:
        # Reverse points b/c traversal is in opposite direction
        points = np.array(cls.PATHS[name][::-1]).T

        # Centre x-y and flip y b/c pixels go downwards
        points[0] -= cls.CENTRE[0]
        points[1] = -(points[1] - cls.CENTRE[1])
        # Scale from pixels to kilometres
        points = points * (R_CIRC / cls.RADIUS)

        # Smooth out path and interpolate
        tck, u_orig = splprep(points, s=cls.SMOOTH_FACTOR)
        u_new = np.linspace(0, 1, cls.POINTS_PER_PATH)
        points = splev(u_new, tck)

        # Convert from x-y to latitude, longitude
        lat_rad = -np.arccos(np.hypot(points[0], points[1]) / R_CIRC)
        lon_rad = np.arctan2(points[1], points[0])
        lats, lons = np.rad2deg(lat_rad), np.rad2deg(lon_rad)

        xyzs = np.array([points[0], points[1], R_CIRC * np.sin(lat_rad)])
        return Path(name, xyzs, lats, lons)

    @classmethod
    def get_all_traverse_paths(cls) -> list[Path]:
        return [cls.parse_path_from_pixels(name) for name in cls.TRAVERSE_PATHS]

    @classmethod
    def get_global_path(cls) -> Path:
        all_paths = cls.get_all_traverse_paths()
        xyzs = np.hstack([path.xyzs[:, :-1] for path in all_paths])
        lats = np.hstack([path.lats[:-1] for path in all_paths])
        lons = np.hstack([path.lons[:-1] for path in all_paths])
        return Path(" → ".join(path.name for path in all_paths), xyzs, lats, lons)

    PATHS = {
        "Beta": [
            (369, 84),
            (386, 83),
            (392, 79),
            (403, 87),
            (413, 99),
            (419, 104),
            (428, 106),
            (434, 117),
            (443, 122),
            (452, 133),
            (464, 134),
            (471, 150),
            (477, 168),
            (483, 193),
            (486, 214),
            (478, 234),
            (475, 244),
            (471, 256),
            (471, 272),
            (473, 283),
            (467, 293),
            (462, 302),
            (452, 310),
            (443, 320),
            (440, 330),
            (431, 330),
            (420, 329),
            (418, 334),
            (411, 334),
            (406, 342),
            (406, 348),
            (393, 353),
            (385, 365),
            (368, 364),
            (361, 358),
            (354, 363),
            (345, 366),
            (341, 371),
            (334, 371),
        ],
        "Alpha_3": [
            (201, 148),
            (197, 137),
            (203, 133),
            (202, 124),
            (207, 122),
            (201, 110),
            (221, 105),
            (232, 93),
            (243, 85),
            (252, 69),
            (269, 65),
            (280, 58),
            (298, 60),
            (314, 60),
            (327, 68),
            (337, 68),
            (344, 73),
            (349, 73),
            (352, 77),
            (360, 81),
            (369, 84),
        ],
        "Alpha_2": [
            (170, 314),
            (165, 307),
            (167, 294),
            (164, 288),
            (161, 276),
            (155, 275),
            (154, 262),
            (149, 254),
            (149, 243),
            (154, 235),
            (149, 227),
            (151, 213),
            (146, 197),
            (146, 179),
            (156, 171),
            (163, 170),
            (170, 160),
            (180, 153),
            (195, 151),
            (201, 148),
        ],
        "Gam_2": [
            (302, 295),
            (287, 296),
            (279, 300),
            (273, 307),
            (266, 306),
            (261, 299),
            (253, 296),
            (250, 303),
            (250, 308),
            (244, 308),
            (239, 306),
            (230, 313),
            (221, 318),
            (212, 317),
            (203, 320),
            (193, 321),
            (185, 317),
            (170, 314),
        ],
        "Delta_2": [
            (378, 256),
            (377, 263),
            (370, 267),
            (362, 269),
            (354, 273),
            (349, 278),
            (338, 280),
            (331, 277),
            (325, 272),
            (316, 271),
            (308, 273),
            (306, 282),
            (303, 289),
            (302, 295),
        ],
    }
