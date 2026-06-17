"""Interactive 3D PyVista visualization."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pyvista as pv

from data_loader.geojson_loader import BuildingFootprint, WorldBounds

logger = logging.getLogger(__name__)


class NavigationVisualizer:
    """Interactive 3D viewer for buildings, paths, and drone trajectory."""

    def __init__(
        self,
        buildings: Sequence[BuildingFootprint],
        bounds: WorldBounds,
        show_gui: bool = True,
    ) -> None:
        self.buildings = list(buildings)
        self.bounds = bounds
        self.show_gui = show_gui
        self.plotter = pv.Plotter(off_screen=not show_gui)

    def _add_buildings(self) -> None:
        for b in self.buildings:
            coords = np.array(b.polygon.exterior.coords)
            n = len(coords) - 1
            poly = pv.PolyData(coords[:, :2], faces=[np.hstack([[n], np.arange(n)])])
            mesh = poly.extrude([0, 0, b.height_m], capping=True)
            self.plotter.add_mesh(mesh, color="#6c7a89", opacity=0.9)

    def show(
        self,
        start: tuple[float, float, float],
        goal: tuple[float, float, float],
        planned_path: Sequence[tuple[float, float, float]] | None,
        trajectory: Sequence[tuple[float, float, float]],
        title: str = "Drone Navigation",
        output_path: str | Path | None = None,
    ) -> None:
        """Display buildings, start, goal, planned path, and actual trajectory."""
        self.plotter.clear()
        self._add_buildings()

        self.plotter.add_mesh(pv.Sphere(radius=3, center=start), color="lime", label="Start")
        self.plotter.add_mesh(pv.Sphere(radius=3, center=goal), color="red", label="Goal")

        if planned_path and len(planned_path) > 1:
            pp = np.array(planned_path)
            self.plotter.add_lines(pp, color="yellow", width=3, label="Planned")

        if len(trajectory) > 1:
            traj = np.array(trajectory)
            self.plotter.add_lines(traj, color="cyan", width=4, label="Trajectory")
            self.plotter.add_mesh(pv.Sphere(radius=2, center=traj[-1]), color="deepskyblue", label="Drone")

        self.plotter.add_axes()
        self.plotter.add_legend()
        self.plotter.camera_position = "iso"
        self.plotter.add_text(title, position="upper_edge")

        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.plotter.screenshot(str(path))
            logger.info("Saved %s", path)

        if self.show_gui:
            self.plotter.show(title=title)
        self.plotter.close()
