"""3D city mesh generation from building footprints."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import trimesh

from data_loader.geojson_loader import BuildingFootprint, WorldBounds

logger = logging.getLogger(__name__)


@dataclass
class CityMesh:
    """Generated city mesh assets."""

    buildings: trimesh.Trimesh
    ground: trimesh.Trimesh
    building_ids: list[int] = field(default_factory=list)
    bounds: WorldBounds | None = None


class CityMeshGenerator:
    """Convert building footprints into extruded 3D meshes."""

    def __init__(
        self,
        buildings: Sequence[BuildingFootprint],
        bounds: WorldBounds,
        output_dir: str | Path = "world",
    ) -> None:
        self.buildings = list(buildings)
        self.bounds = bounds
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _extrude_building(self, building: BuildingFootprint) -> trimesh.Trimesh:
        """Extrude footprint to create walls and roof."""
        return trimesh.creation.extrude_polygon(building.polygon, height=building.height_m)

    def _create_ground(self, thickness: float = 0.5) -> trimesh.Trimesh:
        """Create ground plane mesh."""
        cx, cy = self.bounds.center
        return trimesh.creation.box(
            extents=[self.bounds.width, self.bounds.height, thickness],
            transform=trimesh.transformations.translation_matrix(
                [cx, cy, -thickness / 2.0]
            ),
        )

    def generate(self, export: bool = True) -> CityMesh:
        """Generate combined city mesh and optionally export OBJ files."""
        meshes: list[trimesh.Trimesh] = []
        ids: list[int] = []

        for building in self.buildings:
            try:
                mesh = self._extrude_building(building)
                meshes.append(mesh)
                ids.append(building.building_id)
            except Exception as exc:
                logger.warning("Skipping building %s: %s", building.building_id, exc)

        combined = (
            trimesh.util.concatenate(meshes) if meshes else trimesh.Trimesh()
        )
        ground = self._create_ground()
        city = CityMesh(
            buildings=combined,
            ground=ground,
            building_ids=ids,
            bounds=self.bounds,
        )

        if export:
            self.export(city)
        return city

    def export(self, city: CityMesh) -> None:
        """Export city meshes to OBJ."""
        buildings_path = self.output_dir / "buildings.obj"
        ground_path = self.output_dir / "ground.obj"
        city.buildings.export(buildings_path)
        city.ground.export(ground_path)
        logger.info("Exported city meshes to %s", self.output_dir)

    def individual_meshes(self) -> list[trimesh.Trimesh]:
        """Return per-building meshes for physics import."""
        result: list[trimesh.Trimesh] = []
        for building in self.buildings:
            try:
                result.append(self._extrude_building(building))
            except Exception:
                continue
        return result
