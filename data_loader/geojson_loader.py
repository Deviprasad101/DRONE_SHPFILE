"""GeoJSON building data loading and coordinate conversion."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
from pyproj import Transformer
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import transform as shapely_transform

logger = logging.getLogger(__name__)

HEIGHT_KEYS = ("height_m", "height", "HEIGHT", "building_height", "levels")
DEFAULT_HEIGHT_M = 30.0


@dataclass(frozen=True)
class BuildingFootprint:
    """Building footprint in local ENU coordinates (meters)."""

    building_id: int
    polygon: Polygon
    height_m: float

    @property
    def centroid(self) -> tuple[float, float]:
        c = self.polygon.centroid
        return float(c.x), float(c.y)


@dataclass(frozen=True)
class WorldBounds:
    """Axis-aligned world bounds in local meters."""

    min_x: float
    min_y: float
    max_x: float
    max_y: float
    margin: float = 50.0

    def with_margin(self, margin: float) -> WorldBounds:
        """Return copy with updated margin."""
        return WorldBounds(self.min_x, self.min_y, self.max_x, self.max_y, margin)

    @property
    def width(self) -> float:
        return self.max_x - self.min_x + 2 * self.margin

    @property
    def height(self) -> float:
        return self.max_y - self.min_y + 2 * self.margin

    @property
    def center(self) -> tuple[float, float]:
        return (self.min_x + self.max_x) / 2.0, (self.min_y + self.max_y) / 2.0

    def contains(self, x: float, y: float) -> bool:
        return (
            self.min_x - self.margin <= x <= self.max_x + self.margin
            and self.min_y - self.margin <= y <= self.max_y + self.margin
        )


@dataclass
class LocalFrame:
    """WGS84 to local ENU coordinate frame."""

    origin_lat: float
    origin_lon: float
    transformer_to_local: Transformer
    transformer_to_wgs84: Transformer

    def to_local(self, lon: float, lat: float) -> tuple[float, float]:
        x, y = self.transformer_to_local.transform(lon, lat)
        return float(x), float(y)

    def to_wgs84(self, x: float, y: float) -> tuple[float, float]:
        lon, lat = self.transformer_to_wgs84.transform(x, y)
        return float(lon), float(lat)


def _extract_height(row: Any, default_height: float) -> float:
    for key in HEIGHT_KEYS:
        if key in row.index and row[key] is not None:
            try:
                value = float(row[key])
                if value > 0:
                    return value
            except (TypeError, ValueError):
                continue
    return default_height


def _flatten_polygons(geom) -> list[Polygon]:
    """Extract valid Polygon parts from Polygon or MultiPolygon."""
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom.buffer(0) if not geom.is_valid else geom]
    if isinstance(geom, MultiPolygon):
        return [
            g.buffer(0) if not g.is_valid else g
            for g in geom.geoms
            if not g.is_empty
        ]
    return []


def load_geojson(path: str | Path) -> gpd.GeoDataFrame:
    """Load building GeoJSON as a GeoDataFrame in WGS84."""
    path = Path(path)
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")
    logger.info("Loaded %d features from %s", len(gdf), path)
    return gdf


def extract_buildings(
    gdf: gpd.GeoDataFrame,
    default_height_m: float = DEFAULT_HEIGHT_M,
    max_buildings: int | None = None,
) -> list[tuple[int, Polygon, float]]:
    """Extract building polygons and heights from GeoDataFrame."""
    records: list[tuple[int, Polygon, float]] = []
    for idx, row in gdf.iterrows():
        if max_buildings is not None and len(records) >= max_buildings:
            break
        building_id = int(row.get("building_id", idx) if "building_id" in row.index else idx)
        height = _extract_height(row, default_height_m)
        for poly in _flatten_polygons(row.geometry):
            records.append((building_id, poly, height))
    logger.info("Extracted %d building footprints", len(records))
    return records


def convert_to_local_frame(
    buildings_wgs84: list[tuple[int, Polygon, float]],
    origin_lat: float | None = None,
    origin_lon: float | None = None,
) -> tuple[list[BuildingFootprint], LocalFrame, WorldBounds]:
    """Convert WGS84 footprints to a local ENU meter frame using PyProj."""
    if not buildings_wgs84:
        raise ValueError("No buildings to convert")

    all_coords: list[tuple[float, float]] = []
    for _, poly, _ in buildings_wgs84:
        all_coords.extend(poly.exterior.coords)

    lons = [c[0] for c in all_coords]
    lats = [c[1] for c in all_coords]
    if origin_lat is None:
        origin_lat = float(np.mean(lats))
    if origin_lon is None:
        origin_lon = float(np.mean(lons))

    # Azimuthal equidistant projection centered on the map origin (ENU-like)
    proj_string = (
        f"+proj=aeqd +lat_0={origin_lat} +lon_0={origin_lon} "
        f"+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )
    to_local = Transformer.from_crs("EPSG:4326", proj_string, always_xy=True)
    to_wgs84 = Transformer.from_crs(proj_string, "EPSG:4326", always_xy=True)

    frame = LocalFrame(
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        transformer_to_local=to_local,
        transformer_to_wgs84=to_wgs84,
    )

    def _project(x: float, y: float, z: float | None = None) -> tuple[float, float]:
        px, py = to_local.transform(x, y)
        return float(px), float(py)

    local_buildings: list[BuildingFootprint] = []
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")

    for building_id, poly_wgs84, height in buildings_wgs84:
        poly_local = shapely_transform(_project, poly_wgs84)
        minx, miny, maxx, maxy = poly_local.bounds
        min_x, min_y = min(min_x, minx), min(min_y, miny)
        max_x, max_y = max(max_x, maxx), max(max_y, maxy)
        local_buildings.append(
            BuildingFootprint(building_id=building_id, polygon=poly_local, height_m=height)
        )

    bounds = WorldBounds(min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y)
    return local_buildings, frame, bounds


def load_and_convert(
    path: str | Path,
    default_height_m: float = DEFAULT_HEIGHT_M,
    max_buildings: int | None = None,
    origin_lat: float | None = None,
    origin_lon: float | None = None,
) -> tuple[list[BuildingFootprint], LocalFrame, WorldBounds]:
    """End-to-end loader: GeoJSON -> local ENU building footprints."""
    gdf = load_geojson(path)
    raw = extract_buildings(gdf, default_height_m, max_buildings)
    return convert_to_local_frame(raw, origin_lat, origin_lon)
