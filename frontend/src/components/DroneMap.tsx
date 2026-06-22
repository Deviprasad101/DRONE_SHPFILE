import { useMemo } from "react";
import Map, { NavigationControl, ScaleControl, useControl } from "react-map-gl/maplibre";
import { MapboxOverlay } from "@deck.gl/mapbox";
import { GeoJsonLayer, PathLayer, ScatterplotLayer } from "@deck.gl/layers";
import { ScenegraphLayer } from "@deck.gl/mesh-layers";
import { COORDINATE_SYSTEM } from "@deck.gl/core";
import type { Layer } from "@deck.gl/core";
import type { BuildingCollection } from "../types/geo";
import "maplibre-gl/dist/maplibre-gl.css";

interface ViewState {
  longitude: number;
  latitude: number;
  zoom: number;
  pitch: number;
  bearing: number;
}

interface DroneMapProps {
  buildings: BuildingCollection | null;
  plannedPath: number[][] | null;
  trajectory: number[][] | null;
  dronePosition: number[] | null;
  start: number[] | null;
  goal: number[] | null;
  viewState: ViewState;
  onMove: (vs: ViewState) => void;
  placementMode?: "start" | "goal" | null;
  onMapClick?: (lon: number, lat: number) => void;
}

function DeckGLOverlay({ layers }: { layers: Layer[] }) {
  const overlay = useControl<MapboxOverlay>(
    () => new MapboxOverlay({ interleaved: true, layers }),
    () => {},
    () => {},
    { position: "top-left" }
  );
  // Update layers synchronously on every render — keeps deck.gl in the same
  // MapLibre GL render frame so buildings never "float" during pan/drag.
  overlay.setProps({ layers });
  return null;
}

const LNGLAT = COORDINATE_SYSTEM.LNGLAT;
const BASE_MAP_STYLE = "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json";

function useBuildingLayers(buildings: BuildingCollection | null): Layer[] {
  return useMemo(() => {
    if (!buildings?.features?.length) return [];
    return [
      new GeoJsonLayer({
        id: "buildings-3d",
        data: buildings as GeoJSON.FeatureCollection,
        coordinateSystem: LNGLAT,
        extruded: true,
        wireframe: false,
        opacity: 0.92,
        material: {
          ambient: 0.45,
          diffuse: 0.65,
          shininess: 24,
          specularColor: [180, 180, 200],
        },
        getFillColor: (f: GeoJSON.Feature) => {
          const h = (f.properties as { height_m?: number })?.height_m ?? 30;
          const t = Math.min(1, h / 60);
          return [
            Math.round(70 + t * 40),
            Math.round(95 + t * 35),
            Math.round(125 + t * 30),
            220,
          ];
        },
        getLineColor: [40, 50, 65, 200],
        getElevation: (f: GeoJSON.Feature) =>
          (f.properties as { height_m?: number })?.height_m ?? 30,
        elevationScale: 1,
        pickable: false,
      }),
    ];
  }, [buildings]);
}

function useFlightLayers(
  plannedPath: number[][] | null,
  trajectory: number[][] | null,
  dronePosition: number[] | null,
  start: number[] | null,
  goal: number[] | null
): Layer[] {
  return useMemo(() => {
    const result: Layer[] = [];

    if (plannedPath && plannedPath.length > 1) {
      result.push(
        new PathLayer({
          id: "planned-path",
          data: [{ path: plannedPath }],
          coordinateSystem: LNGLAT,
          getPath: (d: { path: number[][] }) => d.path as [number, number, number][],
          getColor: [245, 158, 11, 255],
          getWidth: 8,
          widthMinPixels: 4,
          capRounded: true,
          jointRounded: true,
          billboard: false,
          parameters: { depthTest: false },
        })
      );
    }

    if (trajectory && trajectory.length > 1) {
      result.push(
        new PathLayer({
          id: "trajectory",
          data: [{ path: trajectory }],
          coordinateSystem: LNGLAT,
          getPath: (d: { path: number[][] }) => d.path as [number, number, number][],
          getColor: [14, 165, 233, 200],
          getWidth: 5,
          widthMinPixels: 2,
          capRounded: true,
          jointRounded: true,
          parameters: { depthTest: false },
        })
      );
    }

    if (start) {
      result.push(
        new ScatterplotLayer({
          id: "start-marker",
          data: [{ position: [start[0], start[1], start[2] ?? 80] as [number, number, number] }],
          coordinateSystem: LNGLAT,
          getPosition: (d) => d.position,
          getFillColor: [37, 99, 235, 255],
          getRadius: 30,
          radiusMinPixels: 10,
          radiusMaxPixels: 18,
          stroked: true,
          getLineColor: [255, 255, 255, 230],
          lineWidthMinPixels: 2,
          parameters: { depthTest: false },
        })
      );
    }

    if (goal) {
      result.push(
        new ScatterplotLayer({
          id: "goal-marker",
          data: [{ position: [goal[0], goal[1], goal[2] ?? 80] as [number, number, number] }],
          coordinateSystem: LNGLAT,
          getPosition: (d) => d.position,
          getFillColor: [22, 163, 74, 255],
          getRadius: 30,
          radiusMinPixels: 10,
          radiusMaxPixels: 18,
          stroked: true,
          getLineColor: [255, 255, 255, 230],
          lineWidthMinPixels: 2,
          parameters: { depthTest: false },
        })
      );
    }

    if (dronePosition) {
      const alt = (dronePosition[2] ?? 80) + 25;
      const pos: [number, number, number] = [
        dronePosition[0],
        dronePosition[1],
        alt,
      ];

      // 3D drone model — flat/horizontal orientation
      result.push(
        new ScenegraphLayer({
          id: "drone-3d",
          data: [{ position: pos }],
          coordinateSystem: LNGLAT,
          scenegraph: "/drone.glb?v=3",
          getPosition: d => d.position,
          getOrientation: _d => [0, 0, 0],
          sizeScale: 40,
          _lighting: "pbr",
          getColor: [255, 255, 255, 255],
          pickable: false,
          parameters: { depthTest: false },
        })
      );
    }

    return result;
  }, [plannedPath, trajectory, dronePosition, start, goal]);
}

export default function DroneMap({
  buildings,
  plannedPath,
  trajectory,
  dronePosition,
  start,
  goal,
  viewState,
  onMove,
  placementMode = null,
  onMapClick,
}: DroneMapProps) {
  const buildingLayers = useBuildingLayers(buildings);
  const flightLayers = useFlightLayers(
    plannedPath,
    trajectory,
    dronePosition,
    start,
    goal
  );
  const layers = useMemo(
    () => [...buildingLayers, ...flightLayers],
    [buildingLayers, flightLayers]
  );

  const cursor = placementMode ? "crosshair" : "grab";

  return (
    <Map
      {...viewState}
      onMove={(e) =>
        onMove({
          longitude: e.viewState.longitude,
          latitude: e.viewState.latitude,
          zoom: e.viewState.zoom,
          pitch: e.viewState.pitch,
          bearing: e.viewState.bearing,
        })
      }
      onClick={(e) => {
        if (placementMode && onMapClick) {
          onMapClick(e.lngLat.lng, e.lngLat.lat);
        }
      }}
      mapStyle={BASE_MAP_STYLE}
      style={{ width: "100%", height: "100%", cursor }}
      attributionControl={true}
      maxPitch={85}
      dragPan={!placementMode}
    >
      <NavigationControl position="top-right" visualizePitch />
      <ScaleControl position="bottom-left" />
      <DeckGLOverlay layers={layers} />
    </Map>
  );
}
