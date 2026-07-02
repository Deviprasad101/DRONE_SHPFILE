import { useMemo, useCallback, useState } from "react";
import Map, { NavigationControl, ScaleControl, useControl } from "react-map-gl/maplibre";
import { MapboxOverlay } from "@deck.gl/mapbox";
import { GeoJsonLayer, PathLayer, ScatterplotLayer } from "@deck.gl/layers";
import { ScenegraphLayer } from "@deck.gl/mesh-layers";
import { COORDINATE_SYSTEM } from "@deck.gl/core";
import type { Layer, PickingInfo } from "@deck.gl/core";
import type { BuildingCollection, FlightPath } from "../types/geo";
import { buildingElevationM, buildingFillColor } from "../utils/buildingHeightColor";
import { getBuildingTooltipData } from "../utils/buildingInfo";
import BuildingTooltip from "./BuildingTooltip";
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
  flights: FlightPath[] | null;
  selectedFlightIndex: number;
  dronePosition: number[] | null;
  start: number[] | null;
  goal: number[] | null;
  viewState: ViewState;
  onMove: (vs: ViewState) => void;
  placementMode?: "start" | "goal" | null;
  onMapClick?: (lon: number, lat: number) => void;
  useHeightColors?: boolean;
}

function DeckGLOverlay({
  layers,
  onHover,
  getCursor,
}: {
  layers: Layer[];
  onHover?: (info: PickingInfo) => void;
  getCursor?: (info: PickingInfo) => string;
}) {
  const overlay = useControl<MapboxOverlay>(
    () => new MapboxOverlay({ interleaved: true, layers }),
    () => {},
    () => {},
    { position: "top-left" }
  );
  overlay.setProps({ layers, onHover, getCursor });
  return null;
}

const LNGLAT = COORDINATE_SYSTEM.LNGLAT;
const BASE_MAP_STYLE = "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json";

const PLANNED_COLORS = [
  [245, 158, 11, 255],  // Amber
  [236, 72, 153, 255],  // Pink
  [168, 85, 247, 255],  // Purple
  [34, 197, 94, 255],   // Green
];

const TRAJ_COLORS = [
  [14, 165, 233, 200],  // Sky Blue
  [244, 63, 94, 200],   // Rose
  [139, 92, 246, 200],  // Violet
  [16, 185, 129, 200],  // Emerald
];

function useBuildingLayers(
  buildings: BuildingCollection | null,
  useHeightColors: boolean
): Layer[] {
  const featureCount = buildings?.features.length ?? 0;

  return useMemo(() => {
    if (!buildings?.features?.length) return [];
    const collection: GeoJSON.FeatureCollection = {
      type: "FeatureCollection",
      features: buildings.features,
    };
    return [
      new GeoJsonLayer({
        id: "buildings-3d",
        data: collection,
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
        getFillColor: (f: GeoJSON.Feature) =>
          buildingFillColor(
            f.properties as Record<string, unknown> | null,
            useHeightColors
          ),
        getLineColor: [40, 50, 65, 200],
        getElevation: (f: GeoJSON.Feature) =>
          buildingElevationM(f.properties as Record<string, unknown> | null),
        elevationScale: 1,
        pickable: true,
        autoHighlight: true,
        highlightColor: [255, 255, 255, 80],
        updateTriggers: {
          data: featureCount,
          getFillColor: [featureCount, useHeightColors],
          getElevation: featureCount,
        },
      }),
    ];
  }, [buildings, featureCount, useHeightColors]);
}

function useFlightLayers(
  flights: FlightPath[] | null,
  selectedFlightIndex: number,
  dronePosition: number[] | null,
  start: number[] | null,
  goal: number[] | null
): Layer[] {
  return useMemo(() => {
    const result: Layer[] = [];

    if (flights) {
      flights.forEach((flight, index) => {
        const isSelected = index === selectedFlightIndex;
        const plannedPath = flight.planned_path;
        const trajectory = flight.trajectory;

        const pColor = PLANNED_COLORS[index % PLANNED_COLORS.length];
        const tColor = TRAJ_COLORS[index % TRAJ_COLORS.length];

        if (plannedPath && plannedPath.length > 1) {
          result.push(
            new PathLayer({
              id: `planned-path-${index}`,
              data: [{ path: plannedPath }],
              coordinateSystem: LNGLAT,
              getPath: (d: { path: number[][] }) => d.path as [number, number, number][],
              getColor: (isSelected ? pColor : [...pColor.slice(0, 3), 100]) as any,
              getWidth: isSelected ? 8 : 4,
              widthMinPixels: isSelected ? 4 : 2,
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
              id: `trajectory-${index}`,
              data: [{ path: trajectory }],
              coordinateSystem: LNGLAT,
              getPath: (d: { path: number[][] }) => d.path as [number, number, number][],
              getColor: (isSelected ? tColor : [...tColor.slice(0, 3), 80]) as any,
              getWidth: isSelected ? 5 : 3,
              widthMinPixels: isSelected ? 2 : 1,
              capRounded: true,
              jointRounded: true,
              parameters: { depthTest: false },
            })
          );
        }
      });
    }

    if (start) {
      result.push(
        new ScatterplotLayer({
          id: "start-marker",
          data: [{ position: [start[0], start[1], start[2] ?? 80] as [number, number, number] }],
          coordinateSystem: LNGLAT,
          getPosition: (d) => d.position,
          getFillColor: [37, 99, 235, 255] as any,
          getRadius: 30,
          radiusMinPixels: 10,
          radiusMaxPixels: 18,
          stroked: true,
          getLineColor: [255, 255, 255, 230] as any,
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
          getFillColor: [22, 163, 74, 255] as any,
          getRadius: 30,
          radiusMinPixels: 10,
          radiusMaxPixels: 18,
          stroked: true,
          getLineColor: [255, 255, 255, 230] as any,
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
          getColor: [255, 255, 255, 255] as any,
          pickable: false,
          parameters: { depthTest: false },
        })
      );
    }

    return result;
  }, [flights, selectedFlightIndex, dronePosition, start, goal]);
}

export default function DroneMap({
  buildings,
  flights,
  selectedFlightIndex,
  dronePosition,
  start,
  goal,
  viewState,
  onMove,
  placementMode = null,
  onMapClick,
  useHeightColors = true,
}: DroneMapProps) {
  const [hover, setHover] = useState<{
    x: number;
    y: number;
    data: ReturnType<typeof getBuildingTooltipData>;
  } | null>(null);

  const buildingLayers = useBuildingLayers(buildings, useHeightColors);
  const flightLayers = useFlightLayers(
    flights,
    selectedFlightIndex,
    dronePosition,
    start,
    goal
  );
  const layers = useMemo(
    () => [...buildingLayers, ...flightLayers],
    [buildingLayers, flightLayers]
  );

  const handleHover = useCallback((info: PickingInfo) => {
    if (info.layer?.id === "buildings-3d" && info.object) {
      const feature = info.object as GeoJSON.Feature;
      setHover({
        x: info.x,
        y: info.y,
        data: getBuildingTooltipData(feature, info.index),
      });
    } else {
      setHover(null);
    }
  }, []);

  const getCursor = useCallback(
    (info: PickingInfo) => {
      if (placementMode) return "crosshair";
      if (info.layer?.id === "buildings-3d" && info.object) return "pointer";
      return "grab";
    },
    [placementMode]
  );

  const cursor = placementMode ? "crosshair" : hover ? "pointer" : "grab";

  return (
    <div className="drone-map-wrap">
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
      <DeckGLOverlay layers={layers} onHover={handleHover} getCursor={getCursor} />
    </Map>
      {hover && (
        <BuildingTooltip data={hover.data} x={hover.x} y={hover.y} />
      )}
    </div>
  );
}
