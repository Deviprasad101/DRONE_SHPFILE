import { useCallback, useEffect, useRef, useState } from "react";

const FLIGHT_DURATION_MS = 16000;

function pathLength(path: number[][]): number {
  let total = 0;
  for (let i = 0; i < path.length - 1; i++) {
    const dx = path[i + 1][0] - path[i][0];
    const dy = path[i + 1][1] - path[i][1];
    const dz = (path[i + 1][2] ?? 0) - (path[i][2] ?? 0);
    total += Math.sqrt(dx * dx + dy * dy + dz * dz);
  }
  return total;
}

function positionAlongPath(path: number[][], dist: number): {
  position: number[];
  stepIndex: number;
} {
  let acc = 0;
  for (let i = 0; i < path.length - 1; i++) {
    const a = path[i];
    const b = path[i + 1];
    const dx = b[0] - a[0];
    const dy = b[1] - a[1];
    const dz = (b[2] ?? 0) - (a[2] ?? 0);
    const seg = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (acc + seg >= dist) {
      const t = seg > 0 ? (dist - acc) / seg : 0;
      return {
        position: [
          a[0] + dx * t,
          a[1] + dy * t,
          (a[2] ?? 80) + dz * t,
        ],
        stepIndex: i,
      };
    }
    acc += seg;
  }
  const last = path[path.length - 1];
  return { position: last, stepIndex: path.length - 1 };
}

export function useDroneAnimation(
  path: number[][] | null,
  speed: number,
  playing: boolean,
  playId: number,
  onComplete?: () => void
) {
  const [position, setPosition] = useState<number[] | null>(null);
  const [stepIndex, setStepIndex] = useState(0);
  const [finished, setFinished] = useState(false);
  const frameRef = useRef<number>();
  const onCompleteRef = useRef(onComplete);
  onCompleteRef.current = onComplete;

  const reset = useCallback(() => {
    if (path && path.length > 0) {
      setPosition([...path[0]]);
      setStepIndex(0);
      setFinished(false);
    }
  }, [path]);

  // Initialise drone at path start when flight data loads
  useEffect(() => {
    reset();
  }, [path, reset]);

  // Run animation whenever playId increments or playing becomes true
  useEffect(() => {
    if (!playing || !path || path.length < 2) {
      if (frameRef.current) cancelAnimationFrame(frameRef.current);
      return;
    }

    if (frameRef.current) cancelAnimationFrame(frameRef.current);

    const total = pathLength(path);
    const durationMs = FLIGHT_DURATION_MS / speed;
    const startTime = performance.now();

    setFinished(false);
    setPosition([...path[0]]);
    setStepIndex(0);

    const tick = (now: number) => {
      const elapsed = now - startTime;
      const t = Math.min(1, elapsed / durationMs);
      const dist = t * total;
      const { position: pos, stepIndex: step } = positionAlongPath(path, dist);

      setPosition(pos);
      setStepIndex(step);

      if (t >= 1) {
        setFinished(true);
        onCompleteRef.current?.();
        return;
      }
      frameRef.current = requestAnimationFrame(tick);
    };

    frameRef.current = requestAnimationFrame(tick);

    return () => {
      if (frameRef.current) cancelAnimationFrame(frameRef.current);
    };
  }, [playing, path, speed, playId]);

  return { position, stepIndex, finished, reset };
}
