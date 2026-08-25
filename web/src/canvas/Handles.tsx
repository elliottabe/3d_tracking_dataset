/** Eight resize handles around the selection bbox, in SCREEN coordinates. */
import type { HandleId } from './resize';

const HANDLES: { id: HandleId; fx: number; fy: number }[] = [
  { id: 'nw', fx: 0, fy: 0 }, { id: 'n', fx: 0.5, fy: 0 }, { id: 'ne', fx: 1, fy: 0 },
  { id: 'w', fx: 0, fy: 0.5 }, { id: 'e', fx: 1, fy: 0.5 },
  { id: 'sw', fx: 0, fy: 1 }, { id: 's', fx: 0.5, fy: 1 }, { id: 'se', fx: 1, fy: 1 },
];

const CURSOR: Record<HandleId, string> = {
  nw: 'nwse-resize', se: 'nwse-resize', ne: 'nesw-resize', sw: 'nesw-resize',
  n: 'ns-resize', s: 'ns-resize', e: 'ew-resize', w: 'ew-resize',
};

export function Handles(
  { box, size = 6, onGrab }:
  { box: { x: number; y: number; w: number; h: number }; size?: number;
    onGrab: (h: HandleId, e: React.PointerEvent) => void },
) {
  return (
    <g id="handles">
      {HANDLES.map(({ id, fx, fy }) => (
        <rect
          key={id}
          x={box.x + fx * box.w - size / 2}
          y={box.y + fy * box.h - size / 2}
          width={size} height={size}
          fill="#ffffff" stroke="#2563eb" strokeWidth={1}
          style={{ cursor: CURSOR[id] }}
          onPointerDown={(e) => { e.stopPropagation(); onGrab(id, e); }}
        />
      ))}
    </g>
  );
}
