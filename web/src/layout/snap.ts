/**
 * Snapping for panel drags.
 *
 * Snapping MOVES a rect; it never resizes it. Tolerance is supplied in
 * FIGURE FRACTIONS but is derived from a screen-pixel budget (see
 * `screenTolToFrac`) so the feel stays constant as the user zooms.
 */
import { edges, type Rect } from './rect';

export type BoxMode = 'axes' | 'ink';

export type SnapKind = 'margin' | 'edge' | 'center' | 'grid' | 'guide';
export type SnapTarget = { axis: 'x' | 'y'; value: number; kind: SnapKind };

export type BuildTargetsOpts = {
  /** Rects of the panels NOT being dragged, in the active box mode. */
  others: Rect[];
  /** Grid spacing in mm; 0 disables the grid. */
  gridMm: number;
  figWmm: number;
  figHmm: number;
  /** User-placed ruler guides, in figure fractions. */
  guides: SnapTarget[];
};

export function buildTargets(o: BuildTargetsOpts): SnapTarget[] {
  const out: SnapTarget[] = [
    { axis: 'x', value: 0, kind: 'margin' },
    { axis: 'x', value: 1, kind: 'margin' },
    { axis: 'y', value: 0, kind: 'margin' },
    { axis: 'y', value: 1, kind: 'margin' },
  ];

  for (const r of o.others) {
    const e = edges(r);
    out.push({ axis: 'x', value: e.left, kind: 'edge' });
    out.push({ axis: 'x', value: e.right, kind: 'edge' });
    out.push({ axis: 'x', value: e.cx, kind: 'center' });
    out.push({ axis: 'y', value: e.bottom, kind: 'edge' });
    out.push({ axis: 'y', value: e.top, kind: 'edge' });
    out.push({ axis: 'y', value: e.cy, kind: 'center' });
  }

  if (o.gridMm > 0) {
    for (let mm = 0; mm <= o.figWmm; mm += o.gridMm) {
      out.push({ axis: 'x', value: mm / o.figWmm, kind: 'grid' });
    }
    for (let mm = 0; mm <= o.figHmm; mm += o.gridMm) {
      out.push({ axis: 'y', value: mm / o.figHmm, kind: 'grid' });
    }
  }

  out.push(...o.guides);
  return out;
}

/** Convert a screen-pixel tolerance into figure fractions at the current zoom. */
export function screenTolToFrac(tolPx: number, figExtentPt: number, zoom: number): number {
  return tolPx / (figExtentPt * zoom);
}

type Best = { delta: number; target: SnapTarget } | null;

function bestFor(candidates: number[], targets: SnapTarget[], axis: 'x' | 'y', tol: number): Best {
  let best: Best = null;
  for (const t of targets) {
    if (t.axis !== axis) continue;
    for (const c of candidates) {
      const d = t.value - c;
      if (Math.abs(d) <= tol && (best === null || Math.abs(d) < Math.abs(best.delta))) {
        best = { delta: d, target: t };
      }
    }
  }
  return best;
}

/**
 * Snap `moving` to the nearest target on each axis independently.
 * Returns the shifted rect and the targets that produced a snap, so the
 * caller can draw a guide line for each.
 */
export function snapRect(
  moving: Rect, targets: SnapTarget[], tolFrac: number,
): { rect: Rect; guides: SnapTarget[] } {
  const e = edges(moving);
  const gx = bestFor([e.left, e.right, e.cx], targets, 'x', tolFrac);
  const gy = bestFor([e.bottom, e.top, e.cy], targets, 'y', tolFrac);

  const guides: SnapTarget[] = [];
  if (gx) guides.push(gx.target);
  if (gy) guides.push(gy.target);

  return {
    rect: {
      x: moving.x + (gx ? gx.delta : 0),
      y: moving.y + (gy ? gy.delta : 0),
      w: moving.w,
      h: moving.h,
    },
    guides,
  };
}
