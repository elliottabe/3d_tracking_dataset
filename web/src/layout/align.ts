/**
 * Align / distribute / match-size, as pure geometry.
 *
 * Callers pass rects ALREADY resolved to the active box mode (axes or ink);
 * these functions know nothing about panels. Alignment never changes a rect's
 * size; match-size never changes its position. All coordinates are figure
 * fractions with y growing UP, so "top" is max(y + h).
 */
import { edges, unionRect, type Rect } from './rect';

export type AlignOp = 'left' | 'hcenter' | 'right' | 'top' | 'vcenter' | 'bottom';
export type RefMode = 'selection' | 'first' | 'figure';

const FIGURE: Rect = { x: 0, y: 0, w: 1, h: 1 };

function reference(rects: Rect[], ref: RefMode): Rect {
  if (ref === 'figure') return FIGURE;
  if (ref === 'first') return rects[0];
  return unionRect(rects) ?? FIGURE;
}

export function alignRects(rects: Rect[], op: AlignOp, ref: RefMode): Rect[] {
  if (rects.length === 0) return rects;
  const R = edges(reference(rects, ref));
  return rects.map((r) => {
    const e = edges(r);
    switch (op) {
      case 'left':    return { ...r, x: R.left };
      case 'right':   return { ...r, x: R.right - r.w };
      case 'hcenter': return { ...r, x: R.cx - r.w / 2 };
      case 'bottom':  return { ...r, y: R.bottom };
      case 'top':     return { ...r, y: R.top - r.h };
      case 'vcenter': return { ...r, y: R.cy - r.h / 2 };
      default:        return { ...r, x: e.left };
    }
  });
}

export function distributeRects(
  rects: Rect[], axis: 'x' | 'y', mode: 'gaps' | 'centers',
): Rect[] {
  if (rects.length < 3) return rects;
  const size = axis === 'x' ? ('w' as const) : ('h' as const);

  // Work in POSITION order, then scatter back to the caller's ordering.
  const order = rects.map((_, i) => i).sort((a, b) => rects[a][axis] - rects[b][axis]);
  const sorted = order.map((i) => rects[i]);
  const out = sorted.map((r) => ({ ...r }));
  const first = sorted[0];
  const last = sorted[sorted.length - 1];

  if (mode === 'gaps') {
    const span = (last[axis] + last[size]) - first[axis];
    const used = sorted.reduce((s, r) => s + r[size], 0);
    const gap = (span - used) / (sorted.length - 1);
    let cursor = first[axis];
    for (let i = 0; i < out.length; i += 1) {
      out[i][axis] = cursor;
      cursor += out[i][size] + gap;
    }
  } else {
    const c0 = first[axis] + first[size] / 2;
    const c1 = last[axis] + last[size] / 2;
    const step = (c1 - c0) / (sorted.length - 1);
    for (let i = 0; i < out.length; i += 1) {
      out[i][axis] = c0 + step * i - out[i][size] / 2;
    }
  }

  const result = rects.map((r) => ({ ...r }));
  order.forEach((origIdx, k) => { result[origIdx] = out[k]; });
  return result;
}

export function matchSize(rects: Rect[], dim: 'w' | 'h'): Rect[] {
  if (rects.length === 0) return rects;
  const target = rects[0][dim];
  return rects.map((r) => ({ ...r, [dim]: target }));
}
