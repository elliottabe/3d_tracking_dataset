/**
 * Ink-box collision detection.
 *
 * The exporter's `overflows` flag tests only CANVAS bounds, so two panels
 * whose ink boxes collide — a neighbour's axis label drawn inside another
 * panel — were reported as fine. That was found by looking at a render, not
 * by a test. This module closes it using the `ink_box` every tile already
 * returns from `POST /api/panel`.
 */
import { rectsIntersect, type Rect } from './rect';

export type InkEntry = { id: string; ink: Rect };
export type Collision = { a: string; b: string; area: number };

function overlapArea(a: Rect, b: Rect): number {
  const w = Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x);
  const h = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y);
  return w > 0 && h > 0 ? w * h : 0;
}

/** Every colliding pair, once each, worst first. */
export function findCollisions(entries: InkEntry[]): Collision[] {
  const out: Collision[] = [];
  for (let i = 0; i < entries.length; i += 1) {
    for (let j = i + 1; j < entries.length; j += 1) {
      const a = entries[i];
      const b = entries[j];
      if (rectsIntersect(a.ink, b.ink)) {
        out.push({ a: a.id, b: b.id, area: overlapArea(a.ink, b.ink) });
      }
    }
  }
  return out.sort((p, q) => q.area - p.area);
}

/** Panels whose ink leaves the canvas — the same condition the exporter warns on. */
export function outOfBounds(entries: InkEntry[], eps = 1e-6): string[] {
  return entries
    .filter(({ ink }) =>
      ink.x < -eps || ink.y < -eps ||
      ink.x + ink.w > 1 + eps || ink.y + ink.h > 1 + eps)
    .map((e) => e.id);
}
