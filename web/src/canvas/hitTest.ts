/** Pointer hit-testing in FIGURE space (y-up). Later panels are on top. */
import { normalizeRect, rectsIntersect, type Rect } from '../layout/rect';

export type Hittable = { id: string; rect: Rect };

export function hitTestPanels(panels: Hittable[], pt: { x: number; y: number }): string | null {
  for (let i = panels.length - 1; i >= 0; i -= 1) {
    const r = panels[i].rect;
    if (pt.x >= r.x && pt.x <= r.x + r.w && pt.y >= r.y && pt.y <= r.y + r.h) {
      return panels[i].id;
    }
  }
  return null;
}

export function marqueeHits(panels: Hittable[], marquee: Rect): string[] {
  const m = normalizeRect(marquee);
  return panels.filter((p) => rectsIntersect(p.rect, m)).map((p) => p.id);
}

/**
 * Inverse of `rectToScreen`: client px -> figure fraction (y-up).
 *
 * `rectToScreen` is forward-only (figure fraction -> viewBox pt, with the
 * one y-flip in the codebase). The <svg> is rendered with
 * width=figWpt*zoom, height=figHpt*zoom and viewBox="0 0 figWpt figHpt", so
 * its live `getBoundingClientRect()` (`svgRect`) already encodes both pan
 * (left/top) and zoom (width/height) — dividing the client-space delta by
 * `zoom` recovers the viewBox pt, which we then flip back (y-up) and
 * normalise by the figure extents to land in the same fraction space
 * `rectToScreen` started from.
 */
export function screenToFrac(
  clientX: number,
  clientY: number,
  svgRect: { left: number; top: number },
  figWpt: number,
  figHpt: number,
  zoom: number,
): { x: number; y: number } {
  const xpt = (clientX - svgRect.left) / zoom;
  const ypt = (clientY - svgRect.top) / zoom;
  return { x: xpt / figWpt, y: 1 - ypt / figHpt };
}
