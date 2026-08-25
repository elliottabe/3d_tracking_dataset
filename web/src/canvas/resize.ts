/**
 * Rect resizing from an edge or corner handle.
 *
 * All coordinates are figure fractions with y growing UP, so 'n' increases
 * height without moving y, and 's' moves y while shrinking height.
 */
import { normalizeRect, type Rect } from '../layout/rect';

export type HandleId = 'nw' | 'n' | 'ne' | 'e' | 'se' | 's' | 'sw' | 'w';

export type ResizeOpts = { minW?: number; minH?: number; aspect?: boolean };

export function resizeRect(
  r: Rect, handle: HandleId, dx: number, dy: number, opts: ResizeOpts = {},
): Rect {
  const minW = opts.minW ?? 0;
  const minH = opts.minH ?? 0;

  let { x, y, w, h } = r;
  if (handle.includes('e')) w += dx;
  if (handle.includes('w')) { x += dx; w -= dx; }
  if (handle.includes('n')) h += dy;
  if (handle.includes('s')) { y += dy; h -= dy; }

  let out = normalizeRect({ x, y, w, h });

  if (opts.aspect && r.h !== 0) {
    const ratio = r.w / r.h;
    // Drive height from width so a horizontal drag feels authoritative.
    const newH = out.w / ratio;
    if (handle.includes('s')) out = { ...out, y: out.y + (out.h - newH) };
    out = { ...out, h: newH };
  }

  if (out.w < minW) {
    if (handle.includes('w')) out = { ...out, x: out.x + (out.w - minW) };
    out = { ...out, w: minW };
  }
  if (out.h < minH) {
    if (handle.includes('s')) out = { ...out, y: out.y + (out.h - minH) };
    out = { ...out, h: minH };
  }
  return out;
}
