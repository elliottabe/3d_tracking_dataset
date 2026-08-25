/**
 * Rect model for the layout editor.
 *
 * A Rect is [x, y, w, h] in ROOT-FIGURE FRACTIONS with y growing UP from the
 * bottom-left, matching matplotlib's `fig.add_axes`. SVG has y growing DOWN,
 * so `rectToScreen` is the ONE place that flip happens. Do not flip elsewhere.
 */
export type Rect = { x: number; y: number; w: number; h: number };

export const MM_PER_INCH = 25.4;
export const PT_PER_MM = 72 / MM_PER_INCH;

export function fracToMm(r: Rect, figWmm: number, figHmm: number): Rect {
  return { x: r.x * figWmm, y: r.y * figHmm, w: r.w * figWmm, h: r.h * figHmm };
}

export function mmToFrac(r: Rect, figWmm: number, figHmm: number): Rect {
  return { x: r.x / figWmm, y: r.y / figHmm, w: r.w / figWmm, h: r.h / figHmm };
}

/** Figure-space rect -> SVG screen box. The only y-flip in the codebase. */
export function rectToScreen(
  r: Rect, figWpt: number, figHpt: number,
): { x: number; y: number; w: number; h: number } {
  return {
    x: r.x * figWpt,
    y: (1 - (r.y + r.h)) * figHpt,
    w: r.w * figWpt,
    h: r.h * figHpt,
  };
}

export function edges(r: Rect) {
  return {
    left: r.x,
    right: r.x + r.w,
    bottom: r.y,
    top: r.y + r.h,
    cx: r.x + r.w / 2,
    cy: r.y + r.h / 2,
  };
}

export function unionRect(rects: Rect[]): Rect | null {
  if (rects.length === 0) return null;
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const r of rects) {
    x0 = Math.min(x0, r.x);
    y0 = Math.min(y0, r.y);
    x1 = Math.max(x1, r.x + r.w);
    y1 = Math.max(y1, r.y + r.h);
  }
  return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
}

/** Strict overlap: shared area only. Touching edges do NOT count. */
export function rectsIntersect(a: Rect, b: Rect, eps = 1e-9): boolean {
  return (
    a.x + a.w > b.x + eps &&
    b.x + b.w > a.x + eps &&
    a.y + a.h > b.y + eps &&
    b.y + b.h > a.y + eps
  );
}

/** A drag can produce negative w/h; canonicalise to positive extents. */
export function normalizeRect(r: Rect): Rect {
  return {
    x: r.w < 0 ? r.x + r.w : r.x,
    y: r.h < 0 ? r.y + r.h : r.y,
    w: Math.abs(r.w),
    h: Math.abs(r.h),
  };
}
