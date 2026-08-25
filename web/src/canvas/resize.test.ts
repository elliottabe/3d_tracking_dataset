import { describe, expect, it } from 'vitest';
import { resizeRect } from './resize';
import type { Rect } from '../layout/rect';

const R: Rect = { x: 0.2, y: 0.2, w: 0.4, h: 0.4 };

describe('resizeRect', () => {
  it('e drags the right edge, keeping x', () => {
    const out = resizeRect(R, 'e', 0.1, 0);
    expect(out.x).toBeCloseTo(0.2);
    expect(out.w).toBeCloseTo(0.5);
    expect(out.h).toBeCloseTo(0.4);
  });

  it('w drags the left edge, moving x and shrinking w', () => {
    const out = resizeRect(R, 'w', 0.1, 0);
    expect(out.x).toBeCloseTo(0.3);
    expect(out.w).toBeCloseTo(0.3);
  });

  it('n grows upward in FIGURE space (y-up)', () => {
    const out = resizeRect(R, 'n', 0, 0.1);
    expect(out.y).toBeCloseTo(0.2);
    expect(out.h).toBeCloseTo(0.5);
  });

  it('s drags the bottom edge, moving y', () => {
    const out = resizeRect(R, 's', 0, 0.1);
    expect(out.y).toBeCloseTo(0.3);
    expect(out.h).toBeCloseTo(0.3);
  });

  it('ne moves both the right and top edges', () => {
    const out = resizeRect(R, 'ne', 0.1, 0.1);
    expect(out.w).toBeCloseTo(0.5);
    expect(out.h).toBeCloseTo(0.5);
    expect(out.x).toBeCloseTo(0.2);
    expect(out.y).toBeCloseTo(0.2);
  });

  // Hand-derived: 'e' step gives w = 0.4 - 0.6 = -0.2 (x, y, h untouched by
  // the 'e' step). normalizeRect then flips: since w < 0, x = 0.2 + (-0.2)
  // = 0.0 and w = abs(-0.2) = 0.2 — the right edge lands where the left
  // edge used to be, height untouched. A "clamp instead of flip" bug
  // (w = max(0, r.w+dx), x untouched) would instead produce {x:0.2, w:0}
  // and must FAIL these exact assertions.
  it('normalizes a rect dragged inside-out rather than emitting negative w', () => {
    const out = resizeRect(R, 'e', -0.6, 0);
    expect(out.x).toBeCloseTo(0.0);
    expect(out.y).toBeCloseTo(0.2);
    expect(out.w).toBeCloseTo(0.2);
    expect(out.h).toBeCloseTo(0.4);
  });

  // Mirrored vertical case. 's' step: y += dy = 0.2+0.6 = 0.8, h -= dy =
  // 0.4-0.6 = -0.2 (top edge y+h = 0.6 stays fixed by the step itself).
  // normalizeRect flips on h < 0: y = 0.8 + (-0.2) = 0.6, h = abs(-0.2) =
  // 0.2 — the south edge lands where the north edge used to be (0.6).
  it('normalizes a rect dragged inside-out vertically (s overshoot)', () => {
    const out = resizeRect(R, 's', 0, 0.6);
    expect(out.x).toBeCloseTo(0.2);
    expect(out.y).toBeCloseTo(0.6);
    expect(out.w).toBeCloseTo(0.4);
    expect(out.h).toBeCloseTo(0.2);
  });

  it('honours a minimum size so a panel cannot collapse to nothing', () => {
    const out = resizeRect(R, 'e', -0.4, 0, { minW: 0.05, minH: 0.05 });
    expect(out.w).toBeCloseTo(0.05);
  });

  it('preserves aspect when asked', () => {
    const out = resizeRect(R, 'se', 0.2, 0, { aspect: true });
    expect(out.w / out.h).toBeCloseTo(R.w / R.h, 6);
  });

  // 's' step: y=0.3, h=0.3. Height is the driver for a pure-vertical
  // handle: newW = out.h * ratio = 0.3*1 = 0.3, centred on cx = 0.2+0.2 =
  // 0.4, so x = 0.4 - 0.15 = 0.25. Before the fix this recomputed the
  // ORIGINAL h from out.w and was a bit-identical no-op.
  it('shift+drag on s changes the rect and preserves aspect ratio', () => {
    const out = resizeRect(R, 's', 0, 0.1, { aspect: true });
    expect(out.x).toBeCloseTo(0.25);
    expect(out.y).toBeCloseTo(0.3);
    expect(out.w).toBeCloseTo(0.3);
    expect(out.h).toBeCloseTo(0.3);
    expect(out.w / out.h).toBeCloseTo(R.w / R.h, 6);
  });

  // 'n' step: h=0.5, x/y untouched. newW = out.h*ratio = 0.5*1 = 0.5,
  // centred on cx = 0.2+0.2 = 0.4, so x = 0.4 - 0.25 = 0.15.
  it('shift+drag on n changes the rect and preserves aspect ratio', () => {
    const out = resizeRect(R, 'n', 0, 0.1, { aspect: true });
    expect(out.x).toBeCloseTo(0.15);
    expect(out.y).toBeCloseTo(0.2);
    expect(out.w).toBeCloseTo(0.5);
    expect(out.h).toBeCloseTo(0.5);
    expect(out.w / out.h).toBeCloseTo(R.w / R.h, 6);
  });
});
