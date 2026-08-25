// web/src/layout/align.test.ts
import { describe, expect, it } from 'vitest';
import { alignRects, distributeRects, matchSize } from './align';
import type { Rect } from './rect';

const R = (x: number, y: number, w: number, h: number): Rect => ({ x, y, w, h });

describe('alignRects', () => {
  const rects = [R(0.1, 0.1, 0.2, 0.1), R(0.5, 0.4, 0.3, 0.2)];

  it('aligns left to the selection bbox', () => {
    const out = alignRects(rects, 'left', 'selection');
    expect(out[0].x).toBeCloseTo(0.1);
    expect(out[1].x).toBeCloseTo(0.1);
  });

  it('aligns right, preserving each width', () => {
    const out = alignRects(rects, 'right', 'selection');
    expect(out[0].x + out[0].w).toBeCloseTo(0.8);
    expect(out[1].x + out[1].w).toBeCloseTo(0.8);
    expect(out[0].w).toBeCloseTo(0.2);
  });

  it('aligns to the FIRST rect when ref is "first"', () => {
    const out = alignRects(rects, 'left', 'first');
    expect(out[0].x).toBeCloseTo(0.1);
    expect(out[1].x).toBeCloseTo(0.1);
  });

  it('aligns to the FIGURE when ref is "figure"', () => {
    const out = alignRects(rects, 'left', 'figure');
    expect(out[0].x).toBeCloseTo(0);
    expect(out[1].x).toBeCloseTo(0);
  });

  it('centres horizontally about the reference centre', () => {
    const out = alignRects(rects, 'hcenter', 'figure');
    expect(out[0].x + out[0].w / 2).toBeCloseTo(0.5);
    expect(out[1].x + out[1].w / 2).toBeCloseTo(0.5);
  });

  it('aligns TOP using figure-space y-up semantics', () => {
    const out = alignRects(rects, 'top', 'selection');   // max top = 0.6
    expect(out[0].y + out[0].h).toBeCloseTo(0.6);
    expect(out[1].y + out[1].h).toBeCloseTo(0.6);
  });

  it('never changes any rect size', () => {
    for (const op of ['left','hcenter','right','top','vcenter','bottom'] as const) {
      const out = alignRects(rects, op, 'selection');
      out.forEach((r, i) => {
        expect(r.w).toBeCloseTo(rects[i].w);
        expect(r.h).toBeCloseTo(rects[i].h);
      });
    }
  });
});

describe('distributeRects', () => {
  it('equalises GAPS horizontally, pinning the outermost rects', () => {
    const rects = [R(0, 0, 0.1, 0.1), R(0.15, 0, 0.2, 0.1), R(0.8, 0, 0.2, 0.1)];
    const out = distributeRects(rects, 'x', 'gaps');
    expect(out[0].x).toBeCloseTo(0);          // first pinned
    expect(out[2].x).toBeCloseTo(0.8);        // last pinned
    const g1 = out[1].x - (out[0].x + out[0].w);
    const g2 = out[2].x - (out[1].x + out[1].w);
    expect(g1).toBeCloseTo(g2, 9);
  });

  it('equalises CENTRES horizontally', () => {
    const rects = [R(0, 0, 0.1, 0.1), R(0.15, 0, 0.3, 0.1), R(0.8, 0, 0.1, 0.1)];
    const out = distributeRects(rects, 'x', 'centers');
    const c = out.map((r) => r.x + r.w / 2);
    expect(c[1] - c[0]).toBeCloseTo(c[2] - c[1], 9);
  });

  it('distributes by position, not by input order', () => {
    const rects = [R(0.8, 0, 0.1, 0.1), R(0, 0, 0.1, 0.1), R(0.4, 0, 0.1, 0.1)];
    const out = distributeRects(rects, 'x', 'gaps');
    expect(out[1].x).toBeCloseTo(0);      // the leftmost input stays leftmost
    expect(out[0].x).toBeCloseTo(0.8);    // the rightmost stays rightmost
  });

  it('returns fewer than 3 rects unchanged', () => {
    const rects = [R(0, 0, 0.1, 0.1), R(0.5, 0, 0.1, 0.1)];
    expect(distributeRects(rects, 'x', 'gaps')).toEqual(rects);
  });
});

describe('matchSize', () => {
  it('matches widths to the FIRST rect without moving anything', () => {
    const rects = [R(0.1, 0.1, 0.25, 0.1), R(0.5, 0.4, 0.4, 0.2)];
    const out = matchSize(rects, 'w');
    expect(out[1].w).toBeCloseTo(0.25);
    expect(out[1].x).toBeCloseTo(0.5);
    expect(out[1].h).toBeCloseTo(0.2);
  });
});
