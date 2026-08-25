import { describe, expect, it } from 'vitest';
import {
  edges, fracToMm, mmToFrac, normalizeRect, rectToScreen, rectsIntersect, unionRect,
  type Rect,
} from './rect';

const R = (x: number, y: number, w: number, h: number): Rect => ({ x, y, w, h });

describe('unit conversion', () => {
  it('round-trips fraction -> mm -> fraction', () => {
    const r = R(0.1, 0.2, 0.5, 0.25);
    const back = mmToFrac(fracToMm(r, 183, 140), 183, 140);
    expect(back.x).toBeCloseTo(r.x, 12);
    expect(back.h).toBeCloseTo(r.h, 12);
  });

  it('converts a rect to millimetres against the figure size', () => {
    const mm = fracToMm(R(0.5, 0.5, 0.25, 0.5), 183, 140);
    expect(mm.x).toBeCloseTo(91.5);
    expect(mm.h).toBeCloseTo(70);
  });
});

describe('rectToScreen', () => {
  it('flips y exactly once: a bottom-anchored rect lands at the BOTTOM of the canvas', () => {
    // figure space y=0 is the BOTTOM; SVG y=0 is the TOP.
    const s = rectToScreen(R(0, 0, 1, 0.25), 400, 200);
    expect(s.x).toBeCloseTo(0);
    expect(s.y).toBeCloseTo(150);      // 200 - (0 + 0.25)*200
    expect(s.h).toBeCloseTo(50);
  });

  it('a top-anchored rect lands at y=0', () => {
    const s = rectToScreen(R(0, 0.75, 1, 0.25), 400, 200);
    expect(s.y).toBeCloseTo(0);
  });
});

describe('edges', () => {
  it('reports left/right/bottom/top and centres in figure space', () => {
    const e = edges(R(0.2, 0.3, 0.4, 0.2));
    expect(e.left).toBeCloseTo(0.2);
    expect(e.right).toBeCloseTo(0.6);
    expect(e.bottom).toBeCloseTo(0.3);
    expect(e.top).toBeCloseTo(0.5);
    expect(e.cx).toBeCloseTo(0.4);
    expect(e.cy).toBeCloseTo(0.4);
  });
});

describe('unionRect', () => {
  it('bounds every input rect', () => {
    const u = unionRect([R(0.1, 0.1, 0.2, 0.2), R(0.5, 0.4, 0.2, 0.3)])!;
    expect(u.x).toBeCloseTo(0.1);
    expect(u.y).toBeCloseTo(0.1);
    expect(u.w).toBeCloseTo(0.6);
    expect(u.h).toBeCloseTo(0.6);
  });
  it('returns null for an empty list', () => {
    expect(unionRect([])).toBeNull();
  });
});

describe('rectsIntersect', () => {
  it('detects overlap and separation', () => {
    expect(rectsIntersect(R(0, 0, 0.5, 0.5), R(0.4, 0.4, 0.5, 0.5))).toBe(true);
    expect(rectsIntersect(R(0, 0, 0.4, 0.4), R(0.5, 0.5, 0.4, 0.4))).toBe(false);
  });
  it('treats mere touching as NOT overlapping', () => {
    expect(rectsIntersect(R(0, 0, 0.5, 1), R(0.5, 0, 0.5, 1))).toBe(false);
  });
});

describe('normalizeRect', () => {
  it('flips a negative-width rect produced by dragging leftwards', () => {
    const n = normalizeRect(R(0.6, 0.5, -0.2, -0.1));
    expect(n.x).toBeCloseTo(0.4);
    expect(n.y).toBeCloseTo(0.4);
    expect(n.w).toBeCloseTo(0.2);
    expect(n.h).toBeCloseTo(0.1);
  });
});
