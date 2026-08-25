import { describe, expect, it } from 'vitest';
import { hitTestPanels, marqueeHits, screenToFrac } from './hitTest';
import { rectToScreen, type Rect } from '../layout/rect';

const P = (id: string, x: number, y: number, w: number, h: number) =>
  ({ id, rect: { x, y, w, h } as Rect });

const panels = [
  P('back', 0.0, 0.0, 1.0, 1.0),
  P('mid', 0.2, 0.2, 0.4, 0.4),
  P('front', 0.3, 0.3, 0.1, 0.1),
];

describe('hitTestPanels', () => {
  it('returns the TOPMOST panel under the point (last wins)', () => {
    expect(hitTestPanels(panels, { x: 0.35, y: 0.35 })).toBe('front');
  });
  it('falls through to a lower panel outside the top one', () => {
    expect(hitTestPanels(panels, { x: 0.55, y: 0.55 })).toBe('mid');
  });
  it('returns null on empty space', () => {
    expect(hitTestPanels([P('a', 0, 0, 0.1, 0.1)], { x: 0.9, y: 0.9 })).toBeNull();
  });
});

describe('marqueeHits', () => {
  it('selects panels that INTERSECT the marquee', () => {
    const ids = marqueeHits(panels, { x: 0.25, y: 0.25, w: 0.2, h: 0.2 });
    expect(ids).toEqual(expect.arrayContaining(['mid', 'front']));
  });
  it('selects nothing for a marquee over empty space', () => {
    expect(marqueeHits([P('a', 0, 0, 0.1, 0.1)], { x: 0.5, y: 0.5, w: 0.2, h: 0.2 })).toEqual([]);
  });
  it('handles a marquee dragged up-left (negative w/h)', () => {
    const ids = marqueeHits([P('a', 0.1, 0.1, 0.2, 0.2)], { x: 0.5, y: 0.5, w: -0.35, h: -0.35 });
    expect(ids).toEqual(['a']);
  });
});

describe('screenToFrac', () => {
  it('round-trips a rect\'s top-left through the inverse transform (zoom != 1, offset != 0)', () => {
    const figWpt = 200, figHpt = 100, zoom = 2.5;
    const svgRect = { left: 37, top: 11 };
    const r: Rect = { x: 0.3, y: 0.2, w: 0.25, h: 0.4 };

    // rectToScreen(r, ...) gives the rect's top-left in viewBox pt.
    const screen = rectToScreen(r, figWpt, figHpt);
    const clientX = screen.x * zoom + svgRect.left;
    const clientY = screen.y * zoom + svgRect.top;

    const frac = screenToFrac(clientX, clientY, svgRect, figWpt, figHpt, zoom);
    // Top-left of a y-up rect is (x, y + h) in fraction space.
    expect(frac.x).toBeCloseTo(r.x, 9);
    expect(frac.y).toBeCloseTo(r.y + r.h, 9);
  });

  it('maps the figure centre to itself', () => {
    const figWpt = 150, figHpt = 90, zoom = 1.7;
    const svgRect = { left: 12, top: 5 };
    // Centre point (0.5, 0.5) forward-transforms to (0.5*figWpt, 0.5*figHpt)
    // in viewBox pt (the y-flip is a no-op at the midpoint).
    const clientX = 0.5 * figWpt * zoom + svgRect.left;
    const clientY = 0.5 * figHpt * zoom + svgRect.top;

    const frac = screenToFrac(clientX, clientY, svgRect, figWpt, figHpt, zoom);
    expect(frac.x).toBeCloseTo(0.5, 9);
    expect(frac.y).toBeCloseTo(0.5, 9);
  });

  it('lands y=0 (figure bottom) and y=1 (figure top) the right way up', () => {
    const figWpt = 120, figHpt = 80, zoom = 1;
    const svgRect = { left: 0, top: 0 };

    // Bottom of the figure (fraction y=0) is the BOTTOM of the SVG, i.e.
    // the largest viewBox-pt y (figHpt), since SVG y grows down.
    const bottom = screenToFrac(0, figHpt, svgRect, figWpt, figHpt, zoom);
    expect(bottom.y).toBeCloseTo(0, 9);

    // Top of the figure (fraction y=1) is viewBox-pt y=0.
    const top = screenToFrac(0, 0, svgRect, figWpt, figHpt, zoom);
    expect(top.y).toBeCloseTo(1, 9);
  });
});
