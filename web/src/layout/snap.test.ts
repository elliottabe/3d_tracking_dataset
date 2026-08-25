import { describe, expect, it } from 'vitest';
import { buildTargets, screenTolToFrac, snapRect } from './snap';
import type { Rect } from './rect';

const R = (x: number, y: number, w: number, h: number): Rect => ({ x, y, w, h });

describe('buildTargets', () => {
  it('emits figure margins on both axes', () => {
    const t = buildTargets({ others: [], gridMm: 0, figWmm: 100, figHmm: 100, guides: [] });
    const xs = t.filter((s) => s.axis === 'x' && s.kind === 'margin').map((s) => s.value);
    expect(xs).toEqual(expect.arrayContaining([0, 1]));
  });

  it('emits neighbour edges AND centres', () => {
    const t = buildTargets({
      others: [R(0.2, 0.1, 0.4, 0.3)], gridMm: 0, figWmm: 100, figHmm: 100, guides: [],
    });
    const xs = t.filter((s) => s.axis === 'x');
    expect(xs.some((s) => s.kind === 'edge' && Math.abs(s.value - 0.2) < 1e-9)).toBe(true);
    expect(xs.some((s) => s.kind === 'edge' && Math.abs(s.value - 0.6) < 1e-9)).toBe(true);
    expect(xs.some((s) => s.kind === 'center' && Math.abs(s.value - 0.4) < 1e-9)).toBe(true);
  });

  it('emits a grid when gridMm > 0 and none when it is 0', () => {
    const on = buildTargets({ others: [], gridMm: 10, figWmm: 100, figHmm: 100, guides: [] });
    expect(on.some((s) => s.kind === 'grid')).toBe(true);
    const off = buildTargets({ others: [], gridMm: 0, figWmm: 100, figHmm: 100, guides: [] });
    expect(off.some((s) => s.kind === 'grid')).toBe(false);
  });
});

describe('snapRect', () => {
  const targets = buildTargets({
    others: [R(0.2, 0.1, 0.4, 0.3)], gridMm: 0, figWmm: 100, figHmm: 100, guides: [],
  });

  it('snaps a near-miss left edge onto a neighbour edge and reports the guide', () => {
    const { rect, guides } = snapRect(R(0.205, 0.5, 0.2, 0.2), targets, 0.01);
    expect(rect.x).toBeCloseTo(0.2, 9);
    expect(rect.w).toBeCloseTo(0.2, 9);   // snapping MOVES, never resizes
    expect(guides.some((g) => g.axis === 'x' && Math.abs(g.value - 0.2) < 1e-9)).toBe(true);
  });

  it('leaves a rect alone when nothing is within tolerance', () => {
    const { rect, guides } = snapRect(R(0.44, 0.55, 0.2, 0.2), targets, 0.001);
    expect(rect.x).toBeCloseTo(0.44, 9);
    expect(guides).toHaveLength(0);
  });

  it('prefers the CLOSEST candidate when several are in range', () => {
    const t = [
      { axis: 'x' as const, value: 0.30, kind: 'edge' as const },
      { axis: 'x' as const, value: 0.34, kind: 'edge' as const },
    ];
    const { rect } = snapRect(R(0.335, 0, 0.1, 0.1), t, 0.05);
    expect(rect.x).toBeCloseTo(0.34, 9);
  });

  it('can snap the RIGHT edge, shifting x by the difference', () => {
    const t = [{ axis: 'x' as const, value: 0.6, kind: 'edge' as const }];
    const { rect } = snapRect(R(0.395, 0, 0.2, 0.1), t, 0.02);  // right edge 0.595
    expect(rect.x).toBeCloseTo(0.4, 9);
  });

  it('snaps x and y independently in one call', () => {
    const t = [
      { axis: 'x' as const, value: 0.5, kind: 'edge' as const },
      { axis: 'y' as const, value: 0.25, kind: 'edge' as const },
    ];
    const { rect } = snapRect(R(0.497, 0.252, 0.1, 0.1), t, 0.01);
    expect(rect.x).toBeCloseTo(0.5, 9);
    expect(rect.y).toBeCloseTo(0.25, 9);
  });
});

describe('screenTolToFrac', () => {
  it('shrinks the fractional tolerance as zoom increases, so snapping feels constant on screen', () => {
    const at1 = screenTolToFrac(6, 400, 1);
    const at4 = screenTolToFrac(6, 400, 4);
    expect(at4).toBeCloseTo(at1 / 4, 12);
  });
});
