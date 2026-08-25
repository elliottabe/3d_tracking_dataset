import { describe, expect, it } from 'vitest';
import { findCollisions, outOfBounds } from './overlap';
import type { Rect } from './rect';

const R = (x: number, y: number, w: number, h: number): Rect => ({ x, y, w, h });

describe('findCollisions', () => {
  it('reports a pair whose INK boxes overlap even though their axes rects do not', () => {
    // This is the real M0 failure: the scut panel's xlabel sat inside the violin panel.
    const hits = findCollisions([
      { id: 'scut', ink: R(0.07, 0.30, 0.90, 0.14) },
      { id: 'zheight', ink: R(0.42, 0.40, 0.22, 0.24) },
    ]);
    expect(hits).toHaveLength(1);
    expect([hits[0].a, hits[0].b].sort()).toEqual(['scut', 'zheight']);
    expect(hits[0].area).toBeGreaterThan(0);
  });

  it('is silent when panels merely touch', () => {
    expect(findCollisions([
      { id: 'a', ink: R(0, 0, 0.5, 1) },
      { id: 'b', ink: R(0.5, 0, 0.5, 1) },
    ])).toHaveLength(0);
  });

  it('reports each pair once, not twice', () => {
    const hits = findCollisions([
      { id: 'a', ink: R(0, 0, 0.6, 0.6) },
      { id: 'b', ink: R(0.4, 0.4, 0.6, 0.6) },
      { id: 'c', ink: R(0.45, 0.45, 0.1, 0.1) },
    ]);
    const keys = hits.map((h) => [h.a, h.b].sort().join('|')).sort();
    expect(new Set(keys).size).toBe(keys.length);
    expect(keys).toHaveLength(3);
  });

  it('sorts the worst overlap first', () => {
    const hits = findCollisions([
      { id: 'a', ink: R(0, 0, 1, 1) },
      { id: 'b', ink: R(0, 0, 0.9, 0.9) },
      { id: 'c', ink: R(0.95, 0.95, 0.1, 0.1) },
    ]);
    expect(hits[0].area).toBeGreaterThanOrEqual(hits[hits.length - 1].area);
  });

  it('returns nothing for a clean layout', () => {
    expect(findCollisions([
      { id: 'a', ink: R(0.0, 0.0, 0.3, 0.3) },
      { id: 'b', ink: R(0.5, 0.5, 0.3, 0.3) },
    ])).toHaveLength(0);
  });
});

describe('outOfBounds', () => {
  it('flags a panel whose ink leaves the canvas', () => {
    expect(outOfBounds([
      { id: 'ok', ink: R(0.1, 0.1, 0.2, 0.2) },
      { id: 'clipped', ink: R(-0.01, 0.5, 0.2, 0.2) },
    ])).toEqual(['clipped']);
  });
  it('tolerates a hair of floating-point slop', () => {
    expect(outOfBounds([{ id: 'a', ink: R(-1e-12, 0, 1, 1) }])).toEqual([]);
  });
});
