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

  it('normalizes a rect dragged inside-out rather than emitting negative w', () => {
    const out = resizeRect(R, 'e', -0.6, 0);
    expect(out.w).toBeGreaterThanOrEqual(0);
    expect(out.x).toBeLessThanOrEqual(0.2);
  });

  it('honours a minimum size so a panel cannot collapse to nothing', () => {
    const out = resizeRect(R, 'e', -0.4, 0, { minW: 0.05, minH: 0.05 });
    expect(out.w).toBeCloseTo(0.05);
  });

  it('preserves aspect when asked', () => {
    const out = resizeRect(R, 'se', 0.2, 0, { aspect: true });
    expect(out.w / out.h).toBeCloseTo(R.w / R.h, 6);
  });
});
