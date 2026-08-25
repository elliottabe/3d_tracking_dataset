import { describe, expect, it } from 'vitest';
import { groupBounds, gutterFromChildren, solveGroup, type Group } from './groups';
import type { Rect } from './rect';

const R = (x: number, y: number, w: number, h: number): Rect => ({ x, y, w, h });
const G = (over: Partial<Group> = {}): Group => ({
  id: 'g', axis: 'x', gutterMm: 2, equal: true, rect: R(0.1, 0.5, 0.8, 0.2), ...over,
});

describe('solveGroup', () => {
  it('lays equal-width children across the group rect with the given gutter', () => {
    const kids = [R(0,0,0.1,0.2), R(0,0,0.3,0.2), R(0,0,0.2,0.2), R(0,0,0.05,0.2)];
    const out = solveGroup(G(), kids, 100, 100);          // gutter 2mm = 0.02 frac
    expect(out).toHaveLength(4);
    const w = out[0].w;
    out.forEach((r) => expect(r.w).toBeCloseTo(w, 9));    // equal:true
    expect(out[0].x).toBeCloseTo(0.1, 9);                  // starts at group left
    expect(out[3].x + out[3].w).toBeCloseTo(0.9, 9);       // ends at group right
    expect(out[1].x - (out[0].x + out[0].w)).toBeCloseTo(0.02, 9);
  });

  it('preserves relative widths when equal is false', () => {
    const kids = [R(0,0,0.1,0.2), R(0,0,0.3,0.2)];
    const out = solveGroup(G({ equal: false, gutterMm: 0 }), kids, 100, 100);
    expect(out[1].w / out[0].w).toBeCloseTo(3, 6);
    expect(out[0].x + out[0].w + out[1].w).toBeCloseTo(0.9, 6);
  });

  it('stacks on the y axis TOP-DOWN so child 0 is visually first', () => {
    const kids = [R(0,0,0.2,0.1), R(0,0,0.2,0.1)];
    const out = solveGroup(G({ axis: 'y', gutterMm: 0 }), kids, 100, 100);
    // group rect y=0.5 h=0.2 -> child 0 occupies the UPPER half
    expect(out[0].y + out[0].h).toBeCloseTo(0.7, 9);
    expect(out[1].y).toBeCloseTo(0.5, 9);
  });

  it('gives every child the group\'s cross-axis extent', () => {
    const kids = [R(0,0,0.1,0.05), R(0,0,0.1,0.9)];
    const out = solveGroup(G(), kids, 100, 100);
    out.forEach((r) => {
      expect(r.y).toBeCloseTo(0.5, 9);
      expect(r.h).toBeCloseTo(0.2, 9);
    });
  });

  it('returns a single child filling the group rect', () => {
    const out = solveGroup(G(), [R(0,0,0.1,0.1)], 100, 100);
    expect(out[0]).toEqual(G().rect);
  });

  it('never produces a negative width when the gutter exceeds the group', () => {
    const kids = [R(0,0,0.1,0.1), R(0,0,0.1,0.1), R(0,0,0.1,0.1)];
    const out = solveGroup(G({ gutterMm: 90 }), kids, 100, 100);
    out.forEach((r) => expect(r.w).toBeGreaterThanOrEqual(0));
  });
});

describe('groupBounds', () => {
  it('is the union of its children', () => {
    const b = groupBounds([R(0.1,0.2,0.2,0.1), R(0.5,0.2,0.2,0.1)]);
    expect(b.x).toBeCloseTo(0.1);
    expect(b.w).toBeCloseTo(0.6);
  });
});

describe('gutterFromChildren', () => {
  it('recovers the mean gap in mm, so ungroup->regroup is stable', () => {
    const kids = [R(0.1,0,0.2,0.1), R(0.35,0,0.2,0.1), R(0.60,0,0.2,0.1)];
    expect(gutterFromChildren(kids, 'x', 100, 100)).toBeCloseTo(5, 6);
  });
  it('is 0 for a single child', () => {
    expect(gutterFromChildren([R(0,0,0.2,0.1)], 'x', 100, 100)).toBe(0);
  });
});
