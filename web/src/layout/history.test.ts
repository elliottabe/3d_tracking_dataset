import { describe, expect, it } from 'vitest';
import { canRedo, canUndo, commit, initHistory, redo, sameLayout, undo } from './history';
import type { Group } from './groups';
import type { Snapshot } from './history';

const S = (x: number): Snapshot => ({
  rects: { p: { x, y: 0, w: 0.1, h: 0.1 } },
  groups: [],
  membership: { p: null },
  specs: { p: {} },
});

const G = (id: string, over?: Partial<Group>): Group => ({
  id, axis: 'x', gutterMm: 5, equal: false, rect: { x: 0, y: 0, w: 1, h: 1 }, ...over,
});

describe('history', () => {
  it('undoes and redoes a single change', () => {
    let h = initHistory(S(0));
    h = commit(h, S(1));
    expect(h.present.rects.p.x).toBe(1);
    h = undo(h);
    expect(h.present.rects.p.x).toBe(0);
    h = redo(h);
    expect(h.present.rects.p.x).toBe(1);
  });

  it('coalesces consecutive commits sharing a key into ONE undo step', () => {
    let h = initHistory(S(0));
    for (let i = 1; i <= 20; i += 1) h = commit(h, S(i), { coalesceKey: 'drag-p' });
    expect(h.present.rects.p.x).toBe(20);
    h = undo(h);
    expect(h.present.rects.p.x).toBe(0);     // the whole drag, not one pointer-move
    expect(canUndo(h)).toBe(false);
  });

  it('starts a new entry when the coalesce key changes', () => {
    let h = initHistory(S(0));
    h = commit(h, S(1), { coalesceKey: 'drag-a' });
    h = commit(h, S(2), { coalesceKey: 'drag-b' });
    h = undo(h);
    expect(h.present.rects.p.x).toBe(1);
  });

  it('an uncoalesced commit always starts a new entry', () => {
    let h = initHistory(S(0));
    h = commit(h, S(1));
    h = commit(h, S(2));
    h = undo(h);
    expect(h.present.rects.p.x).toBe(1);
  });

  it('clears the redo stack once a new change is committed', () => {
    let h = initHistory(S(0));
    h = commit(h, S(1));
    h = undo(h);
    expect(canRedo(h)).toBe(true);
    h = commit(h, S(9));
    expect(canRedo(h)).toBe(false);
    expect(h.present.rects.p.x).toBe(9);
  });

  it('is a no-op at the ends of the stack', () => {
    let h = initHistory(S(0));
    expect(canUndo(h)).toBe(false);
    h = undo(h);
    expect(h.present.rects.p.x).toBe(0);
    h = redo(h);
    expect(h.present.rects.p.x).toBe(0);
  });

  it('does not record a commit identical to the present', () => {
    let h = initHistory(S(0));
    h = commit(h, S(0));
    expect(canUndo(h)).toBe(false);
  });
});

describe('sameLayout', () => {
  it('is exported and treats equal snapshots as equal', () => {
    expect(sameLayout(S(1), S(1))).toBe(true);
  });

  it('detects a differing rect on a shared key', () => {
    expect(sameLayout(S(1), S(2))).toBe(false);
  });

  it('detects a differing key count', () => {
    const a = S(1);
    const b: Snapshot = {
      ...S(1),
      rects: { ...S(1).rects, q: { x: 0, y: 0, w: 0.2, h: 0.2 } },
    };
    expect(sameLayout(a, b)).toBe(false);
  });

  it('detects a differing groups array (a group appearing/disappearing)', () => {
    const a: Snapshot = { rects: {}, groups: [], membership: {}, specs: {} };
    const b: Snapshot = { rects: {}, groups: [G('g1')], membership: {}, specs: {} };
    expect(sameLayout(a, b)).toBe(false);
  });

  it('detects a differing group field (gutterMm) even with identical rects', () => {
    const a: Snapshot = { rects: {}, groups: [G('g1', { gutterMm: 5 })], membership: {}, specs: {} };
    const b: Snapshot = { rects: {}, groups: [G('g1', { gutterMm: 20 })], membership: {}, specs: {} };
    expect(sameLayout(a, b)).toBe(false);
  });

  it('detects a differing group order (same members, different order)', () => {
    const a: Snapshot = { rects: {}, groups: [G('g1'), G('g2')], membership: {}, specs: {} };
    const b: Snapshot = { rects: {}, groups: [G('g2'), G('g1')], membership: {}, specs: {} };
    expect(sameLayout(a, b)).toBe(false);
  });

  it('detects a differing membership entry with identical rects and groups', () => {
    const groups = [G('g1')];
    const a: Snapshot = { rects: {}, groups, membership: { p: 'g1' }, specs: { p: {} } };
    const b: Snapshot = { rects: {}, groups, membership: { p: null }, specs: { p: {} } };
    expect(sameLayout(a, b)).toBe(false);
  });

  it('treats snapshots with identical rects, groups, and membership as equal', () => {
    const groups = [G('g1')];
    const a: Snapshot = { rects: { p: { x: 1, y: 0, w: 0.1, h: 0.1 } }, groups, membership: { p: 'g1' }, specs: { p: {} } };
    const b: Snapshot = { rects: { p: { x: 1, y: 0, w: 0.1, h: 0.1 } }, groups, membership: { p: 'g1' }, specs: { p: {} } };
    expect(sameLayout(a, b)).toBe(true);
  });

  // `spec` values are arbitrary JSON (booleans/numbers/strings/arrays/nested
  // objects), so this comparator needs a real deep-equal, not per-key `===`
  // (which is all `sameRects`/`sameMembership` need, since rects/ids are
  // flat) — these three cases are the ones a naive `===` would get wrong.
  it('a nested spec object differing at depth 2 is NOT equal', () => {
    const a = { ...S(1), specs: { p: { legend: { loc: 'best' } } } };
    const b = { ...S(1), specs: { p: { legend: { loc: 'upper right' } } } };
    expect(sameLayout(a, b)).toBe(false);
  });

  it('spec arrays of different length or order are NOT equal', () => {
    const a = { ...S(1), specs: { p: { legend: { bbox_to_anchor: [0, 1] } } } };
    const b = { ...S(1), specs: { p: { legend: { bbox_to_anchor: [0, 1, 0.2] } } } };
    const c = { ...S(1), specs: { p: { legend: { bbox_to_anchor: [1, 0] } } } };
    expect(sameLayout(a, b)).toBe(false);
    expect(sameLayout(a, c)).toBe(false);
  });

  it('identical nested spec structures (freshly-built, not shared by reference) ARE equal', () => {
    const spec = { spines: { top: false, right: false, bottom: true, left: true }, legend: { hide: false, loc: 'best', bbox_to_anchor: [0, 1] } };
    const a = { ...S(1), specs: { p: JSON.parse(JSON.stringify(spec)) } };
    const b = { ...S(1), specs: { p: JSON.parse(JSON.stringify(spec)) } };
    expect(sameLayout(a, b)).toBe(true);
  });
});
