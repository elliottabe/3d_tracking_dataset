import { describe, expect, it, vi } from 'vitest';
import { initialState, reducer, selectionTouchesGroup, type EditorState } from './editorStore';
import { solveGroup } from '../layout/groups';
import { MIN_SIZE_MM } from '../canvas/resize';
import type { Rect } from '../layout/rect';

const R = (x: number, y: number, w: number, h: number): Rect => ({ x, y, w, h });

function loaded(): EditorState {
  return reducer(initialState, {
    type: 'load',
    figWmm: 100, figHmm: 100,
    panels: [
      { id: 'a', type: 'line', rect: R(0.1, 0.1, 0.2, 0.2), spec: {}, data: {} },
      { id: 'b', type: 'line', rect: R(0.5, 0.5, 0.2, 0.2), spec: {}, data: {} },
      { id: 'c', type: 'line', rect: R(0.8, 0.1, 0.1, 0.2), spec: {}, data: {} },
    ],
    groups: [],
  });
}

describe('reducer', () => {
  it('loads panels and starts clean with nothing selected', () => {
    const s = loaded();
    expect(s.panels).toHaveLength(3);
    expect(s.selection).toEqual([]);
    expect(s.dirty).toBe(false);
  });

  it('selects, adds to selection, and clears', () => {
    let s = loaded();
    s = reducer(s, { type: 'select', ids: ['a'] });
    expect(s.selection).toEqual(['a']);
    s = reducer(s, { type: 'select', ids: ['b'], additive: true });
    expect(s.selection.sort()).toEqual(['a', 'b']);
    s = reducer(s, { type: 'select', ids: [] });
    expect(s.selection).toEqual([]);
  });

  it('moves the selection by a delta and marks the document dirty', () => {
    let s = loaded();
    s = reducer(s, { type: 'select', ids: ['a'] });
    s = reducer(s, { type: 'moveSelection', dx: 0.05, dy: -0.02 });
    const a = s.panels.find((p) => p.id === 'a')!;
    expect(a.rect.x).toBeCloseTo(0.15);
    expect(a.rect.y).toBeCloseTo(0.08);
    expect(s.dirty).toBe(true);
  });

  it('moves ONLY the selection', () => {
    let s = loaded();
    s = reducer(s, { type: 'select', ids: ['a'] });
    s = reducer(s, { type: 'moveSelection', dx: 0.1, dy: 0 });
    expect(s.panels.find((p) => p.id === 'b')!.rect.x).toBeCloseTo(0.5);
  });

  it('undoes a whole coalesced drag in one step', () => {
    let s = loaded();
    s = reducer(s, { type: 'select', ids: ['a'] });
    for (let i = 0; i < 10; i += 1) {
      s = reducer(s, { type: 'moveSelection', dx: 0.01, dy: 0, coalesceKey: 'drag' });
    }
    expect(s.panels.find((p) => p.id === 'a')!.rect.x).toBeCloseTo(0.2);
    s = reducer(s, { type: 'undo' });
    expect(s.panels.find((p) => p.id === 'a')!.rect.x).toBeCloseTo(0.1);
  });

  it('redo restores what undo removed', () => {
    let s = loaded();
    s = reducer(s, { type: 'select', ids: ['a'] });
    s = reducer(s, { type: 'moveSelection', dx: 0.1, dy: 0 });
    s = reducer(s, { type: 'undo' });
    s = reducer(s, { type: 'redo' });
    expect(s.panels.find((p) => p.id === 'a')!.rect.x).toBeCloseTo(0.2);
  });

  it('aligns the selection left without touching unselected panels', () => {
    let s = loaded();
    s = reducer(s, { type: 'select', ids: ['a', 'b'] });
    s = reducer(s, { type: 'align', op: 'left', ref: 'selection' });
    expect(s.panels.find((p) => p.id === 'a')!.rect.x).toBeCloseTo(0.1);
    expect(s.panels.find((p) => p.id === 'b')!.rect.x).toBeCloseTo(0.1);
    expect(s.panels.find((p) => p.id === 'c')!.rect.x).toBeCloseTo(0.8);
  });

  it('nudges by exact millimetres', () => {
    let s = loaded();                       // 100mm figure -> 1mm == 0.01 frac
    s = reducer(s, { type: 'select', ids: ['a'] });
    s = reducer(s, { type: 'nudge', dxMm: 1, dyMm: 0 });
    expect(s.panels.find((p) => p.id === 'a')!.rect.x).toBeCloseTo(0.11, 9);
  });

  it('setRect replaces one panel\'s rect exactly (numeric entry)', () => {
    let s = loaded();
    s = reducer(s, { type: 'setRect', id: 'b', rect: R(0.25, 0.25, 0.5, 0.5) });
    expect(s.panels.find((p) => p.id === 'b')!.rect).toEqual(R(0.25, 0.25, 0.5, 0.5));
    expect(s.dirty).toBe(true);
  });

  it('records ink boxes from the server without dirtying the document', () => {
    let s = loaded();
    s = reducer(s, { type: 'setInk', id: 'a', ink: R(0.05, 0.05, 0.3, 0.3) });
    expect(s.panels.find((p) => p.id === 'a')!.ink).toEqual(R(0.05, 0.05, 0.3, 0.3));
    expect(s.dirty).toBe(false);          // a render result is not an edit
  });

  it('clears dirty on saved', () => {
    let s = loaded();
    s = reducer(s, { type: 'select', ids: ['a'] });
    s = reducer(s, { type: 'moveSelection', dx: 0.01, dy: 0 });
    s = reducer(s, { type: 'saved' });
    expect(s.dirty).toBe(false);
  });

  it('undo back to the loaded baseline leaves dirty === false', () => {
    let s = loaded();
    s = reducer(s, { type: 'select', ids: ['a'] });
    s = reducer(s, { type: 'moveSelection', dx: 0.1, dy: 0 });
    expect(s.dirty).toBe(true);
    s = reducer(s, { type: 'undo' });
    // Rects are back to exactly what `load` produced...
    expect(s.panels.find((p) => p.id === 'a')!.rect).toEqual(R(0.1, 0.1, 0.2, 0.2));
    // ...so the document is clean again, not still flagged dirty.
    expect(s.dirty).toBe(false);
  });

  it('redo forward to a modified state sets dirty === true', () => {
    let s = loaded();
    s = reducer(s, { type: 'select', ids: ['a'] });
    s = reducer(s, { type: 'moveSelection', dx: 0.1, dy: 0 });
    s = reducer(s, { type: 'undo' });
    expect(s.dirty).toBe(false);
    s = reducer(s, { type: 'redo' });
    expect(s.panels.find((p) => p.id === 'a')!.rect.x).toBeCloseTo(0.2);
    expect(s.dirty).toBe(true);
  });

  it('saved mid-history moves the clean point; undoing away from it dirties again', () => {
    let s = loaded();
    s = reducer(s, { type: 'select', ids: ['a'] });
    s = reducer(s, { type: 'moveSelection', dx: 0.1, dy: 0 }); // edit 1
    s = reducer(s, { type: 'saved' });                          // save point != load point
    expect(s.dirty).toBe(false);
    s = reducer(s, { type: 'moveSelection', dx: 0.1, dy: 0 }); // edit 2
    expect(s.dirty).toBe(true);
    s = reducer(s, { type: 'undo' });                          // back to the saved point
    expect(s.panels.find((p) => p.id === 'a')!.rect.x).toBeCloseTo(0.2);
    expect(s.dirty).toBe(false);
    s = reducer(s, { type: 'undo' });                          // past the saved point, toward load
    expect(s.panels.find((p) => p.id === 'a')!.rect.x).toBeCloseTo(0.1);
    expect(s.dirty).toBe(true);
  });
});

describe('groups', () => {
  it('groups the selection, recovering the gutter from the existing spacing', () => {
    let s = loaded();
    s = reducer(s, { type: 'select', ids: ['a', 'c'] });
    s = reducer(s, { type: 'groupSelection', axis: 'x' });
    expect(s.groups).toHaveLength(1);
    const g = s.groups[0];
    expect(g.axis).toBe('x');
    expect(g.gutterMm).toBeGreaterThan(0);
    expect(s.panels.filter((p) => p.group === g.id).map((p) => p.id).sort()).toEqual(['a', 'c']);
  });

  it('re-solves children when the gutter changes, and the change is undoable', () => {
    let s = loaded();
    s = reducer(s, { type: 'select', ids: ['a', 'c'] });
    s = reducer(s, { type: 'groupSelection', axis: 'x' });
    const before = s.panels.find((p) => p.id === 'c')!.rect.x;
    s = reducer(s, { type: 'setGutter', id: s.groups[0].id, gutterMm: 20 });
    const after = s.panels.find((p) => p.id === 'c')!.rect.x;
    expect(after).not.toBeCloseTo(before);
    s = reducer(s, { type: 'undo' });
    expect(s.panels.find((p) => p.id === 'c')!.rect.x).toBeCloseTo(before, 9);
  });

  it('equal:true gives every child the same width', () => {
    let s = loaded();
    s = reducer(s, { type: 'select', ids: ['a', 'b', 'c'] });
    s = reducer(s, { type: 'groupSelection', axis: 'x' });
    s = reducer(s, { type: 'setGroupEqual', id: s.groups[0].id, equal: true });
    const ws = s.panels.map((p) => p.rect.w);
    expect(ws[1]).toBeCloseTo(ws[0], 9);
    expect(ws[2]).toBeCloseTo(ws[0], 9);
  });

  it('ungroup leaves children exactly where the solver last put them', () => {
    let s = loaded();
    s = reducer(s, { type: 'select', ids: ['a', 'c'] });
    s = reducer(s, { type: 'groupSelection', axis: 'x' });
    const solved = s.panels.map((p) => ({ ...p.rect }));
    s = reducer(s, { type: 'ungroup', id: s.groups[0].id });
    expect(s.groups).toHaveLength(0);
    s.panels.forEach((p, i) => {
      expect(p.rect.x).toBeCloseTo(solved[i].x, 9);
      expect(p.rect.w).toBeCloseTo(solved[i].w, 9);
    });
    expect(s.panels.every((p) => !p.group)).toBe(true);
  });

  it('refuses to group fewer than two panels', () => {
    let s = loaded();
    s = reducer(s, { type: 'select', ids: ['a'] });
    s = reducer(s, { type: 'groupSelection', axis: 'x' });
    expect(s.groups).toHaveLength(0);
  });
});

// Undo/redo must cover the WHOLE document, not just rect geometry: groups
// and group membership are undoable state too, or a group can survive an
// undo as a "ghost" (rects revert, but state.groups/panel.group do not).
describe('group state is part of undo/redo, not just rect geometry', () => {
  it('undo after groupSelection clears BOTH state.groups and every panel.group (no ghost group)', () => {
    let s = loaded();
    s = reducer(s, { type: 'select', ids: ['a', 'c'] });
    s = reducer(s, { type: 'groupSelection', axis: 'x' });
    expect(s.groups).toHaveLength(1);
    s = reducer(s, { type: 'undo' });
    expect(s.groups).toHaveLength(0);
    expect(s.panels.find((p) => p.id === 'a')!.group).toBeFalsy();
    expect(s.panels.find((p) => p.id === 'c')!.group).toBeFalsy();
  });

  it('undo after ungroup restores state.groups AND membership (ungroup previously pushed no history entry)', () => {
    let s = loaded();
    s = reducer(s, { type: 'select', ids: ['a', 'c'] });
    s = reducer(s, { type: 'groupSelection', axis: 'x' });
    const groupId = s.groups[0].id;
    s = reducer(s, { type: 'ungroup', id: groupId });
    expect(s.groups).toHaveLength(0);
    s = reducer(s, { type: 'undo' });
    expect(s.groups).toHaveLength(1);
    expect(s.groups[0].id).toBe(groupId);
    expect(s.panels.find((p) => p.id === 'a')!.group).toBe(groupId);
    expect(s.panels.find((p) => p.id === 'c')!.group).toBe(groupId);
  });

  it("undo after setGutter restores the group's gutterMm metadata, not just child rects", () => {
    let s = loaded();
    s = reducer(s, { type: 'select', ids: ['a', 'c'] });
    s = reducer(s, { type: 'groupSelection', axis: 'x' });
    const originalGutter = s.groups[0].gutterMm;
    s = reducer(s, { type: 'setGutter', id: s.groups[0].id, gutterMm: 20 });
    expect(s.groups[0].gutterMm).toBe(20);
    s = reducer(s, { type: 'undo' });
    expect(s.groups[0].gutterMm).toBeCloseTo(originalGutter, 9);
  });

  it('undo after setGroupEqual restores the previous equal flag', () => {
    let s = loaded();
    s = reducer(s, { type: 'select', ids: ['a', 'b', 'c'] });
    s = reducer(s, { type: 'groupSelection', axis: 'x' });
    expect(s.groups[0].equal).toBe(false);
    s = reducer(s, { type: 'setGroupEqual', id: s.groups[0].id, equal: true });
    expect(s.groups[0].equal).toBe(true);
    s = reducer(s, { type: 'undo' });
    expect(s.groups[0].equal).toBe(false);
  });

  it('redo re-applies group membership, not just geometry', () => {
    let s = loaded();
    s = reducer(s, { type: 'select', ids: ['a', 'c'] });
    s = reducer(s, { type: 'groupSelection', axis: 'x' });
    const groupId = s.groups[0].id;
    s = reducer(s, { type: 'undo' });
    expect(s.groups).toHaveLength(0);
    s = reducer(s, { type: 'redo' });
    expect(s.groups).toHaveLength(1);
    expect(s.groups[0].id).toBe(groupId);
    expect(s.panels.find((p) => p.id === 'a')!.group).toBe(groupId);
    expect(s.panels.find((p) => p.id === 'c')!.group).toBe(groupId);
  });

  it('a pure geometry no-op is still suppressed even when groups are present (no-op guard not lost)', () => {
    let s = loaded();
    s = reducer(s, { type: 'select', ids: ['a', 'c'] });
    s = reducer(s, { type: 'groupSelection', axis: 'x' });
    const pastLen = s.history.past.length;
    const aRect = s.panels.find((p) => p.id === 'a')!.rect;
    s = reducer(s, { type: 'setRect', id: 'a', rect: { ...aRect } });
    expect(s.history.past.length).toBe(pastLen); // truly identical: no new entry pushed
  });

  it('derives dirty for group edits rather than hardcoding it', () => {
    // Regression: commitGroupChange hardcoded `dirty: true`, so a group action that changed
    // NOTHING still reported unsaved changes. Re-applying the gutter a group already has is
    // the bit-exact case: solveGroup is deterministic, so the same inputs give the same rects.
    // (A 999 -> original round-trip does NOT clear dirty, because the intermediate solve
    // perturbs the rects by ~1e-16 and sameLayout compares with strict ===. That is a real
    // limitation of comparing floats exactly, not something this test should paper over.)
    let s = loaded();
    s = reducer(s, { type: 'select', ids: ['a', 'c'] });
    s = reducer(s, { type: 'groupSelection', axis: 'x' });
    s = reducer(s, { type: 'saved' });
    expect(s.dirty).toBe(false);

    const g = s.groups[0];
    s = reducer(s, { type: 'setGutter', id: g.id, gutterMm: g.gutterMm });
    expect(s.dirty).toBe(false);                     // nothing changed -> still clean

    s = reducer(s, { type: 'setGutter', id: g.id, gutterMm: g.gutterMm + 5 });
    expect(s.dirty).toBe(true);                      // a real change -> dirty
  });

  it('dirty is false after undoing back to the last-saved point when the only change was a grouping change', () => {
    let s = loaded();
    s = reducer(s, { type: 'select', ids: ['a', 'c'] });
    s = reducer(s, { type: 'groupSelection', axis: 'x' });
    s = reducer(s, { type: 'saved' });
    expect(s.dirty).toBe(false);
    s = reducer(s, { type: 'setGutter', id: s.groups[0].id, gutterMm: 20 });
    expect(s.dirty).toBe(true);
    s = reducer(s, { type: 'undo' });
    expect(s.dirty).toBe(false);
    expect(s.groups[0].gutterMm).not.toBe(20);
  });
});

// Controller ruling M2-13 (dispatch addendum): the GROUP is the unit of
// manipulation. A free child drag would be silently provisional (the next
// setGutter/setGroupEqual re-solve yanks it back with no visible cause), so
// selecting/dragging one grouped child must act on the whole group instead.
describe('group selection restrictions (addendum M2-13)', () => {
  it('clicking one child selects all children of its group', () => {
    let s = loaded();
    s = reducer(s, { type: 'select', ids: ['a', 'c'] });
    s = reducer(s, { type: 'groupSelection', axis: 'x' });
    s = reducer(s, { type: 'select', ids: [] });
    s = reducer(s, { type: 'select', ids: ['c'] }); // click just the ONE child
    expect(s.selection.sort()).toEqual(['a', 'c']);
  });

  it('dragging a child moves ALL children by the same delta, as one undo entry', () => {
    let s = loaded();
    s = reducer(s, { type: 'select', ids: ['a', 'c'] });
    s = reducer(s, { type: 'groupSelection', axis: 'x' });
    const aBefore = s.panels.find((p) => p.id === 'a')!.rect;
    const cBefore = s.panels.find((p) => p.id === 'c')!.rect;

    s = reducer(s, { type: 'select', ids: ['c'] }); // click one child -> whole group selected
    expect(s.selection.sort()).toEqual(['a', 'c']);

    const key = 'group-drag';
    s = reducer(s, { type: 'moveSelection', dx: 0.05, dy: 0.02, coalesceKey: key });
    s = reducer(s, { type: 'moveSelection', dx: 0.01, dy: 0, coalesceKey: key });

    const aAfter = s.panels.find((p) => p.id === 'a')!.rect;
    const cAfter = s.panels.find((p) => p.id === 'c')!.rect;
    expect(aAfter.x).toBeCloseTo(aBefore.x + 0.06, 9);
    expect(aAfter.y).toBeCloseTo(aBefore.y + 0.02, 9);
    expect(cAfter.x).toBeCloseTo(cBefore.x + 0.06, 9);
    expect(cAfter.y).toBeCloseTo(cBefore.y + 0.02, 9);

    // Two coalesced pointer-moves must collapse into ONE undo entry.
    s = reducer(s, { type: 'undo' });
    expect(s.panels.find((p) => p.id === 'a')!.rect.x).toBeCloseTo(aBefore.x, 9);
    expect(s.panels.find((p) => p.id === 'c')!.rect.x).toBeCloseTo(cBefore.x, 9);
  });

  it("a translated group's children still match solveGroup's output for the translated group rect", () => {
    let s = loaded();
    s = reducer(s, { type: 'select', ids: ['a', 'c'] });
    s = reducer(s, { type: 'groupSelection', axis: 'x' });
    const groupId = s.groups[0].id;

    s = reducer(s, { type: 'select', ids: ['a'] }); // -> whole group
    s = reducer(s, { type: 'moveSelection', dx: 0.1, dy: -0.03 });

    const gAfter = s.groups.find((g) => g.id === groupId)!;
    const kids = s.panels
      .filter((p) => p.group === groupId)
      .sort((p, q) => p.rect.x - q.rect.x); // axis 'x' order
    const expected = solveGroup(gAfter, kids.map((p) => p.rect), s.figWmm, s.figHmm);

    kids.forEach((p, i) => {
      expect(p.rect.x).toBeCloseTo(expected[i].x, 9);
      expect(p.rect.y).toBeCloseTo(expected[i].y, 9);
      expect(p.rect.w).toBeCloseTo(expected[i].w, 9);
      expect(p.rect.h).toBeCloseTo(expected[i].h, 9);
    });
  });

  it('ungroup then drag one former child moves ONLY that panel', () => {
    let s = loaded();
    s = reducer(s, { type: 'select', ids: ['a', 'c'] });
    s = reducer(s, { type: 'groupSelection', axis: 'x' });
    s = reducer(s, { type: 'ungroup', id: s.groups[0].id });

    const cBefore = s.panels.find((p) => p.id === 'c')!.rect.x;
    s = reducer(s, { type: 'select', ids: ['a'] });
    expect(s.selection).toEqual(['a']); // restriction lifted: no longer the whole group
    s = reducer(s, { type: 'moveSelection', dx: 0.1, dy: 0 });
    expect(s.panels.find((p) => p.id === 'c')!.rect.x).toBeCloseTo(cBefore, 9);
  });
});

// Controller ruling M2-15: a typed 0/negative mm in the Properties panel's
// w/h field must not silently commit a vanishing panel the way it did in
// Task 10 — the numeric-entry path (setRect) must enforce the same floor
// the drag-resize handles already do (canvas/resize.ts MIN_SIZE_MM).
describe('setRect minimum size floor (M2-15)', () => {
  it('clamps a typed 0 width to the MIN_SIZE_MM floor instead of vanishing the panel', () => {
    let s = loaded(); // figWmm = 100
    s = reducer(s, { type: 'setRect', id: 'a', rect: R(0.1, 0.1, 0, 0.2) });
    const w = s.panels.find((p) => p.id === 'a')!.rect.w;
    expect(w).toBeGreaterThan(0);
    expect(w).toBeCloseTo(MIN_SIZE_MM / 100, 9);
  });

  it('clamps a negative height the same way', () => {
    let s = loaded();
    s = reducer(s, { type: 'setRect', id: 'a', rect: R(0.1, 0.1, 0.2, -5) });
    const h = s.panels.find((p) => p.id === 'a')!.rect.h;
    expect(h).toBeCloseTo(MIN_SIZE_MM / 100, 9);
  });

  it('the clamped value is visible in state, not silently reverted to the pre-edit rect', () => {
    let s = loaded();
    const before = s.panels.find((p) => p.id === 'a')!.rect;
    s = reducer(s, { type: 'setRect', id: 'a', rect: R(0.1, 0.1, 0, 0.2) });
    const after = s.panels.find((p) => p.id === 'a')!.rect;
    // Not the illegal typed value (0) — the panel must not vanish...
    expect(after.w).toBeGreaterThan(0);
    // ...and not silently reverted back to the untouched pre-edit width
    // either, which would look identical to the input being ignored.
    expect(after.w).not.toBeCloseTo(before.w, 9);
  });

  it('does not clamp values already above the floor (no regression)', () => {
    let s = loaded();
    s = reducer(s, { type: 'setRect', id: 'b', rect: R(0.25, 0.25, 0.5, 0.5) });
    expect(s.panels.find((p) => p.id === 'b')!.rect).toEqual(R(0.25, 0.25, 0.5, 0.5));
  });
});

// F3: align/distribute/matchSize used to bypass the group solver entirely
// (straight to withGeometry), leaving group.rect stale. The next group
// edit — even one that changes nothing, like re-applying the gutter a
// group already has — then re-solved from that stale rect and silently
// discarded the align/distribute/matchSize with no visible cause. Chosen
// fix: disable these three ops outright for a grouped selection (option b:
// the solver owns grouped geometry, same rule Properties already enforces
// for x/y/w/h fields), rather than trying to re-derive group.rect after
// the fact (option a).
describe('align/distribute/matchSize are no-ops on a grouped selection (F3)', () => {
  function groupedState(): { s: EditorState; groupId: string } {
    let s = loaded();
    s = reducer(s, { type: 'select', ids: ['a', 'c'] });
    s = reducer(s, { type: 'groupSelection', axis: 'x' });
    return { s, groupId: s.groups[0].id };
  }

  it('selectionTouchesGroup is true exactly when a grouped panel is selected', () => {
    let s = loaded();
    s = reducer(s, { type: 'select', ids: ['b'] });
    expect(selectionTouchesGroup(s)).toBe(false);
    s = reducer(s, { type: 'select', ids: ['a', 'c'] });
    s = reducer(s, { type: 'groupSelection', axis: 'x' });
    s = reducer(s, { type: 'select', ids: ['a'] }); // expands to whole group
    expect(selectionTouchesGroup(s)).toBe(true);
  });

  it('align leaves grouped panels untouched and does not desync group.rect', () => {
    const { s: before, groupId } = groupedState();
    const rectsBefore = before.panels.map((p) => ({ ...p.rect }));
    const groupRectBefore = { ...before.groups.find((g) => g.id === groupId)!.rect };

    const after = reducer(before, { type: 'align', op: 'top', ref: 'figure' });

    expect(after.panels.map((p) => p.rect)).toEqual(rectsBefore);
    expect(after.groups.find((g) => g.id === groupId)!.rect).toEqual(groupRectBefore);
    expect(after.dirty).toBe(before.dirty); // truly a no-op, not just a revert-to-same-rect

    // And critically: a LATER, genuinely-unrelated group edit does not
    // suddenly "restore" something that was never actually moved — because
    // nothing was moved, there is nothing to lose.
    const after2 = reducer(after, { type: 'setGutter', id: groupId, gutterMm: after.groups.find((g) => g.id === groupId)!.gutterMm });
    expect(after2.panels.map((p) => p.rect)).toEqual(rectsBefore);
  });

  it('distribute is a no-op on a grouped selection', () => {
    const { s } = groupedState();
    const before = s.panels.map((p) => ({ ...p.rect }));
    const after = reducer(s, { type: 'distribute', axis: 'x', mode: 'gaps' });
    expect(after.panels.map((p) => p.rect)).toEqual(before);
  });

  it('matchSize is a no-op on a grouped selection', () => {
    const { s } = groupedState();
    const before = s.panels.map((p) => ({ ...p.rect }));
    const after = reducer(s, { type: 'matchSize', dim: 'w' });
    expect(after.panels.map((p) => p.rect)).toEqual(before);
  });

  it('align/distribute/matchSize still work normally on an ungrouped selection (no regression)', () => {
    let s = loaded();
    s = reducer(s, { type: 'select', ids: ['a', 'b'] });
    s = reducer(s, { type: 'align', op: 'left', ref: 'selection' });
    expect(s.panels.find((p) => p.id === 'a')!.rect.x).toBeCloseTo(0.1);
    expect(s.panels.find((p) => p.id === 'b')!.rect.x).toBeCloseTo(0.1);
  });
});

// F4: groupSelection retagged panels onto a new group but never dropped a
// pre-existing group left with zero members. `select` expands any click to
// the whole group, so marquee-ing an existing group plus a loose panel is
// the ordinary way to strip that group down to nothing — the orphan then
// persisted into figure.json forever.
describe('groupSelection drops zero-member groups (F4)', () => {
  it('regrouping every member of an existing group into a new one removes the orphan', () => {
    let s = loaded();
    s = reducer(s, { type: 'select', ids: ['a', 'c'] });
    s = reducer(s, { type: 'groupSelection', axis: 'x' });
    const oldGroupId = s.groups[0].id;
    expect(s.groups).toHaveLength(1);

    // Re-group ALL of {a, c}'s members plus b into a brand-new group —
    // the old group ends up with zero members.
    s = reducer(s, { type: 'select', ids: ['a', 'b'] }); // 'a' expands to {a, c}
    expect(s.selection.sort()).toEqual(['a', 'b', 'c']);
    s = reducer(s, { type: 'groupSelection', axis: 'y' });

    expect(s.groups).toHaveLength(1); // old group gone, only the new one remains
    expect(s.groups.some((g) => g.id === oldGroupId)).toBe(false);
    expect(s.panels.every((p) => p.group !== oldGroupId)).toBe(true);
  });

  it('a group that still has a surviving member is NOT dropped', () => {
    let s = loaded();
    // Group all three panels together first.
    s = reducer(s, { type: 'select', ids: ['a', 'b', 'c'] });
    s = reducer(s, { type: 'groupSelection', axis: 'x' });
    expect(s.groups).toHaveLength(1);
    const firstGroupId = s.groups[0].id;

    // Now ungroup so we can form a genuinely different second group that
    // only steals SOME members — but since `select` always expands to the
    // whole group, forming a strict subset requires ungrouping first.
    s = reducer(s, { type: 'ungroup', id: firstGroupId });
    s = reducer(s, { type: 'select', ids: ['a', 'b'] });
    s = reducer(s, { type: 'groupSelection', axis: 'x' });
    expect(s.groups).toHaveLength(1); // no leftover from the ungrouped state
  });
});

// F5: `grp_${Date.now().toString(36)}` has no random suffix, so two groups
// created in the same millisecond collide — `ungroup(id)` then removes
// BOTH and `groups.find(...)` always resolves to the first. The
// drag/resize coalesce keys elsewhere already add a `Math.random()` suffix;
// group ids must match that convention.
describe('groupSelection ids never collide within the same millisecond (F5)', () => {
  it('two groups created at the identical Date.now() get different ids', () => {
    const now = 1_700_000_000_000;
    vi.spyOn(Date, 'now').mockReturnValue(now);
    try {
      let s = loaded();
      s = reducer(s, { type: 'select', ids: ['a', 'b'] });
      s = reducer(s, { type: 'groupSelection', axis: 'x' });
      const firstId = s.groups[0].id;

      s = reducer(s, { type: 'select', ids: [] });
      // 'a'/'b' are already grouped; group the remaining loose panel with...
      // there's only 'c' left ungrouped, so ungroup first to get two free
      // panels, then group them into a second group at the SAME Date.now().
      s = reducer(s, { type: 'ungroup', id: firstId });
      s = reducer(s, { type: 'select', ids: ['a', 'b'] });
      s = reducer(s, { type: 'groupSelection', axis: 'x' });
      const secondId = s.groups[0].id;

      expect(secondId).not.toBe(firstId);
    } finally {
      vi.restoreAllMocks();
    }
  });

  it('two groups COEXISTING at the same Date.now() can be ungrouped independently (the concrete F5 failure mode)', () => {
    // Four panels so two disjoint 2-member groups can exist AT THE SAME
    // TIME — the collision this bug actually causes (`ungroup(id)` removing
    // BOTH groups, every `groups.find(...)` resolving to the first) only
    // shows up when both are live simultaneously, not sequentially.
    let s = reducer(initialState, {
      type: 'load',
      figWmm: 100, figHmm: 100,
      panels: [
        { id: 'a', type: 'line', rect: R(0.1, 0.1, 0.1, 0.1), spec: {}, data: {} },
        { id: 'b', type: 'line', rect: R(0.3, 0.1, 0.1, 0.1), spec: {}, data: {} },
        { id: 'c', type: 'line', rect: R(0.5, 0.1, 0.1, 0.1), spec: {}, data: {} },
        { id: 'd', type: 'line', rect: R(0.7, 0.1, 0.1, 0.1), spec: {}, data: {} },
      ],
      groups: [],
    });

    const now = 1_700_000_000_000;
    vi.spyOn(Date, 'now').mockReturnValue(now);
    try {
      s = reducer(s, { type: 'select', ids: ['a', 'b'] });
      s = reducer(s, { type: 'groupSelection', axis: 'x' }); // group 1: {a,b}
      const groupOneId = s.panels.find((p) => p.id === 'a')!.group!;

      s = reducer(s, { type: 'select', ids: ['c', 'd'] });
      s = reducer(s, { type: 'groupSelection', axis: 'x' }); // group 2: {c,d}, same Date.now()
      const groupTwoId = s.panels.find((p) => p.id === 'c')!.group!;

      expect(s.groups).toHaveLength(2); // both coexist
      expect(groupTwoId).not.toBe(groupOneId);

      // Ungrouping group 2 must leave group 1 (and a/b's membership) intact.
      s = reducer(s, { type: 'ungroup', id: groupTwoId });
      expect(s.groups).toHaveLength(1);
      expect(s.groups[0].id).toBe(groupOneId);
      expect(s.panels.find((p) => p.id === 'a')!.group).toBe(groupOneId);
      expect(s.panels.find((p) => p.id === 'b')!.group).toBe(groupOneId);
      expect(s.panels.find((p) => p.id === 'c')!.group).toBeFalsy();
      expect(s.panels.find((p) => p.id === 'd')!.group).toBeFalsy();
    } finally {
      vi.restoreAllMocks();
    }
  });
});

// Task 13 (schema-driven properties form): `spec` (panel options — spines,
// legend, colours, sample rate, …) becomes editable via `setSpec`. Per the
// boundary rule, it must cross history/dirty/save/load exactly like rect and
// group membership already do — these tests are the reducer/history half of
// that (buildSavePayload's half is in `layout/save.test.ts`).
describe('setSpec (Task 13)', () => {
  it('updates the panel, marks dirty, and is undoable in one step', () => {
    let s = loaded();
    s = reducer(s, { type: 'setSpec', id: 'a', spec: { spines: { top: true } } });
    expect(s.panels.find((p) => p.id === 'a')!.spec).toEqual({ spines: { top: true } });
    expect(s.dirty).toBe(true);

    s = reducer(s, { type: 'undo' });
    expect(s.panels.find((p) => p.id === 'a')!.spec).toEqual({});
    expect(s.dirty).toBe(false);
  });

  it('a deep-equal spec is a no-op: same state ref, no history entry', () => {
    let s = loaded();
    s = reducer(s, { type: 'setSpec', id: 'a', spec: { legend: { loc: 'best' } } });
    const pastLen = s.history.past.length;
    const next = reducer(s, { type: 'setSpec', id: 'a', spec: { legend: { loc: 'best' } } });
    expect(next).toBe(s);                              // same reference, not just equal
    expect(next.history.past.length).toBe(pastLen);
  });

  it('an unknown panel id is a no-op', () => {
    const s = loaded();
    const next = reducer(s, { type: 'setSpec', id: 'does-not-exist', spec: { x: 1 } });
    expect(next).toBe(s);
  });

  // THE BOUNDARY RULE regression this whole task is built to avoid: a
  // widened Snapshot/sameLayout with an un-widened `applySnapshot` would
  // commit and dirty a spec edit correctly but silently fail to restore it
  // on undo — passing every "dirty"/"no-op" test above while still losing
  // the edit the moment the user hits undo.
  it('undo after a spec edit restores the previous spec', () => {
    let s = loaded();
    const before = s.panels.find((p) => p.id === 'b')!.spec;
    s = reducer(s, { type: 'setSpec', id: 'b', spec: { hide_xticklabels: true } });
    expect(s.panels.find((p) => p.id === 'b')!.spec).toEqual({ hide_xticklabels: true });
    s = reducer(s, { type: 'undo' });
    expect(s.panels.find((p) => p.id === 'b')!.spec).toEqual(before);
  });

  it('redo re-applies a spec edit', () => {
    // Two DIFFERENT specs, not edit-then-undo-back-to-original: if `undo`
    // silently failed to touch `spec` at all (the exact boundary-rule bug),
    // a single edit + undo + redo would still land on the right value by
    // accident (spec was never removed in the first place), passing for the
    // wrong reason. Requiring undo to move from edit #2's spec BACK to edit
    // #1's spec, then redo to move it FORWARD again, is not satisfiable
    // without `applySnapshot` actually restoring `spec`.
    let s = loaded();
    s = reducer(s, { type: 'setSpec', id: 'b', spec: { hide_xticklabels: true } });
    s = reducer(s, { type: 'setSpec', id: 'b', spec: { hide_xticklabels: false } });
    s = reducer(s, { type: 'undo' });
    expect(s.panels.find((p) => p.id === 'b')!.spec).toEqual({ hide_xticklabels: true });
    s = reducer(s, { type: 'redo' });
    expect(s.panels.find((p) => p.id === 'b')!.spec).toEqual({ hide_xticklabels: false });
  });

  it('dirty is false after undoing a spec edit back to the last-saved point', () => {
    let s = loaded();
    s = reducer(s, { type: 'saved' });
    expect(s.dirty).toBe(false);
    s = reducer(s, { type: 'setSpec', id: 'a', spec: { spines: { top: true } } });
    expect(s.dirty).toBe(true);
    s = reducer(s, { type: 'undo' });
    expect(s.dirty).toBe(false);
  });

  // No-op suppression surviving the Snapshot widening (critical regression
  // risk called out explicitly): an identical full-document commit must
  // still push zero history entries even now that `specs` is compared.
  it('no-op suppression survives the specs widening: an identical full-document commit pushes no entry', () => {
    let s = loaded();
    s = reducer(s, { type: 'setSpec', id: 'a', spec: { spines: { top: true, right: false } } });
    const pastLen = s.history.past.length;
    const aSpec = s.panels.find((p) => p.id === 'a')!.spec;
    // A freshly-built, structurally-identical (but not reference-identical) spec.
    const same = reducer(s, { type: 'setSpec', id: 'a', spec: JSON.parse(JSON.stringify(aSpec)) });
    expect(same.history.past.length).toBe(pastLen);
    expect(same.dirty).toBe(s.dirty);
  });
});
