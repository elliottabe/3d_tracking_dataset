/**
 * F2 regression: `App.tsx`'s `onSave` overlaid only `panels[].rect` onto the
 * refetched on-disk document, never `groups` or `panels[].group` — so a
 * grouped edit round-tripped through save+reload as if it had never
 * happened, while `dirty` (derived over rects+groups+membership) had
 * already gone false and the toolbar said "Saved". `buildSavePayload` is
 * the extracted, pure, directly-testable replacement.
 */
import { describe, expect, it } from 'vitest';
import type { FigureDoc } from '../api';
import { reducer, initialState, type EditorState } from '../state/editorStore';
import { buildSavePayload } from './save';

function baseDoc(): FigureDoc {
  return {
    figure: { width_mm: 100, height_mm: 100, dpi: 300 },
    panels: [
      { id: 'a', type: 'line', rect: [0.1, 0.1, 0.2, 0.2], data: {}, spec: {}, group: null, z: 0 },
      { id: 'b', type: 'line', rect: [0.5, 0.5, 0.2, 0.2], data: {}, spec: {}, group: null, z: 0 },
      { id: 'c', type: 'line', rect: [0.8, 0.1, 0.1, 0.2], data: {}, spec: {}, group: null, z: 0 },
    ],
    groups: [],
    annotations: [{ note: 'kept untouched' }],
  };
}

function loadedFrom(doc: FigureDoc): EditorState {
  return reducer(initialState, {
    type: 'load',
    figWmm: doc.figure.width_mm,
    figHmm: doc.figure.height_mm,
    panels: doc.panels.map((p) => ({
      id: p.id, type: p.type,
      rect: { x: p.rect[0], y: p.rect[1], w: p.rect[2], h: p.rect[3] },
      group: p.group ?? null, spec: p.spec, data: p.data,
    })),
    groups: doc.groups.map((g) => ({
      id: g.id, axis: g.axis, gutterMm: g.gutter_mm, equal: g.equal,
      rect: { x: g.rect[0], y: g.rect[1], w: g.rect[2], h: g.rect[3] },
    })),
  });
}

describe('buildSavePayload (F2)', () => {
  it('a grouped, edited state produces a payload whose groups AND panels[].group reflect the edit', () => {
    const original = baseDoc();
    let s = loadedFrom(original);
    s = reducer(s, { type: 'select', ids: ['a', 'c'] });
    s = reducer(s, { type: 'groupSelection', axis: 'x' });
    expect(s.groups).toHaveLength(1);
    const groupId = s.groups[0].id;

    const payload = buildSavePayload(original, s);

    // groups written back, in on-disk shape (snake_case, tuple rect).
    expect(payload.groups).toHaveLength(1);
    expect(payload.groups[0].id).toBe(groupId);
    expect(payload.groups[0].gutter_mm).toBe(s.groups[0].gutterMm);
    expect(payload.groups[0].rect).toEqual([
      s.groups[0].rect.x, s.groups[0].rect.y, s.groups[0].rect.w, s.groups[0].rect.h,
    ]);

    // panels[].group written back for both members, untouched for 'b'.
    const byId = Object.fromEntries(payload.panels.map((p) => [p.id, p]));
    expect(byId.a.group).toBe(groupId);
    expect(byId.c.group).toBe(groupId);
    expect(byId.b.group).toBeNull();

    // rects reflect the solver's output, not the pre-group rects.
    const solvedA = s.panels.find((p) => p.id === 'a')!.rect;
    expect(byId.a.rect).toEqual([solvedA.x, solvedA.y, solvedA.w, solvedA.h]);

    // fields the editor doesn't manage survive untouched.
    expect(payload.annotations).toEqual(original.annotations);
  });

  it("a group referencing a panel and the panel referencing the group are never split (figure.py's unknown-group check)", () => {
    const original = baseDoc();
    let s = loadedFrom(original);
    s = reducer(s, { type: 'select', ids: ['a', 'c'] });
    s = reducer(s, { type: 'groupSelection', axis: 'x' });
    const payload = buildSavePayload(original, s);

    const groupIds = new Set(payload.groups.map((g) => g.id));
    for (const p of payload.panels) {
      if (p.group != null) expect(groupIds.has(p.group)).toBe(true);
    }
  });

  it('ungrouping is reflected: payload has no groups and no panel references one', () => {
    const original = baseDoc();
    let s = loadedFrom(original);
    s = reducer(s, { type: 'select', ids: ['a', 'c'] });
    s = reducer(s, { type: 'groupSelection', axis: 'x' });
    s = reducer(s, { type: 'ungroup', id: s.groups[0].id });

    const payload = buildSavePayload(original, s);
    expect(payload.groups).toHaveLength(0);
    expect(payload.panels.every((p) => p.group == null)).toBe(true);
  });

  it('an un-grouped rect edit still overlays correctly (no regression on the original rect-only behaviour)', () => {
    const original = baseDoc();
    let s = loadedFrom(original);
    s = reducer(s, { type: 'select', ids: ['b'] });
    s = reducer(s, { type: 'moveSelection', dx: 0.05, dy: -0.02 });

    const payload = buildSavePayload(original, s);
    const b = payload.panels.find((p) => p.id === 'b')!;
    expect(b.rect).toEqual([0.55, 0.48, 0.2, 0.2]);
  });

  // Task 13: `spec` is now editable, so it must cross the save boundary
  // exactly like rect/group already do (buildSavePayload's own docstring —
  // the F2 fix this file exists for — was the earlier instance of this same
  // defect class for group/membership).
  it('includes an edited spec, in on-disk shape, preserving unknown fields and untouched panels', () => {
    const original = baseDoc();
    let s = loadedFrom(original);
    s = reducer(s, {
      type: 'setSpec',
      id: 'a',
      spec: { spines: { top: true, right: false }, legend: { hide: true, bbox_to_anchor: [0, 1] } },
    });

    const payload = buildSavePayload(original, s);
    const a = payload.panels.find((p) => p.id === 'a')!;
    expect(a.spec).toEqual({ spines: { top: true, right: false }, legend: { hide: true, bbox_to_anchor: [0, 1] } });

    // untouched panels keep their on-disk spec unchanged.
    const b = payload.panels.find((p) => p.id === 'b')!;
    expect(b.spec).toEqual(original.panels.find((p) => p.id === 'b')!.spec);

    // fields the editor doesn't manage still survive untouched.
    expect(payload.annotations).toEqual(original.annotations);
  });
});
