import { describe, expect, it } from 'vitest';
import { initialState, reducer, type EditorState } from './editorStore';
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
