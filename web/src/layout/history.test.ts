import { describe, expect, it } from 'vitest';
import { canRedo, canUndo, commit, initHistory, redo, undo } from './history';
import type { Rect } from './rect';

const S = (x: number): Record<string, Rect> => ({ p: { x, y: 0, w: 0.1, h: 0.1 } });

describe('history', () => {
  it('undoes and redoes a single change', () => {
    let h = initHistory(S(0));
    h = commit(h, S(1));
    expect(h.present.p.x).toBe(1);
    h = undo(h);
    expect(h.present.p.x).toBe(0);
    h = redo(h);
    expect(h.present.p.x).toBe(1);
  });

  it('coalesces consecutive commits sharing a key into ONE undo step', () => {
    let h = initHistory(S(0));
    for (let i = 1; i <= 20; i += 1) h = commit(h, S(i), { coalesceKey: 'drag-p' });
    expect(h.present.p.x).toBe(20);
    h = undo(h);
    expect(h.present.p.x).toBe(0);     // the whole drag, not one pointer-move
    expect(canUndo(h)).toBe(false);
  });

  it('starts a new entry when the coalesce key changes', () => {
    let h = initHistory(S(0));
    h = commit(h, S(1), { coalesceKey: 'drag-a' });
    h = commit(h, S(2), { coalesceKey: 'drag-b' });
    h = undo(h);
    expect(h.present.p.x).toBe(1);
  });

  it('an uncoalesced commit always starts a new entry', () => {
    let h = initHistory(S(0));
    h = commit(h, S(1));
    h = commit(h, S(2));
    h = undo(h);
    expect(h.present.p.x).toBe(1);
  });

  it('clears the redo stack once a new change is committed', () => {
    let h = initHistory(S(0));
    h = commit(h, S(1));
    h = undo(h);
    expect(canRedo(h)).toBe(true);
    h = commit(h, S(9));
    expect(canRedo(h)).toBe(false);
    expect(h.present.p.x).toBe(9);
  });

  it('is a no-op at the ends of the stack', () => {
    let h = initHistory(S(0));
    expect(canUndo(h)).toBe(false);
    h = undo(h);
    expect(h.present.p.x).toBe(0);
    h = redo(h);
    expect(h.present.p.x).toBe(0);
  });

  it('does not record a commit identical to the present', () => {
    let h = initHistory(S(0));
    h = commit(h, S(0));
    expect(canUndo(h)).toBe(false);
  });
});
