/**
 * Undo/redo over whole-layout snapshots.
 *
 * Snapshots rather than inverse-operations: a layout is a few dozen small
 * rects, so copying it is cheap and every operation becomes undoable for free
 * without each one having to describe its own inverse.
 *
 * Coalescing exists because a drag emits dozens of pointer-move commits;
 * without it one drag would cost dozens of undos.
 */
import type { Rect } from './rect';

export type Snapshot = Record<string, Rect>;

export type History = {
  past: Snapshot[];
  present: Snapshot;
  future: Snapshot[];
  /** Key of the entry currently open for coalescing, if any. */
  openKey?: string;
};

export function initHistory(present: Snapshot): History {
  return { past: [], present, future: [] };
}

export function sameLayout(a: Snapshot, b: Snapshot): boolean {
  const ka = Object.keys(a);
  const kb = Object.keys(b);
  if (ka.length !== kb.length) return false;
  return ka.every((k) => {
    const p = a[k];
    const q = b[k];
    return q !== undefined && p.x === q.x && p.y === q.y && p.w === q.w && p.h === q.h;
  });
}

export function commit(
  h: History, next: Snapshot, opts?: { coalesceKey?: string },
): History {
  if (sameLayout(h.present, next)) return h;

  const key = opts?.coalesceKey;
  // Continuing an open coalesced entry: replace `present`, do not grow `past`.
  if (key !== undefined && h.openKey === key) {
    return { past: h.past, present: next, future: [], openKey: key };
  }
  return {
    past: [...h.past, h.present],
    present: next,
    future: [],
    openKey: key,
  };
}

export function undo(h: History): History {
  if (h.past.length === 0) return h;
  const prev = h.past[h.past.length - 1];
  return {
    past: h.past.slice(0, -1),
    present: prev,
    future: [h.present, ...h.future],
    openKey: undefined,
  };
}

export function redo(h: History): History {
  if (h.future.length === 0) return h;
  const [next, ...rest] = h.future;
  return {
    past: [...h.past, h.present],
    present: next,
    future: rest,
    openKey: undefined,
  };
}

export const canUndo = (h: History): boolean => h.past.length > 0;
export const canRedo = (h: History): boolean => h.future.length > 0;
