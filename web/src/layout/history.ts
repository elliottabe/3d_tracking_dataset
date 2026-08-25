/**
 * Undo/redo over whole-document snapshots.
 *
 * Snapshots rather than inverse-operations: a layout is a few dozen small
 * rects plus a handful of groups, so copying it is cheap and every operation
 * becomes undoable for free without each one having to describe its own
 * inverse.
 *
 * A snapshot covers the WHOLE undoable document, not just rect geometry:
 * `groups` (the group metadata: axis/gutter/equal/rect) and `membership`
 * (which group, if any, each panel belongs to) are captured and restored
 * exactly like rects. Geometry-only snapshots left grouping/ungrouping
 * invisible to both undo (a "ghost" group could survive undo) and to the
 * no-op guard in `commit` (ungroup usually moves no rects, so it pushed NO
 * history entry at all).
 *
 * Coalescing exists because a drag emits dozens of pointer-move commits;
 * without it one drag would cost dozens of undos.
 */
import type { Group } from './groups';
import type { Rect } from './rect';

export type Snapshot = {
  rects: Record<string, Rect>;
  groups: Group[];
  /** Panel id -> the group id it belongs to, or null if ungrouped. */
  membership: Record<string, string | null>;
};

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

function sameRect(p: Rect, q: Rect): boolean {
  return p.x === q.x && p.y === q.y && p.w === q.w && p.h === q.h;
}

function sameRects(a: Record<string, Rect>, b: Record<string, Rect>): boolean {
  const ka = Object.keys(a);
  const kb = Object.keys(b);
  if (ka.length !== kb.length) return false;
  return ka.every((k) => {
    const q = b[k];
    return q !== undefined && sameRect(a[k], q);
  });
}

// Order matters: two group lists with the same members in a different order
// are a different document state (e.g. redo must reproduce insertion order).
function sameGroups(a: Group[], b: Group[]): boolean {
  if (a.length !== b.length) return false;
  return a.every((g, i) => {
    const h = b[i];
    return h !== undefined
      && g.id === h.id
      && g.axis === h.axis
      && g.gutterMm === h.gutterMm
      && g.equal === h.equal
      && sameRect(g.rect, h.rect);
  });
}

function sameMembership(
  a: Record<string, string | null>, b: Record<string, string | null>,
): boolean {
  const ka = Object.keys(a);
  const kb = Object.keys(b);
  if (ka.length !== kb.length) return false;
  return ka.every((k) => k in b && a[k] === b[k]);
}

export function sameLayout(a: Snapshot, b: Snapshot): boolean {
  return sameRects(a.rects, b.rects)
    && sameGroups(a.groups, b.groups)
    && sameMembership(a.membership, b.membership);
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
