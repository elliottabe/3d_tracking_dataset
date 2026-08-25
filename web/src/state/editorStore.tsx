/**
 * The whole editor as a pure reducer, plus a thin React context over it.
 *
 * Rects live in exactly ONE place (`panels[].rect`); `history.present` mirrors
 * them so undo is uniform across drags, aligns, nudges and numeric entry. The
 * reducer imports no React and is tested without it.
 */
import { createContext, useContext, useMemo, useReducer, type ReactNode } from 'react';
import { alignRects, distributeRects, matchSize, type AlignOp, type RefMode } from '../layout/align';
import { canRedo, canUndo, commit, initHistory, redo, sameLayout, undo, type History, type Snapshot } from '../layout/history';
import { groupBounds, gutterFromChildren, solveGroup, type Group } from '../layout/groups';
import { MIN_SIZE_MM } from '../canvas/resize';
import type { BoxMode, SnapTarget } from '../layout/snap';
import type { Rect } from '../layout/rect';

export type PanelState = {
  id: string;
  type: string;
  rect: Rect;
  /** Last ink box reported by the server for this panel, if rendered. */
  ink?: Rect;
  group?: string | null;
  spec: Record<string, unknown>;
  data: Record<string, unknown>;
};

export type EditorState = {
  figWmm: number;
  figHmm: number;
  panels: PanelState[];
  groups: Group[];
  selection: string[];
  boxMode: BoxMode;
  gridMm: number;
  guides: SnapTarget[];
  history: History;
  dirty: boolean;
  /** Layout snapshot as of the last `load`/`saved`; `dirty` is derived against it. */
  savedSnapshot: Snapshot;
};

export type Action =
  | { type: 'load'; figWmm: number; figHmm: number; panels: PanelState[]; groups: Group[] }
  | { type: 'select'; ids: string[]; additive?: boolean }
  | { type: 'moveSelection'; dx: number; dy: number; coalesceKey?: string }
  | { type: 'setRect'; id: string; rect: Rect; coalesceKey?: string }
  | { type: 'align'; op: AlignOp; ref: RefMode }
  | { type: 'distribute'; axis: 'x' | 'y'; mode: 'gaps' | 'centers' }
  | { type: 'matchSize'; dim: 'w' | 'h' }
  | { type: 'nudge'; dxMm: number; dyMm: number }
  | { type: 'setBoxMode'; mode: BoxMode }
  | { type: 'setInk'; id: string; ink: Rect }
  | { type: 'groupSelection'; axis: 'x' | 'y' }
  | { type: 'ungroup'; id: string }
  | { type: 'setGutter'; id: string; gutterMm: number }
  | { type: 'setGroupEqual'; id: string; equal: boolean }
  | { type: 'undo' }
  | { type: 'redo' }
  | { type: 'saved' };

const EMPTY_SNAPSHOT: Snapshot = { rects: {}, groups: [], membership: {} };

export const initialState: EditorState = {
  figWmm: 183, figHmm: 140,
  panels: [], groups: [], selection: [],
  boxMode: 'axes', gridMm: 1, guides: [],
  history: initHistory(EMPTY_SNAPSHOT), dirty: false, savedSnapshot: EMPTY_SNAPSHOT,
};

/**
 * The whole undoable document: rect geometry, group metadata, and each
 * panel's group membership. All three round-trip through undo/redo and
 * through the no-op guard in `commit` (via `sameLayout`) — geometry alone
 * left grouping/ungrouping invisible to both.
 */
const snapshotOf = (panels: PanelState[], groups: Group[]): Snapshot => ({
  rects: Object.fromEntries(panels.map((p) => [p.id, p.rect])),
  groups,
  membership: Object.fromEntries(panels.map((p) => [p.id, p.group ?? null])),
});

const applySnapshot = (panels: PanelState[], snap: Snapshot): PanelState[] =>
  panels.map((p) => ({
    ...p,
    rect: snap.rects[p.id] ?? p.rect,
    group: snap.membership[p.id] ?? null,
  }));

/** Commit a new set of panels (and possibly-translated groups) through
 *  history, re-deriving `dirty`. */
function withGeometry(
  state: EditorState, panels: PanelState[], coalesceKey?: string,
): EditorState {
  const history = commit(state.history, snapshotOf(panels, state.groups), { coalesceKey });
  if (history === state.history) return state;   // nothing actually moved
  return { ...state, panels, history, dirty: !sameLayout(history.present, state.savedSnapshot) };
}

/**
 * Commit a group create/edit/ungroup. `groups`/`membership` (carried inside
 * `panels[].group`) are part of the snapshot now, so a group create/edit/
 * ungroup is a genuine snapshot change even when no rect moves (e.g.
 * ungroup, or a solver re-run that reproduces the original spacing exactly)
 * — `commit`'s no-op guard no longer discards it, and it always marks the
 * document dirty since the persisted `groups` array changed.
 */
function commitGroupChange(
  state: EditorState, panels: PanelState[], groups: Group[],
): EditorState {
  const history = commit(state.history, snapshotOf(panels, groups));
  // Derive `dirty` exactly as withGeometry does. Hardcoding `true` here left a group edit
  // round-tripped back to its saved value (setGutter(x) -> setGutter(original)) reporting
  // unsaved changes forever. Now that Snapshot carries groups + membership, the comparison
  // against savedSnapshot is meaningful for group edits too.
  return { ...state, panels, groups, history, dirty: !sameLayout(history.present, state.savedSnapshot) };
}

/** Map a transform over the selected panels only. */
function mapSelected(
  state: EditorState, fn: (rects: Rect[]) => Rect[],
): PanelState[] {
  const sel = state.panels.filter((p) => state.selection.includes(p.id));
  if (sel.length === 0) return state.panels;
  const out = fn(sel.map((p) => p.rect));
  const byId = new Map(sel.map((p, i) => [p.id, out[i]]));
  return state.panels.map((p) => (byId.has(p.id) ? { ...p, rect: byId.get(p.id)! } : p));
}

/**
 * The GROUP is the unit of manipulation (controller ruling M2-13): clicking
 * or marquee-touching one grouped child must select every sibling in that
 * group, never the child alone — otherwise a lone-child drag is silently
 * provisional and a later setGutter/setGroupEqual yanks it back with no
 * visible cause. Expand every requested id to its whole group before it
 * ever reaches `state.selection`, so every other action that reads
 * `state.selection` (moveSelection, align, …) already sees whole groups.
 */
function expandToGroups(panels: PanelState[], ids: string[]): string[] {
  const out = new Set<string>();
  for (const id of ids) {
    const g = panels.find((p) => p.id === id)?.group;
    if (g) {
      panels.forEach((p) => { if (p.group === g) out.add(p.id); });
    } else {
      out.add(id);
    }
  }
  return Array.from(out);
}

/**
 * Translating a grouped child moves the whole group by the same delta (a
 * pure translation commutes with `solveGroup`, per the same ruling) — but
 * the group's own stored `rect` must move with it, or the NEXT
 * setGutter/setGroupEqual re-solve uses the stale pre-drag rect and snaps
 * the group straight back to where it started.
 */
function translateGroups(
  state: EditorState, movedIds: Set<string>, dx: number, dy: number,
): Group[] {
  const touched = new Set(
    state.panels.filter((p) => p.group && movedIds.has(p.id)).map((p) => p.group!),
  );
  if (touched.size === 0) return state.groups;
  return state.groups.map((g) => (touched.has(g.id)
    ? { ...g, rect: { ...g.rect, x: g.rect.x + dx, y: g.rect.y + dy } }
    : g));
}

/** Re-solve one group's children and return the full panel list. */
function resolveGroup(state: EditorState, groupId: string, groups: Group[]): PanelState[] {
  const g = groups.find((x) => x.id === groupId);
  if (!g) return state.panels;
  const kids = state.panels.filter((p) => p.group === groupId);
  if (kids.length === 0) return state.panels;
  // Solve in the children's own left-to-right / top-to-bottom order.
  const axisKey = g.axis;
  const ordered = [...kids].sort((a, b) =>
    (axisKey === 'x' ? a.rect.x - b.rect.x : b.rect.y - a.rect.y));
  const solved = solveGroup(g, ordered.map((p) => p.rect), state.figWmm, state.figHmm);
  const byId = new Map(ordered.map((p, i) => [p.id, solved[i]]));
  return state.panels.map((p) => (byId.has(p.id) ? { ...p, rect: byId.get(p.id)! } : p));
}

/** Floor shared with the drag-resize handles (canvas/resize.ts) so a typed
 *  0/negative mm cannot commit a vanishing panel (controller ruling M2-15). */
function clampMinSize(r: Rect, figWmm: number, figHmm: number): Rect {
  const minW = MIN_SIZE_MM / figWmm;
  const minH = MIN_SIZE_MM / figHmm;
  return { ...r, w: Math.max(r.w, minW), h: Math.max(r.h, minH) };
}

/**
 * True when the current selection includes any panel that belongs to a
 * group (F3). `align`/`distribute`/`matchSize` write rects directly via
 * `withGeometry` and never touch `group.rect`, so running them on a grouped
 * panel leaves the group's stored rect stale; the NEXT group edit (even one
 * that changes nothing, like retyping the gutter it already has) re-solves
 * from that stale rect and silently reverts the align/distribute/match-size
 * with no visible cause. `expandToGroups` already guarantees that selecting
 * ANY grouped panel selects its whole group, so this is equivalent to "the
 * selection is a whole group" for the ordinary case, and also catches a
 * mixed selection (grouped + ungrouped) reached via additive shift-select.
 *
 * Chosen fix (option b in the M2 review, F3): disable these three ops for a
 * grouped selection, consistent with the existing rule that the group
 * solver — not direct geometry edits — owns a grouped child's rect
 * (`Properties.tsx` already withholds x/y/w/h fields the same way). Option
 * (a) — re-deriving `group.rect` from `groupBounds` after these ops — would
 * let them work on grouped selections too, but for align/distribute in
 * particular the solver's own axis/gutter model doesn't obviously agree
 * with "align left" or "distribute equal gaps": recomputing the bounding
 * rect afterward can still leave the CHILDREN's relative spacing fighting
 * the group's gutter on the very next re-solve. Disabling is the smaller,
 * unambiguous behaviour, and matches the "solver owns grouped geometry"
 * rule already enforced everywhere else.
 */
export function selectionTouchesGroup(state: EditorState): boolean {
  return state.selection.some((id) => !!state.panels.find((p) => p.id === id)?.group);
}

export function reducer(state: EditorState, action: Action): EditorState {
  switch (action.type) {
    case 'load': {
      const panels = action.panels;
      const groups = action.groups;
      const snapshot = snapshotOf(panels, groups);
      return {
        ...state,
        figWmm: action.figWmm,
        figHmm: action.figHmm,
        panels,
        groups,
        selection: [],
        history: initHistory(snapshot),
        dirty: false,
        savedSnapshot: snapshot,
      };
    }

    case 'select': {
      const expanded = expandToGroups(state.panels, action.ids);
      const ids = action.additive
        ? Array.from(new Set([...state.selection, ...expanded]))
        : expanded;
      return { ...state, selection: ids };
    }

    case 'moveSelection': {
      const groups = translateGroups(state, new Set(state.selection), action.dx, action.dy);
      const panels = mapSelected(state, (rs) =>
        rs.map((r) => ({ ...r, x: r.x + action.dx, y: r.y + action.dy })));
      return withGeometry({ ...state, groups }, panels, action.coalesceKey);
    }

    case 'nudge': {
      const dx = action.dxMm / state.figWmm;
      const dy = action.dyMm / state.figHmm;
      const groups = translateGroups(state, new Set(state.selection), dx, dy);
      const panels = mapSelected(state, (rs) => rs.map((r) => ({ ...r, x: r.x + dx, y: r.y + dy })));
      return withGeometry({ ...state, groups }, panels);
    }

    case 'setRect': {
      const rect = clampMinSize(action.rect, state.figWmm, state.figHmm);
      return withGeometry(
        state,
        state.panels.map((p) => (p.id === action.id ? { ...p, rect } : p)),
        action.coalesceKey,
      );
    }

    case 'groupSelection': {
      if (state.selection.length < 2) return state;
      const kids = state.panels.filter((p) => state.selection.includes(p.id));
      // F5: a plain `Date.now()` id collides whenever two groups are
      // created in the same millisecond — then `ungroup(id)` removes BOTH
      // and every `groups.find(...)` resolves to the first. Match the
      // random-suffix convention already used for coalesce keys elsewhere
      // (Canvas.tsx's drag/resize gesture keys).
      const id = `grp_${Date.now().toString(36)}_${Math.random().toString(36).slice(2)}`;
      const group: Group = {
        id,
        axis: action.axis,
        gutterMm: gutterFromChildren(
          kids.map((p) => p.rect), action.axis, state.figWmm, state.figHmm),
        equal: false,
        rect: groupBounds(kids.map((p) => p.rect)),
      };
      const tagged = state.panels.map((p) =>
        (state.selection.includes(p.id) ? { ...p, group: id } : p));
      // F4: `select` expands a click/marquee to whole groups, so
      // marquee-ing an existing group plus a loose panel is the ordinary
      // way to retag every one of an existing group's members onto the new
      // group, leaving the old group with zero members. Drop any such
      // orphan here rather than letting it persist into figure.json.
      const survivingGroups = state.groups.filter((g) =>
        tagged.some((p) => p.group === g.id));
      const groups = [...survivingGroups, group];
      const panels = resolveGroup({ ...state, panels: tagged }, id, groups);
      return commitGroupChange(state, panels, groups);
    }

    case 'ungroup': {
      // Children keep their solved rects; only the tag and the group go away.
      const panels = state.panels.map((p) => (p.group === action.id ? { ...p, group: null } : p));
      const groups = state.groups.filter((g) => g.id !== action.id);
      return commitGroupChange(state, panels, groups);
    }

    case 'setGutter': {
      const groups = state.groups.map((g) =>
        (g.id === action.id ? { ...g, gutterMm: action.gutterMm } : g));
      const panels = resolveGroup({ ...state, groups }, action.id, groups);
      return commitGroupChange(state, panels, groups);
    }

    case 'setGroupEqual': {
      const groups = state.groups.map((g) =>
        (g.id === action.id ? { ...g, equal: action.equal } : g));
      const panels = resolveGroup({ ...state, groups }, action.id, groups);
      return commitGroupChange(state, panels, groups);
    }

    case 'align':
      if (selectionTouchesGroup(state)) return state;
      return withGeometry(state, mapSelected(state, (rs) => alignRects(rs, action.op, action.ref)));

    case 'distribute':
      if (selectionTouchesGroup(state)) return state;
      return withGeometry(state, mapSelected(state, (rs) => distributeRects(rs, action.axis, action.mode)));

    case 'matchSize':
      if (selectionTouchesGroup(state)) return state;
      return withGeometry(state, mapSelected(state, (rs) => matchSize(rs, action.dim)));

    case 'setBoxMode':
      return { ...state, boxMode: action.mode };

    // A render RESULT is not an edit: it must not dirty the document or
    // enter the undo stack.
    case 'setInk':
      return {
        ...state,
        panels: state.panels.map((p) => (p.id === action.id ? { ...p, ink: action.ink } : p)),
      };

    case 'undo': {
      const history = undo(state.history);
      if (history === state.history) return state;
      return {
        ...state,
        history,
        panels: applySnapshot(state.panels, history.present),
        groups: history.present.groups,
        dirty: !sameLayout(history.present, state.savedSnapshot),
      };
    }

    case 'redo': {
      const history = redo(state.history);
      if (history === state.history) return state;
      return {
        ...state,
        history,
        panels: applySnapshot(state.panels, history.present),
        groups: history.present.groups,
        dirty: !sameLayout(history.present, state.savedSnapshot),
      };
    }

    case 'saved':
      return { ...state, dirty: false, savedSnapshot: state.history.present };

    default:
      return state;
  }
}

type Ctx = { state: EditorState; dispatch: React.Dispatch<Action>; canUndo: boolean; canRedo: boolean };
const EditorCtx = createContext<Ctx | null>(null);

export function EditorProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, initialState);
  const value = useMemo(
    () => ({ state, dispatch, canUndo: canUndo(state.history), canRedo: canRedo(state.history) }),
    [state],
  );
  return <EditorCtx.Provider value={value}>{children}</EditorCtx.Provider>;
}

export function useEditor(): Ctx {
  const ctx = useContext(EditorCtx);
  if (!ctx) throw new Error('useEditor must be used inside <EditorProvider>');
  return ctx;
}
