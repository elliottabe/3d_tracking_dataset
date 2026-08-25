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
import type { Group } from '../layout/groups';
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
  | { type: 'undo' }
  | { type: 'redo' }
  | { type: 'saved' };

export const initialState: EditorState = {
  figWmm: 183, figHmm: 140,
  panels: [], groups: [], selection: [],
  boxMode: 'axes', gridMm: 1, guides: [],
  history: initHistory({}), dirty: false, savedSnapshot: {},
};

const snapshotOf = (panels: PanelState[]): Snapshot =>
  Object.fromEntries(panels.map((p) => [p.id, p.rect]));

const applySnapshot = (panels: PanelState[], snap: Snapshot): PanelState[] =>
  panels.map((p) => (snap[p.id] ? { ...p, rect: snap[p.id] } : p));

/** Commit a new set of panels through history, re-deriving `dirty`. */
function withGeometry(
  state: EditorState, panels: PanelState[], coalesceKey?: string,
): EditorState {
  const history = commit(state.history, snapshotOf(panels), { coalesceKey });
  if (history === state.history) return state;   // nothing actually moved
  return { ...state, panels, history, dirty: !sameLayout(history.present, state.savedSnapshot) };
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

export function reducer(state: EditorState, action: Action): EditorState {
  switch (action.type) {
    case 'load': {
      const panels = action.panels;
      const snapshot = snapshotOf(panels);
      return {
        ...state,
        figWmm: action.figWmm,
        figHmm: action.figHmm,
        panels,
        groups: action.groups,
        selection: [],
        history: initHistory(snapshot),
        dirty: false,
        savedSnapshot: snapshot,
      };
    }

    case 'select': {
      const ids = action.additive
        ? Array.from(new Set([...state.selection, ...action.ids]))
        : action.ids;
      return { ...state, selection: ids };
    }

    case 'moveSelection':
      return withGeometry(
        state,
        mapSelected(state, (rs) =>
          rs.map((r) => ({ ...r, x: r.x + action.dx, y: r.y + action.dy }))),
        action.coalesceKey,
      );

    case 'nudge': {
      const dx = action.dxMm / state.figWmm;
      const dy = action.dyMm / state.figHmm;
      return withGeometry(
        state,
        mapSelected(state, (rs) => rs.map((r) => ({ ...r, x: r.x + dx, y: r.y + dy }))),
      );
    }

    case 'setRect':
      return withGeometry(
        state,
        state.panels.map((p) => (p.id === action.id ? { ...p, rect: action.rect } : p)),
        action.coalesceKey,
      );

    case 'align':
      return withGeometry(state, mapSelected(state, (rs) => alignRects(rs, action.op, action.ref)));

    case 'distribute':
      return withGeometry(state, mapSelected(state, (rs) => distributeRects(rs, action.axis, action.mode)));

    case 'matchSize':
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
