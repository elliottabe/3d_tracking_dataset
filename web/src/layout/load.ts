/**
 * Pure, testable conversion from the on-disk/DTO document shape (`FigureDoc`,
 * snake_case, tuple rects) to the editor's in-memory shape (camelCase,
 * `{x,y,w,h}` rects) — the `load` half of the round trip that `save.ts`
 * mirrors for saving.
 *
 * F1 regression: `Canvas.tsx` used to build the `load` action's `panels`
 * inline (correctly, via `tupleToRect`) but hand the raw DTO `groups` array
 * straight through with `as unknown as Group[]` — silencing the type
 * mismatch instead of converting it. `group.gutterMm` came out `undefined`
 * and `group.rect` stayed an array, which crashes the first thing that reads
 * either (`Properties.tsx`'s `formatMm(group.gutterMm)`, `setGutter`,
 * dragging a grouped child). Extracting the conversion here — instead of
 * leaving it inline in the component — makes it testable without jsdom, and
 * gives the real `figures/paper_figures/fig4.json` fixture a path to run
 * through in a unit test.
 */
import type { FigureDoc, PanelSpecDTO } from '../api';
import type { Group } from './groups';
import type { Rect } from './rect';

export type LoadedPanel = {
  id: string;
  type: string;
  rect: Rect;
  group: string | null;
  spec: Record<string, unknown>;
  data: Record<string, unknown>;
};

const tupleToRect = ([x, y, w, h]: [number, number, number, number]): Rect => ({ x, y, w, h });

export function panelFromDto(p: PanelSpecDTO): LoadedPanel {
  return {
    id: p.id, type: p.type, rect: tupleToRect(p.rect),
    group: p.group ?? null, spec: p.spec, data: p.data,
  };
}

export function panelsFromDoc(doc: FigureDoc): LoadedPanel[] {
  return doc.panels.map(panelFromDto);
}

/** Convert one on-disk group (snake_case, tuple rect) to the editor's
 *  `Group` shape (camelCase, object rect) — the actual conversion the
 *  F1 cast was silencing. */
export function groupFromDto(g: FigureDoc['groups'][number]): Group {
  return {
    id: g.id,
    axis: g.axis,
    gutterMm: g.gutter_mm,
    equal: g.equal,
    rect: tupleToRect(g.rect),
  };
}

export function groupsFromDoc(doc: FigureDoc): Group[] {
  return doc.groups.map(groupFromDto);
}
