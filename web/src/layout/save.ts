/**
 * Pure construction of the PUT /api/figure payload from editor state — the
 * `save` mirror of `load.ts`.
 *
 * F2 regression: `App.tsx`'s `onSave` re-fetched the on-disk document and
 * overlaid only `panels[].rect` on top of it. `groups` and `panels[].group`
 * were never written back, so any group create/edit/ungroup vanished on the
 * next reload even though `dirty` (derived over the widened `Snapshot` —
 * rects + groups + membership) had already gone false and the toolbar said
 * "Saved". This was inline in the component and therefore untestable
 * without jsdom (no new deps allowed); pulling it out as a pure function
 * makes it directly unit-testable.
 *
 * Both `groups` and `panels[].group` are written together deliberately:
 * `figbuilder/figure.py` (`_parse`) raises "panel … references unknown
 * group" if a panel's `group` names an id the `groups` array doesn't
 * contain, so writing one without the other turns silent data loss into a
 * 400 on the very next save.
 */
import type { FigureDoc, GroupDTO, PanelSpecDTO } from '../api';
import type { EditorState } from '../state/editorStore';

const rectToTuple = (r: { x: number; y: number; w: number; h: number }): [number, number, number, number] =>
  [r.x, r.y, r.w, r.h];

function groupsToDto(groups: EditorState['groups']): GroupDTO[] {
  return groups.map((g) => ({
    id: g.id, axis: g.axis, gutter_mm: g.gutterMm, equal: g.equal,
    rect: rectToTuple(g.rect),
  }));
}

/**
 * `originalDoc` is the just-refetched on-disk document — everything the
 * editor does not manage (`data`, unknown top-level fields, `annotations`)
 * survives untouched. `state` overlays the editor's own geometry
 * (`panels[].rect`), group membership (`panels[].group`), the groups array
 * itself, and each panel's `spec` (options — spines, legend, colours, …) on
 * top of it. `spec` is now editable in the GUI (Task 13), so it must cross
 * this boundary exactly like rect/group did, or a spec edit would round-trip
 * through save+reload as if it had never happened — the same defect class
 * as the F2 group-save bug this function was already written to fix.
 */
export function buildSavePayload(originalDoc: FigureDoc, state: EditorState): FigureDoc {
  const byId = new Map(state.panels.map((p) => [p.id, p]));
  const panels: PanelSpecDTO[] = originalDoc.panels.map((p) => {
    const edited = byId.get(p.id);
    if (!edited) return p;
    return { ...p, rect: rectToTuple(edited.rect), group: edited.group ?? null, spec: edited.spec };
  });
  return { ...originalDoc, panels, groups: groupsToDto(state.groups) };
}
