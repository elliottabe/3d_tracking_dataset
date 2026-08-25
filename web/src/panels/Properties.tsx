/**
 * Numeric x/y/w/h entry in MILLIMETRES for the selected panel.
 *
 * mm, not fractions: a paper figure is specified in mm, and 183 mm
 * double-column is a number the researcher thinks in. State stays in
 * fractions (Rect); this component converts at the boundary with
 * fracToMm/mmToFrac and never stores mm.
 *
 * Field convention: rects are y-UP (figure fractions grow upward from the
 * bottom-left, matching matplotlib). "y" here is therefore the BOTTOM edge
 * of the panel in mm from the bottom of the page, not a screen/top offset —
 * it matches fracToMm(p.rect, ...).y, the same value the rest of the app
 * uses, so what's typed here always agrees with what's stored.
 *
 * A committed edit (blur / Enter) dispatches `setRect`, so it joins the
 * normal undo stack like a drag or an align does.
 *
 * Grouped panels (controller ruling M2-13): the reducer's `select` action
 * expands any grouped-panel selection to the whole group, so a selection
 * that is entirely one group's children lands here as `sel.length > 1`.
 * Per the ruling, an individual grouped child's geometry is NOT offered for
 * numeric editing (the group's solver owns it — a per-child edit would be
 * silently reverted by the next setGutter/setGroupEqual). Instead this
 * renders the group's own controls (gutter, equal sizes) with a note
 * explaining why the x/y/w/h fields aren't here.
 */
import { fracToMm, mmToFrac } from '../layout/rect';
import { useEditor } from '../state/editorStore';
import { formatMm, isValidMmText, parseMm } from './format';

const FIELD_LABEL: Record<'x' | 'y' | 'w' | 'h', string> = {
  x: 'x (left, mm)',
  y: 'y (bottom, mm)',
  w: 'w (mm)',
  h: 'h (mm)',
};

export function Properties() {
  const { state, dispatch } = useEditor();
  const sel = state.panels.filter((p) => state.selection.includes(p.id));

  // The reducer's `select` action expands a click/marquee on ANY grouped
  // panel to that whole group, so "every selected panel shares one
  // non-null group" here means "the whole group is selected" — never a
  // partial/mixed selection that happens to share a group id.
  const groupId = sel.length > 0 ? sel[0].group : null;
  const isWholeGroup = sel.length > 0 && groupId != null && sel.every((x) => x.group === groupId);

  if (isWholeGroup) {
    const group = state.groups.find((g) => g.id === groupId)!;
    return (
      <aside style={{ padding: 12, width: 220, fontFamily: 'system-ui', fontSize: 13 }}>
        <h3 style={{ margin: '0 0 8px' }}>Group ({sel.length} panels)</h3>
        <p style={{ color: '#666', marginBottom: 8 }}>
          This panel is in a group — the group controls its geometry.
          Ungroup to edit x/y/w/h individually.
        </p>
        <label style={{ display: 'flex', gap: 6, marginBottom: 6, alignItems: 'center' }}>
          <span style={{ width: 96 }}>gutter (mm)</span>
          <input
            style={{ width: 70 }}
            defaultValue={formatMm(group.gutterMm)}
            key={`${group.id}-gutter-${formatMm(group.gutterMm)}`}
            onBlur={(e) => {
              // F6: a rejected parse must not leave the field showing text
              // that disagrees with state — this input is uncontrolled and
              // only re-syncs via the `key` above, which does not change
              // when the dispatched value is a no-op fallback.
              if (!isValidMmText(e.target.value)) {
                e.target.value = formatMm(group.gutterMm);
                return;
              }
              dispatch({
                type: 'setGutter', id: group.id, gutterMm: parseMm(e.target.value, group.gutterMm),
              });
            }}
            onKeyDown={(e) => {
              if (e.key === 'Enter') (e.target as HTMLInputElement).blur();
            }}
          />
        </label>
        <label style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          <input
            type="checkbox" checked={group.equal}
            onChange={(e) => dispatch({ type: 'setGroupEqual', id: group.id, equal: e.target.checked })}
          />
          {' '}equal sizes
        </label>
      </aside>
    );
  }

  if (sel.length !== 1) {
    return (
      <aside style={{ padding: 12, width: 220, fontFamily: 'system-ui', fontSize: 13 }}>
        <p>{sel.length === 0 ? 'No panel selected' : `${sel.length} panels selected`}</p>
      </aside>
    );
  }

  const p = sel[0];
  const mm = fracToMm(p.rect, state.figWmm, state.figHmm);

  // F6: a rejected parse (parseMm falling back) must reset the field's
  // displayed text to match state, or the uncontrolled input — which only
  // re-syncs via the `key` below — keeps showing the rejected garbage
  // ("1,5", "abc", a bare "-") indefinitely.
  const commit = (k: 'x' | 'y' | 'w' | 'h') => (e: React.FocusEvent<HTMLInputElement>) => {
    if (!isValidMmText(e.target.value)) {
      e.target.value = formatMm(mm[k]);
      return;
    }
    const next = { ...mm, [k]: parseMm(e.target.value, mm[k]) };
    dispatch({ type: 'setRect', id: p.id, rect: mmToFrac(next, state.figWmm, state.figHmm) });
  };

  return (
    <aside style={{ padding: 12, width: 220, fontFamily: 'system-ui', fontSize: 13 }}>
      <h3 style={{ margin: '0 0 8px' }}>{p.id}</h3>
      <div style={{ color: '#666', marginBottom: 8 }}>{p.type}</div>
      {(['x', 'y', 'w', 'h'] as const).map((k) => (
        <label key={k} style={{ display: 'flex', gap: 6, marginBottom: 6, alignItems: 'center' }}>
          <span style={{ width: 96 }}>{FIELD_LABEL[k]}</span>
          <input
            style={{ width: 70 }}
            defaultValue={formatMm(mm[k])}
            // Remount when the underlying value changes externally (undo,
            // selecting a different panel) so the uncontrolled input's
            // displayed text stays in sync with state.
            key={`${p.id}-${k}-${formatMm(mm[k])}`}
            onBlur={commit(k)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') (e.target as HTMLInputElement).blur();
            }}
          />
        </label>
      ))}
    </aside>
  );
}
