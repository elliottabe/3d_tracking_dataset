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
 */
import { fracToMm, mmToFrac } from '../layout/rect';
import { useEditor } from '../state/editorStore';
import { formatMm, parseMm } from './format';

const FIELD_LABEL: Record<'x' | 'y' | 'w' | 'h', string> = {
  x: 'x (left, mm)',
  y: 'y (bottom, mm)',
  w: 'w (mm)',
  h: 'h (mm)',
};

export function Properties() {
  const { state, dispatch } = useEditor();
  const sel = state.panels.filter((p) => state.selection.includes(p.id));

  if (sel.length !== 1) {
    return (
      <aside style={{ padding: 12, width: 220, fontFamily: 'system-ui', fontSize: 13 }}>
        <p>{sel.length === 0 ? 'No panel selected' : `${sel.length} panels selected`}</p>
      </aside>
    );
  }

  const p = sel[0];
  const mm = fracToMm(p.rect, state.figWmm, state.figHmm);

  const commit = (k: 'x' | 'y' | 'w' | 'h') => (text: string) => {
    const next = { ...mm, [k]: parseMm(text, mm[k]) };
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
            onBlur={(e) => commit(k)(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') (e.target as HTMLInputElement).blur();
            }}
          />
        </label>
      ))}
    </aside>
  );
}
