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
 *
 * Task 13 (schema-driven properties form): panel OPTIONS (`spec` — spines,
 * tick labels, legend placement, colours, …) are also editable here, driven
 * entirely by that panel type's JSON Schema (`getPanelTypes()`), never a
 * hand-written list of field names — the architecture rule this restores is
 * "adding a panel type is Python-only, the UI is generated from the
 * schema". `spec` is NOT geometry: unlike x/y/w/h, spec fields stay
 * editable for a panel in a group (the group solver owns rects, not spec),
 * so the spec section is rendered unconditionally for the single selected
 * panel rather than being gated behind the same "not grouped" guard as the
 * geometry fields.
 */
import { useEffect, useState } from 'react';
import { getPanelTypes, type JsonSchemaField, type PanelTypeDTO } from '../api';
import { fracToMm, mmToFrac } from '../layout/rect';
import { type Action, type PanelState, useEditor } from '../state/editorStore';
import { controlFor } from './controlFor';
import { formatMm, isValidMmText, parseMm } from './format';

const FIELD_LABEL: Record<'x' | 'y' | 'w' | 'h', string> = {
  x: 'x (left, mm)',
  y: 'y (bottom, mm)',
  w: 'w (mm)',
  h: 'h (mm)',
};

export function Properties() {
  const { state, dispatch } = useEditor();
  const [panelTypes, setPanelTypes] = useState<Record<string, PanelTypeDTO>>({});

  // Fetched ONCE — the panel-type/schema registry doesn't change while the
  // editor is open ("adding a panel type is Python-only"), so there is
  // nothing to re-fetch on selection changes.
  useEffect(() => {
    let alive = true;
    void getPanelTypes()
      .then((types) => {
        if (alive) setPanelTypes(Object.fromEntries(types.map((t) => [t.id, t])));
      })
      .catch((err: unknown) => {
        console.error('getPanelTypes failed', err);
      });
    return () => { alive = false; };
  }, []);

  const sel = state.panels.filter((p) => state.selection.includes(p.id));

  // The reducer's `select` action expands a click/marquee on ANY grouped
  // panel to that whole group, so "every selected panel shares one
  // non-null group" here means "the whole group is selected" — never a
  // partial/mixed selection that happens to share a group id. Gated on
  // `sel.length > 1` (not `> 0`) so a single selected panel — even a
  // grouped one — always falls through to the single-panel render below,
  // where its `spec` fields must stay reachable regardless of grouping.
  const groupId = sel.length > 0 ? sel[0].group : null;
  const isWholeGroup = sel.length > 1 && groupId != null && sel.every((x) => x.group === groupId);

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

  const panelType = panelTypes[p.type];

  return (
    <aside style={{ padding: 12, width: 220, fontFamily: 'system-ui', fontSize: 13 }}>
      <h3 style={{ margin: '0 0 8px' }}>{p.id}</h3>
      <div style={{ color: '#666', marginBottom: 8 }}>{p.type}</div>
      {p.group ? (
        // Unreachable via the normal click/marquee flow today (`select`
        // expands any grouped click to the whole group, and groups always
        // have 2+ members), kept defensively for a single-member group.
        // Geometry is withheld — the group's solver owns it — but this
        // guard is scoped to x/y/w/h ONLY; the spec section below still
        // renders unconditionally.
        <p style={{ color: '#666', marginBottom: 8 }}>
          This panel is in a group — the group controls its geometry.
          Ungroup to edit x/y/w/h individually.
        </p>
      ) : (
        (['x', 'y', 'w', 'h'] as const).map((k) => (
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
        ))
      )}
      {panelType && (
        <>
          <hr style={{ border: 'none', borderTop: '1px solid #eee', margin: '10px 0' }} />
          <SchemaFields
            panel={p}
            properties={panelType.schema.properties}
            path={[]}
            dispatch={dispatch}
          />
        </>
      )}
    </aside>
  );
}

/** `text`/`number`/`array` values in a nested `spec` path — comma-separated
 *  text for the array case, so a JSON array control never dispatches
 *  per-keystroke. */
function parseNumberList(text: string): number[] {
  return text
    .split(',')
    .map((s) => s.trim())
    .filter((s) => s !== '')
    .map(Number)
    .filter(Number.isFinite);
}

/** Read a possibly-nested value out of a panel's `spec` by key path
 *  (e.g. `['legend', 'loc']`); `undefined` if any step along the way is
 *  missing or not an object. */
function getAtPath(obj: Record<string, unknown>, path: string[]): unknown {
  let cur: unknown = obj;
  for (const key of path) {
    if (typeof cur !== 'object' || cur === null || Array.isArray(cur)) return undefined;
    cur = (cur as Record<string, unknown>)[key];
  }
  return cur;
}

/** Return a NEW spec object with `value` placed at `path`, leaving
 *  everything else — including sibling keys this form doesn't know about —
 *  untouched. Never mutates `obj`, matching the reducer's own immutable-
 *  update convention (`setSpec` replaces the whole object, so every write
 *  here must be a full, fresh `spec` to dispatch). */
function setAtPath(
  obj: Record<string, unknown>, path: string[], value: unknown,
): Record<string, unknown> {
  if (path.length === 0) return obj;
  const [head, ...rest] = path;
  if (rest.length === 0) return { ...obj, [head]: value };
  const child = obj[head];
  const childObj = (typeof child === 'object' && child !== null && !Array.isArray(child))
    ? (child as Record<string, unknown>)
    : {};
  return { ...obj, [head]: setAtPath(childObj, rest, value) };
}

/**
 * Recursively renders one schema `properties` map as controls, driven
 * entirely by `controlFor` — the piece that keeps this generic over ANY
 * panel type's schema rather than a hand-written field list. `path` is the
 * key path within `panel.spec` this level of properties lives at (`[]` at
 * the top, `['legend']` for `legend`'s nested `properties`, …).
 *
 * Each control commits on blur (checkbox/select/colour commit on change,
 * matching how a browser actually fires events for those inputs) by
 * dispatching ONE `setSpec` with the FULL updated spec object — never one
 * dispatch per keystroke, and never a partial patch (`setSpec` replaces the
 * whole `spec`, so every commit here re-derives the whole thing via
 * `setAtPath` on the panel's current spec).
 */
function SchemaFields({
  panel, properties, path, dispatch,
}: {
  panel: PanelState;
  properties: Record<string, JsonSchemaField>;
  path: string[];
  dispatch: React.Dispatch<Action>;
}) {
  const commit = (key: string, value: unknown) => {
    dispatch({ type: 'setSpec', id: panel.id, spec: setAtPath(panel.spec, [...path, key], value) });
  };

  return (
    <>
      {Object.entries(properties).map(([key, field]) => {
        const control = controlFor(field);
        if (control.kind === 'unsupported') return null;

        const valuePath = [...path, key];
        const fieldKey = `${panel.id}-${valuePath.join('.')}`;
        const label = field.title ?? key;
        // Show the schema `default` only for DISPLAY when the spec has no
        // value of its own — a user edit is the only thing that ever
        // writes to `spec` (never the default, on load or on render).
        const raw = getAtPath(panel.spec, valuePath);
        const display = raw !== undefined ? raw : field.default;

        if (control.kind === 'object') {
          return (
            <fieldset
              key={fieldKey}
              style={{ border: '1px solid #e5e5e5', borderRadius: 4, padding: '4px 8px 8px', margin: '0 0 8px' }}
            >
              <legend style={{ fontSize: 12, color: '#666', padding: '0 4px' }}>{label}</legend>
              <SchemaFields panel={panel} properties={control.properties} path={valuePath} dispatch={dispatch} />
            </fieldset>
          );
        }

        return (
          <label
            key={fieldKey}
            style={{ display: 'flex', gap: 6, marginBottom: 6, alignItems: 'center' }}
            title={field.description}
          >
            <span style={{ width: 96 }}>{label}</span>
            {control.kind === 'checkbox' && (
              <input
                type="checkbox"
                checked={Boolean(display)}
                onChange={(e) => commit(key, e.target.checked)}
              />
            )}
            {control.kind === 'number' && (
              <input
                type="number"
                style={{ width: 70 }}
                defaultValue={typeof display === 'number' ? display : ''}
                key={`${fieldKey}-${String(display)}`}
                onBlur={(e) => {
                  if (e.target.value.trim() === '') return;
                  const v = control.integer ? parseInt(e.target.value, 10) : Number(e.target.value);
                  if (Number.isFinite(v)) commit(key, v);
                }}
                onKeyDown={(e) => { if (e.key === 'Enter') (e.target as HTMLInputElement).blur(); }}
              />
            )}
            {control.kind === 'select' && (
              <select
                value={typeof display === 'string' && control.options.includes(display) ? display : ''}
                onChange={(e) => commit(key, e.target.value)}
              >
                <option value="" disabled hidden>—</option>
                {control.options.map((o) => <option key={o} value={o}>{o}</option>)}
              </select>
            )}
            {control.kind === 'color' && (
              <input
                type="color"
                value={typeof display === 'string' && display !== '' ? display : '#000000'}
                onChange={(e) => commit(key, e.target.value)}
              />
            )}
            {control.kind === 'text' && (
              <input
                style={{ width: 120 }}
                defaultValue={typeof display === 'string' ? display : ''}
                key={`${fieldKey}-${String(display)}`}
                onBlur={(e) => commit(key, e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') (e.target as HTMLInputElement).blur(); }}
              />
            )}
            {control.kind === 'numberArray' && (
              <input
                style={{ width: 120 }}
                defaultValue={Array.isArray(display) ? display.join(', ') : ''}
                key={`${fieldKey}-${JSON.stringify(display ?? null)}`}
                onBlur={(e) => commit(key, parseNumberList(e.target.value))}
                onKeyDown={(e) => { if (e.key === 'Enter') (e.target as HTMLInputElement).blur(); }}
              />
            )}
          </label>
        );
      })}
    </>
  );
}
