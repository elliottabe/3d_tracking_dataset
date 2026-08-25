/** Align / distribute / box-mode / undo / save, plus a live layout-warning count. */
import { findCollisions, outOfBounds } from '../layout/overlap';
import { useEditor } from '../state/editorStore';
import type { AlignOp } from '../layout/align';

const ALIGNS: { op: AlignOp; label: string; title: string }[] = [
  { op: 'left', label: '⇤', title: 'Align left' },
  { op: 'hcenter', label: '↔', title: 'Align horizontal center' },
  { op: 'right', label: '⇥', title: 'Align right' },
  { op: 'bottom', label: '⇩', title: 'Align bottom' },
  { op: 'vcenter', label: '↕', title: 'Align vertical center' },
  { op: 'top', label: '⇧', title: 'Align top' },
];

export function Toolbar({ onSave }: { onSave: () => void }) {
  const { state, dispatch, canUndo, canRedo } = useEditor();
  const inked = state.panels.filter((p) => p.ink).map((p) => ({ id: p.id, ink: p.ink! }));

  const selPanels = state.panels.filter((p) => state.selection.includes(p.id));
  // Ungroup is offered only when the selection IS exactly one whole group —
  // guaranteed by the reducer's group-selection expansion (clicking any
  // child selects every sibling), so "every selected panel shares one
  // non-null group" is equivalent to "the whole group is selected".
  const selGroupIds = new Set(selPanels.map((p) => p.group ?? null));
  const ungroupId = selPanels.length > 0 && selGroupIds.size === 1 ? [...selGroupIds][0] : null;

  // Task 5's carried-forward finding: a zero-area ink box (e.g. a panel that
  // hasn't rendered real content yet) can register as an area-0 "collision"
  // in findCollisions. Only surface pairs with a genuine, positive overlap
  // area as warnings — otherwise the badge fires on panels that do not
  // visually overlap anything on screen.
  const collisions = findCollisions(inked).filter((c) => c.area > 0);
  const clipped = outOfBounds(inked);
  const n = state.selection.length;
  const warnCount = collisions.length + clipped.length;

  return (
    <div
      style={{
        display: 'flex', gap: 8, alignItems: 'center', padding: 8,
        borderBottom: '1px solid #ddd', fontFamily: 'system-ui', fontSize: 13,
      }}
    >
      {ALIGNS.map(({ op, label, title }) => (
        <button
          key={op} disabled={n < 2} title={title}
          onClick={() => dispatch({ type: 'align', op, ref: 'selection' })}
        >
          {label}
        </button>
      ))}
      <button
        disabled={n < 3} title="Distribute horizontally (equal gaps)"
        onClick={() => dispatch({ type: 'distribute', axis: 'x', mode: 'gaps' })}
      >
        ⇔
      </button>
      <button
        disabled={n < 3} title="Distribute vertically (equal gaps)"
        onClick={() => dispatch({ type: 'distribute', axis: 'y', mode: 'gaps' })}
      >
        ⇕
      </button>
      <button
        disabled={n < 2} title="Match width"
        onClick={() => dispatch({ type: 'matchSize', dim: 'w' })}
      >
        =w
      </button>
      <button
        disabled={n < 2} title="Match height"
        onClick={() => dispatch({ type: 'matchSize', dim: 'h' })}
      >
        =h
      </button>

      <span style={{ width: 12 }} />
      <button
        disabled={n < 2} title="Group the selection into a row/column"
        onClick={() => dispatch({ type: 'groupSelection', axis: 'x' })}
      >
        group ⇥⇤
      </button>
      <button
        disabled={!ungroupId} title="Ungroup — children keep their current geometry"
        onClick={() => ungroupId && dispatch({ type: 'ungroup', id: ungroupId })}
      >
        ungroup
      </button>

      <span style={{ width: 12 }} />
      <label title="Align by the axes box or by the ink box (which includes tick labels)">
        <input
          type="checkbox" checked={state.boxMode === 'ink'}
          onChange={(e) => dispatch({ type: 'setBoxMode', mode: e.target.checked ? 'ink' : 'axes' })}
        />
        {' '}ink box
      </label>

      <span style={{ width: 12 }} />
      <button disabled={!canUndo} onClick={() => dispatch({ type: 'undo' })}>undo</button>
      <button disabled={!canRedo} onClick={() => dispatch({ type: 'redo' })}>redo</button>

      <span style={{ flex: 1 }} />
      {warnCount > 0 && (
        <span
          style={{ color: '#b45309' }}
          title={[
            ...collisions.map((c) => `${c.a} overlaps ${c.b}`),
            ...clipped.map((id) => `${id} leaves the canvas`),
          ].join('\n')}
        >
          ⚠ {warnCount}
        </span>
      )}
      <button onClick={onSave} disabled={!state.dirty}>
        {state.dirty ? 'Save' : 'Saved'}
      </button>
    </div>
  );
}
