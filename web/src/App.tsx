import { useState } from 'react';
import { getFigure, saveFigure } from './api';
import { Canvas } from './canvas/Canvas';
import { buildSavePayload } from './layout/save';
import { Properties } from './panels/Properties';
import { Toolbar } from './panels/Toolbar';
import { EditorProvider, useEditor } from './state/editorStore';

/**
 * Layout shell: a Toolbar across the top, the pan/zoom Canvas filling the
 * middle, and the Properties panel on the right for the current selection.
 *
 * `onSave` re-fetches the current on-disk document (so anything the editor
 * does not manage — `data`, `spec`, `annotations`, unknown fields — survives
 * untouched), overlays the editor's geometry, group membership, AND the
 * groups array on top of it via the pure `buildSavePayload` (F2 — overlaying
 * rects alone silently discarded every group edit), and PUTs the result.
 * Only on success does it dispatch `saved`; a failure is surfaced, never
 * swallowed.
 */
function EditorShell() {
  const { state, dispatch } = useEditor();
  const [saveError, setSaveError] = useState<string | null>(null);

  async function onSave() {
    setSaveError(null);
    try {
      const original = await getFigure();
      const doc = buildSavePayload(original, state);
      await saveFigure(doc);
      dispatch({ type: 'saved' });
    } catch (e) {
      setSaveError(e instanceof Error ? e.message : String(e));
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      <Toolbar onSave={() => void onSave()} />
      {saveError && (
        <div
          role="alert"
          style={{ background: '#fee2e2', color: '#991b1b', padding: '6px 12px', fontFamily: 'system-ui', fontSize: 13 }}
        >
          Save failed: {saveError}
        </div>
      )}
      <div style={{ display: 'flex', flex: 1, minHeight: 0 }}>
        <div style={{ flex: 1, minWidth: 0, height: '100%' }}>
          <Canvas />
        </div>
        <Properties />
      </div>
    </div>
  );
}

function App() {
  return (
    <EditorProvider>
      <EditorShell />
    </EditorProvider>
  );
}

export default App;
