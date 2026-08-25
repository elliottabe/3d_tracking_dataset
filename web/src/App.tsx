import { Canvas } from './canvas/Canvas';
import { Properties } from './panels/Properties';
import { Toolbar } from './panels/Toolbar';
import { EditorProvider, useEditor } from './state/editorStore';

/**
 * Layout shell: a Toolbar across the top, the pan/zoom Canvas filling the
 * middle, and the Properties panel on the right for the current selection.
 *
 * `onSave` here is a local placeholder (marks the current layout as the new
 * saved point) — there is no backend persistence endpoint yet. A later task
 * wires this to a real PUT /api/figure and this dispatch is expected to be
 * replaced/extended then, not the `saved` action itself.
 */
function EditorShell() {
  const { dispatch } = useEditor();

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      <Toolbar onSave={() => dispatch({ type: 'saved' })} />
      <div style={{ display: 'flex', flex: 1, minHeight: 0 }}>
        <div style={{ flex: 1, minWidth: 0 }}>
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
